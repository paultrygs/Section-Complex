""" 
Name
----
    sSet

Description
-----------
    This package represents (bi)simplicial sets, section and reeb complexes in Python 

Functions
---------
    sset_to_ch(sset_in)
        Converts a simplicial set to the corresponding chain complex over Z_2.
    sset_to_sh(sset_h)
        Converts SimplicialSetWithHeight sset_h to BisimplicialSet s_h (the section complex).
    reeb_complex(sset_h, dim, num_lvls)
        Compute the boundary facemaps of the dim-th Reeb complex (with Z_2-coefficients) of sections traversing num_lvls heights given a SimplicialSetWithHeight
Classes 
-------
    Simplex
        A class used to represent a Simplex. A simplex of dim >= 1 is defined from its faces.
    SimplicialSet
        A class representing a simplicial set. This class keeps track of how simplices form a space
    ChainComplex
        A class representing a chain complex
    SimplexWithHeight
        A class representing a vertex with a simplex of dimension zero with height
    SimplicialSetWithHeight
        A class representing a simplicial set. 
    Bisimplex
        Class representing a Bisimplicial set.
    BisimplicialSet
        Class representing a bisimplicial set consisting of bisimplices    
"""    
import numpy as np
import scipy
import sympy
from itertools import combinations

class Simplex:
    """  
    A class used to represent a Simplex. A simplex of dimension >= 1 is defined from its faces. 
    The dimension of a simplex equals the number of faces - 1. A simplex without faces is a vertex, i.e. of dimension 0.
    Note: class of integers are interpreted as the class of vertices.

    Attributes
    -----------
    faces : list
        list of faces 
    dimension : int 
        dimension of simplex 
        
    Methods
    -------
    face(i)
        Returns the i-th face of the current simplex. 
    """
    def __init__(self, faces=None):
        """
        Parameters
        ----------
        faces : list
            list of faces (of type Simplex) of the form [d_0(self), d_1(self),...,d_n(self)]
        """
        self.faces = []
        self.dimension = 0  
        if faces and isinstance(faces, list):  
            self.faces = faces
            self.dimension = len(self.faces)-1  

    def face(self, i):  
        """
        Get the i-th face of the current simplex. 
        Note: list of faces is assumed preordered.
        
        Parameters
        ----------
        i : int 
            index of face
        
        Returns
        -------
        Simplex
            Simplex assigned as the i-th face
        """
        if isinstance(i, int) and i in range(self.dimension+1):
            return self.faces[i]


class SimplicialSet:
    """
    A class representing a simplicial set, keeping track of how simplices form a space
    Dimension of simplicial set is defined as the maximum dimension of its simplices. 

    Attributes
    -----------
    simplexes : list
        list of simplices 
    dimension : int 
        dimension of simplicial set
        
    Methods
    -------
    is_simplicial_set()
        Checks if the face maps of the simplicial set are well-defined
    add_simplex(simplex)
        Add simplex to simplicial set        
    """
    def __init__(self, simplexes=None):
        """
        Parameters
        ----------
        simplexes : list 
            list of lists of simplices (of type Simplex) starting at 0-simplices, then 1-simplices and so on
        """ 
        self.simplexes = []
        if simplexes:
            if all(isinstance(k, list) for k in simplexes):  # the k-simplices should be given as separate lists
                for x in simplexes:
                    self.simplexes.append(x)
                self.dimension = max(0, len(self.simplexes)-1)  
        else:
            self.dimension = -1  # an empty simplicial set

    def is_simplicial_set(self): 
        """        
        Checks if the face maps of the simplicial set are well-defined
        For each list of i-simplices, check if the j-th simplex is of type Simplex, of dimension i and that its faces are in the simplicial set
        Note: a 0-simplex does not have any faces

        Returns
        ----------
        bool 
            True if face maps are well-defined        
        """ 
        for v in self.simplexes[0]:
            if not isinstance(v, Simplex) or not v.dimension == 0:
                return False 
        for i in range(1, self.dimension+1):  
            s_i = self.simplexes[i]  
            if isinstance(s_i, list):
                for j in range(len(s_i)):
                    s_ij = self.simplexes[i][j]  
                    if isinstance(s_ij, Simplex) and s_ij.dimension == i:  
                        if not all(f in self.simplexes[i-1] for f in s_ij.faces):
                            return False 
                    else:  
                        return False 
        return True 

    def add_simplex(self, simplex):  
        """
        Add Simplex to simplicial set. 
        If simplex.dimension-self.dimension = n <= 0 add simplex to existing list
        Otherwise, attach n-1 empty lists to the simplicial set and add the simplex in the last list
        Note: possible to add a simplex, even if its faces are not present.        

        Parameters
        ----------
        simplex : Simplex
            simplex to be added to set        
        """        
        if isinstance(simplex, Simplex): 
            if self.dimension >= simplex.dimension: 
                self.simplexes[simplex.dimension].append(simplex)
            else:  
                temp_list = [[] for _ in range(simplex.dimension-self.dimension)] 
                temp_list[simplex.dimension-self.dimension-1].append(simplex)  
                self.simplexes.extend(temp_list)  
                self.dimension = simplex.dimension


class ChainComplex:  
    """
    A class representing a chain complex
    
    Attributes
    -----------
    boundary : list
        list of ndarrays corresponding to boundary matrices 
    zero : int 
        zero object 
        
    Methods
    -------
    is_chain_complex()
        Checks if boundary matrices compose to zero
    homology(n)
        Computes the generators of the nth homology of a chain complex with Z_2-coefficients in terms of basis vectors for ker(boundary[n-1])/im(boundary[n]). 
    """    
    def __init__(self, boundary=[], zero = 0):
        """
        Takes a list of numpy-arrays corresponding to the boundary matrices.
        index convention: the i-th boundary matrix bound_i goes from C_(i+1) to C_i
         
        Parameters
        -----------
        boundary : list
            list of ndarrays corresponding to boundary matrices 
        zero : int 
            zero object
        """
        self.boundary = []
        if all(isinstance(bound_i, np.ndarray) for bound_i in boundary):
            self.boundary = boundary
        self.zero = zero

    def is_chain_complex(self):
        """
        Checks if boundary matrices compose to zero, i.e. if all entries in matrix product are zero
        
        Returns
        -----------
        bool
            True when all entries in  boundary compositions are zero            
        """
        for i in range(len(self.boundary)-1):
            d0 = self.boundary[i]
            d1 = self.boundary[i+1]
            if not isinstance(d0, np.ndarray) and not isinstance(d1, np.ndarray):
                return False  
            elif all(np.dot(d0, d1)):  
                return False
        return True

    def homology(self, n):        
        """
        Computes the generators of the nth homology with Z_2-coefficients of a chain complex in terms of basis vectors for ker(boundary[n-1])/im(boundary[n]). 
        Additionally provides a basis for the boundaries, used to determine the homology class in reeb_complex()
        
        First, go through the possible cases of n and find the image and kernel of the nth and n-1th boundary matrices. 
        If n is equal to the degree of the chain complex, the incoming boundary map is zero
        The basis for the kernel of the n-1th boundary map is the orthonormal bases for its null space, found using SVD       
         
        Next, we wish to complete a basis of im to a basis of ker, constructin a matrix im_ker = (im|ker) 
        Obtain indices of pivots of row-reduced echelon form (modulo 2) of im_ker
        Pivot columns of index less than size of im give basis elements for the image of the nth boundary map 
        Pivot columns of index larger than im give basis for the homology (ker/im)

        Note: we need to round due to computational instability in the SVD-algorithm
        Parameters
        -----------
        n : int
            dimension
            
        Returns
        -----------
        list         
            List of vectors, basis of homology generators
        list         
            List of vectors, basis of boundary cycles (image of nth boundary map)
        """
        
        if n < 0 or n > len(self.boundary):  
            ker = []
            im = []
        elif n == 0:
            if len(self.boundary) == 0:
                ker = np.identity(self.zero)
                im = []
            else:
                ker = np.identity(len(self.boundary[n][:, 0])) 
                im = np.transpose(self.boundary[n])  
        elif n == len(self.boundary):
            ker = np.array(sympy.Matrix(self.boundary[n - 1]).nullspace(iszerofunc=lambda x: x % 2==0), dtype = int)
            if len(ker)>0: ker = ker[:,:,0]

            im = []  
        else:
            im = np.transpose(self.boundary[n])
            ker = np.array(sympy.Matrix(self.boundary[n - 1]).nullspace(iszerofunc=lambda x: x % 2==0), dtype = int)
            if len(ker)>0: ker = ker[:,:,0]
        im_ker = []  
        im = np.round(im,1)
        ker = np.round(ker,1) 
        for u in im:  
            im_ker.append(u)
        for v in ker:
            im_ker.append(v)
        pivot_indices = sympy.Matrix(np.transpose(im_ker)).rref(iszerofunc=lambda x: x % 2==0)[1]
        im_basis = []
        hom_basis = []
        for i in range(len(pivot_indices)):
            index = pivot_indices[i]
            index_ker = index - len(im)
            if index_ker >= 0:
                hom = ker[index_ker] / np.min(np.absolute(ker[index_ker])[np.nonzero(ker[index_ker])])
                hom = np.mod(np.rint(hom),2) 
                hom_basis.append(hom)
            else: 
                bas = im[index] / np.min(np.absolute(im[index])[np.nonzero(im[index])])
                bas = np.mod(np.rint(bas),2)
                im_basis.append(bas)
        return [hom_basis, im_basis]         

def sset_to_ch(sset_in): 
    """
    Converts a simplicial set to its assoicated chain complex over Z_2.
    Makes a list [d0, d1,... ,dn] of boundary matrices and gives this as input to the ChainComplex class
    
    Iterate through lists of i-simplices in sset_in
    Initialize boundary maps, bound_i, from (i+1)-simplices, sset_1, to i-simplices, sset_0 of size |sset_0|x|sset_1|
    Iterate through all (i+1)-simplices s_1 and its faces s_0
    s_1 is the j-th simplex and s_0 the k-th face of s_1, so we update bound_i with a (-1)^k in correct entry
    Note: Z_2-coefficients are assumed in current implementation

    Parameters
    ----------
    sset_in : SimplicialSet
        Input simplicial set

    Returns
    -------
    ChainComplex
        chain complex associated with sset_in
    """
    if not isinstance(sset_in, SimplicialSet) or not sset_in.is_simplicial_set(): 
        return None
    bound_mat = []  
    if len(sset_in.simplexes) == 1:
        return ChainComplex(zero= len(sset_in.simplexes[0]))
    else:
        for i in range(len(sset_in.simplexes)-1):
            sset_0 = sset_in.simplexes[i]  
            sset_1 = sset_in.simplexes[i+1] 
            bound_i = np.zeros((len(sset_0), len(sset_1)), dtype=int) 
            for j, s_1 in enumerate(sset_1):  
                for k, s_0 in enumerate(s_1.faces):  
                    bound_i[sset_0.index(s_0)][j] += (-1)**k  
            bound_i = np.mod(np.rint(bound_i),2)
            bound_mat.append(bound_i)
        return ChainComplex(boundary=bound_mat)  

class SimplexWithHeight:
    """
    A class representing a vertex with a simplex of dimension zero with a given height
    A simplex of dim >= 1 is defined from its faces.
    Note: class of integers are to be interpreted as the class of vertices.
    Note: See SimplicialSetWithHeight to check if the height data actually defines a height function

    Attributes
    ----------
    faces : list
        list of faces 
    height : float/int
        height of vertex
    dimension : int 
        dimension of simplex 
    
    Methods
    -------
    face(i)
        returns the i-th face, assuming the face list is sorted in terms of faces    
    """
    def __init__(self, faces=None, height=None):
        """
        The list of faces should have the same length as the numpy array of heights. 
        The exception is vertices with default height zero. A vertex is a simplex with no faces
        The number of faces determines its dimension. An empty Simplex has dimension -1.
        The list of heights must be non-decreasing to qualify as height function

        Parameters
        ----------
        faces : list
            list of faces (simplices) of the form [d_0(self), d_1(self),...,d_n(self)]
        height : ndarray
            numpy array of heights (float/int) The ith entry is the height of the simplex's ith vertex/node
        """
        self.faces = [] 
        self.height = np.array([0])  
        if faces is None:
            self.dimension = 0
            if isinstance(height, list) and len(height) == 1:
                self.height = height
        elif isinstance(faces, list):  
            self.faces = faces
            self.dimension = len(self.faces)-1
            if isinstance(height, list) and all(np.diff(height) >= 0):
                self.height = height

    def face(self, i):  
        """
        Returns the i-th face, assuming the face list is sorted in terms of faces
        
        Returns
        -------
        Simplex
            i-th face        
        """        
        if isinstance(i, int) and i in range(self.dimension+1):
            return self.faces[i]

class SimplicialSetWithHeight:
    """
    A class representing a simplicial set with heights.
    This class keeps track of how simplices form a space
    
    Attributes
    -----------
    simplexes : list
        List of simplices with heights starting at 0-simplices, then 1-simplices and so on
    dimension : int 
        Dimension of simplicial set
        
    Methods
    -------
    is_simplicial_set()
        Checks if the face maps of the simplicial set are well-defined
    is_height_function()
        Checks if height data on simplices defines a well-defined height function on the simplicial set
    add_simplex(simplex)
        Add SimplexWithHeight to simplicial set. 
    """    
    def __init__(self, *simplexes):  
        """
        The dimension of the simplicial set is given as the maximum dimension of its simplices. An empty simplicial set has dimension -1
        
        Parameters
        ----------
        simplexes : list 
            list of lists, one for each set of k-simplices (of type SimplexWithHeight) pre-ordered by increasing k 
        """
        self.simplexes = []
        if all(isinstance(k, list) for k in simplexes):  
            for x in simplexes:
                self.simplexes.append(x)
            self.dimension = max(0, len(self.simplexes)-1)
        else:
            self.dimension = -1  

    def is_simplicial_set(self): 
        """        
        Checks if the face maps of the simplicial set are well-defined
        For each list of i-simplices, check if the j-th simplex is of type SimplexWithHeight, of dimension i and that its faces are in the simplicial set
        Note: a 0-simplex does not have any faces

        Returns
        ----------
        bool 
            True if face maps are well-defined        
        """ 
        for v in self.simplexes[0]:
            if not isinstance(v, SimplexWithHeight) or not v.dimension == 0:
                return False 
        for i in range(1, self.dimension+1):  
            s_i = self.simplexes[i]  
            if isinstance(s_i, list):
                for j in range(len(s_i)):
                    s_ij = self.simplexes[i][j]  
                    if isinstance(s_ij, SimplexWithHeight) and s_ij.dimension == i:  
                        if not all(f in self.simplexes[i-1] for f in s_ij.faces):
                            return False 
                    else:  
                        return False 
        return True 

    def is_height_function(self): 
        """
        Checks if height data on simplices defines a well-defined height function on the simplicial set
        Iterate over all lists of i-simplices, s_i, in increasing manner. 
        For the j-th i-simplex, s_ij, the heights of its d_f-differential should be equal to the height of its f-th face  
        Note: list of heights is already required to be non-decreasing

        Returns
        ----------
        bool 
            True if consistent heights of faces and differentials
        """
        for i in range(1, self.dimension+1):  
            s_i = self.simplexes[i]  
            if isinstance(s_i, list):
                for j in range(len(s_i)):
                    s_ij = self.simplexes[i][j] 
                    for f in range(len(s_ij.faces)):
                        if list(np.delete(s_ij.height, f)) != s_ij.faces[f].height:
                            return False
        return True  

    def add_simplex(self, simplex):  
        """
        Add SimplexWithHeight to simplicial set. 
        If simplex is a vertex, add empty list
        If simplex.dimension-self.dimension = n <= 0 add simplex to existing list
        Otherwise, attach n-1 empty lists to the simplicial set and add the simplex in the last list
        Note: possible to add a simplex, even if its faces are not present.        

        Parameters
        ----------
        simplex : Simplex
            simplex to be added to set        
        """        
        if isinstance(simplex, SimplexWithHeight): 
            if self.dimension == 0:
                self.simplexes.append([])
            if self.dimension >= simplex.dimension: 
                self.simplexes[simplex.dimension].append(simplex)
            else:  
                temp_list = [[] for _ in range(simplex.dimension-self.dimension)] 
                temp_list[simplex.dimension-self.dimension-1].append(simplex)  
                self.simplexes.extend(temp_list)  
                self.dimension = simplex.dimension

class Bisimplex:
    """
    Class representing a Bisimplex
    A bisimplex has both horizontal- and vertical faces
    
    Attributes
    ----------
    vertical : list
        list of vertical faces
    horizontal : list
        list of horizontal faces
    dimension : int
        dimension of bisimplicial set

    Methods
    -------
    horizontal_face(i)
        Returns the i-th horizontal face, assuming the face list is sorted in terms of faces
    vertical_face(i)
        Returns the i-th vertical face, assuming the face list is sorted in terms of faces
    """    
    def __init__(self, vertical=None, horizontal=None):
        """       
        Dimension is given in both directinos as max dimension of corresponding simplexes. A vertex has dimension 0 

        Parameters
        ----------
        vertical : list
            list of vertical faces of type Bisimplex. 
            Should be of the form [d_0(self), d_1(self),...,d_n(self)]
        horizontal : list
            list of horizontal faces of type Bisimplex.
            Should be of the form [d_0(self), d_1(self),...,d_n(self)]
        """
        self.vertical = []
        self.horizontal = []
        if isinstance(vertical, list):
            self.vertical = vertical
        if isinstance(horizontal, list):
            self.horizontal = horizontal
        self.dimension = [max(len(self.horizontal)-1, 0), max(len(self.vertical)-1, 0)]  

    def horizontal_face(self, i):
        """       
        Returns the i-th horizontal face, assuming the face list is sorted in terms of faces
        
        Parameters
        ----------
        i : int
            Bisimplex face index

        Returns
        -------
        Bisimplex
            i-th horizontal face
        """
        
        if isinstance(i, int) and i in range(self.dimension[0]+1):
            return self.horizontal[i]

    def vertical_face(self, i):
        """       
        Returns the i-th vertical face, assuming the face list is sorted in terms of faces
        
        Parameters
        ----------
        i : int
            Bisimplex face index

        Returns
        -------
        Bisimplex
            i-th vertical face
        """
        if isinstance(i, int) and i in range(self.dimension[1]+1):
            return self.vertical[i]


class BisimplicialSet:
    """
    Class representing a bisimplicial set
    
    Attributes
    ----------
    bisimplexes : list
        list of bisimplices
    dimension : list
        dimension of bisimplicial set

    Methods
    -------
    vertical_simplicial_set(n)
        Picks out the n-th vertical simplicial set
    horizontal_simplicial_set(n)
        Picks out the n-th vertical simplicial set
    is_simplicial_vertical()
        Checks if the vertical simplicial sets are simplicial
    is_simplicial_horizontal()
        Checks if the horizontal simplicial sets are simplicial
    add_bisimplex()
        Add bisimplex to bisimplicial set
    """    
    def __init__(self, bisimplexes=None):  
        """
        Dimension is given in both directions as max dimension of corresponding simplexes. 
        An empty bisimplicial set has dimensions (-1,-1)
        Note: len(self.bisimplexes[i]) is assumed consistent for all i

        Parameters
        ----------
        bisimplexes : list 
            bisimplexes[i][j] is a list of (i,j)-simplexes
        """        
        if bisimplexes:  
            self.bisimplexes = bisimplexes
            verticaldim = max(0, len(self.bisimplexes)-1)
            horizontaldim = max(0, len(self.bisimplexes[0])-1)               
            self.dimension = np.array([verticaldim, horizontaldim], int)
        else:
            self.bisimplexes = []
            self.dimension = np.array([-1, -1], int)  

    def vertical_simplicial_set(self, q):
        """
        Pick out the q-th vertical simplicial set
        First, instantiate output simplicial set, sset_out
        Iterate over all dimensions, p, in the horizontal direction and get the p-bisimplices of the q-th vertical level
        For each such bisimplex, make a list of its horizontal faces, create a simplex based on these and insert into sset_out
        bisimplex_to_simplex remembers which simplices correspond to which bisimplices

        Parameters
        ----------
        q : int 
            vertical index
        
        Returns
        -------
        SimplicialSet
            q-th vertical simplicial set
        dictionary
            dictionary giving the bisimplexes of the q-th vertical simplicial set, indexed with simplices 
        """        
        if q in range(self.dimension[1]+1):
            sset_out = SimplicialSet()
            bisimplex_to_simplex = {}  
            for p in range(self.dimension[0]+1):  
                bisimps = self.bisimplexes[p][q]  
                for bisimp in bisimps: 
                    faces = []
                    for face in bisimp.horizontal:
                        faces.append(bisimplex_to_simplex[face])
                    bisimplex_to_simplex[bisimp] = Simplex(faces)  
                    sset_out.add_simplex(bisimplex_to_simplex[bisimp])
            return [sset_out, bisimplex_to_simplex]
        return [SimplicialSet(),{}]  


    def horizontal_simplicial_set(self, p):
        """
        Pick out the p-th horizontal simplicial set
        First, instantiate output simplicial set, sset_out
        Iterate over all dimensions, q, in the vertical direction and get the p-bisimplices of the p-th horizontal level
        For each such bisimplex, make a list of its vertical faces, create a simplex based on these and insert into sset_out
        bisimplex_to_simplex remembers which simplices correspond to which bisimplices

        Parameters
        ----------
        p : int 
            horizontal index
        
        Returns
        -------
        SimplicialSet
            p-th horizontal simplicial set
        dictionary
            dictionary giving the bisimplexes of the p-th horizontal simplicial set, indexed with simplices 
        """        
        if p in range(self.dimension[0]+1):
            sset_out = SimplicialSet()
            bisimplex_to_simplex = {}  
            for q in range(self.dimension[1]+1):  
                bisimps = self.bisimplexes[p][q]  
                for bisimp in bisimps: 
                    faces = []
                    for face in bisimp.vertical:
                        faces.append(bisimplex_to_simplex[face])
                    bisimplex_to_simplex[bisimp] = Simplex(faces)  
                    sset_out.add_simplex(bisimplex_to_simplex[bisimp])
            return [sset_out, bisimplex_to_simplex]
        return [SimplicialSet(),{}]  


    def is_simplicial_vertical(self):  
        """        
        Checks if the vertical simplicial sets are simplicial sets
        
        Returns
        ----------
        bool 
            True if simplicial set
        """ 
        for i in range(self.dimension[1]+1):
            sset_i = self.vertical_simplicial_set(i)[0]
            if not sset_i.is_simplicial_set(): 
                return False
        return True

    def is_simplicial_horizontal(self):  
        """        
        Checks if the horizontal simplicial sets are simplicial sets
        
        Returns
        ----------
        bool 
            True if simplicial set        
        """ 
        for j in range(self.dimension[0]+1):
            sset_j = self.horizontal_simplicial_set(j)[0] 
            if not sset_j.is_simplicial_set(): 
                return False

    def add_bisimplex(self, bisimplex):           
        """
        Add Bisimplex to bisimplicial set
        If the dimensions of the bisimplex in both directions are less than those of the bisimplicial set, add bisimplex to existing list
        Otherwise, first attach empty lists corresponding to the difference in horizontal dimensions to the bisimplicial set 
        Next, for each horiztonal direction, add empty lists corresponding to the difference in vertical dimensions to the bisimplicial set  
        Lastly, update dimension of bisimplicial set and add bisimplex in the last list
        Note: possible to add a bisimplex, even if its faces are not present
        
        Parameters
        ----------
        bisimplex : Bisimplex
            Bisimplex to be added to set        
        """        
        if isinstance(bisimplex, Bisimplex):  
            if self.dimension[0] >= bisimplex.dimension[0] and self.dimension[1] >= bisimplex.dimension[1]:
                self.bisimplexes[bisimplex.dimension[0]][bisimplex.dimension[1]].append(bisimplex)
            else:
                hor = [[] for _ in range(bisimplex.dimension[0] - self.dimension[0])]    
                self.bisimplexes.extend(hor)   
                for i in range(self.dimension[0]+1):  
                    ver = [[] for _ in range(bisimplex.dimension[1] - self.dimension[1])]  
                    self.bisimplexes[i].extend(ver)  
                    del ver
                verticaldim = max(self.dimension[1] + 1, bisimplex.dimension[1] + 1) 
                for i in range(self.dimension[0]+1, len(self.bisimplexes)):  
                    ver = [[] for _ in range(verticaldim)]
                    self.bisimplexes[i].extend(ver)   
                    del ver
                self.dimension[0] = len(self.bisimplexes)-1         
                self.dimension[1] = len(self.bisimplexes[0])-1
                self.bisimplexes[bisimplex.dimension[0]][bisimplex.dimension[1]].append(bisimplex)

def sset_to_sh(sset_h):
    """
    Converts SimplicialSetWithHeight sset_h to BisimplicialSet s_h (the section complex).
    Loop over all simplices s of sset_h, create bisimplex bs by sorting all faces of s and insert bs into s_h
    horizontal faces of bs correspond to faces of s going between fewer heights than s, and are ordered by the complementary height in s
    vertical faces of bs correspond to faces of s going between same heights as s
    simplex_to_bisimplex encodes the map s <-> bs 

    Parameters
    ----------
    sset_h : SimplicialSetWithHeight
        Input simplicial set
    
    Returns
    ----------
    BisimplicialSet
        Output bisimplicial set 
    dictionary
        Dictionary saving map between simplices of sset and corresponding bisimplices of s_h
    """
    if not isinstance(sset_h, SimplicialSetWithHeight) or not sset_h.is_simplicial_set():  
        return None    
    s_h = BisimplicialSet()
    simplex_to_bisimplex = {} 
    for s_i in sset_h.simplexes:
        for s in s_i:
            s_heights = set(s.height)
            s_heights_sorted = sorted(list(s_heights))
            num_heights = len(s_heights) 
            vertical_faces = []
            horizontal_faces = [] if num_heights == 0 else [0] * num_heights
            for s_face in s.faces: 
                f_heights = set(s_face.height)
                if len(f_heights)==num_heights: 
                    vertical_faces.append(simplex_to_bisimplex[s_face])  
                else:
                    ind_height = s_heights_sorted.index(list(s_heights.difference(f_heights))[0])
                    horizontal_faces[ind_height] = simplex_to_bisimplex[s_face]                     
            bs = Bisimplex(vertical=vertical_faces,horizontal=horizontal_faces)
            simplex_to_bisimplex[s] = bs
            s_h.add_bisimplex(bs)
    return [s_h, simplex_to_bisimplex]

def homology_vector_space(homology_basis, boundaries):
    """
    Computes all vector representatives in homology vector space 

    Parameters
    ----------
    homology_basis : list
        Input list of basis vectors in homology 
    boundaries : list
        Boundaries given by image of boundary map of one dimension higher

    Returns
    ----------
    list
        list of all vectors in vector space
    list
        indices corresponding to basis vectors 
    """
    homology_reps, homology_inds = [], []
    cycles = np.array([v for v in homology_basis])
    boundaries = np.array([v for v in boundaries])
    if np.sum(np.shape(cycles))>0: 
        if cycles.ndim == 1:
            cycles = cycles[np.newaxis, :]
        for num_cycles in np.arange(1, len(cycles[:,0])+1):
            combs = combinations(range(len(cycles[:,0])), num_cycles) 
            for comb in combs:
                homology_inds.append(comb)
                homology_comb = np.sum(cycles[comb,:],0)
                homology_comb = np.mod(np.rint(homology_comb),2)
                homology_reps.append(homology_comb)        

    if np.sum(np.shape(boundaries))>0: 
        if boundaries.ndim == 1:
            boundaries = boundaries[np.newaxis, :]
        for ind_hom, homology_rep in enumerate(homology_basis):            
            for num_bound in np.arange(1, len(boundaries[:,0])+1):
                combs = combinations(range(len(boundaries[:,0])), num_bound) 
                for comb in combs:
                    homology_inds.append(ind_hom)
                    homology_comb = homology_rep + np.sum(boundaries[comb,:],0)
                    homology_comb = np.mod(np.rint(homology_comb),2)
                    homology_reps.append(homology_comb)
    return homology_reps, homology_inds

def reeb_facemap(homology_l0, homology_l1, bisimplex_to_simplex_l0, bisimplex_to_simplex_l1,
    simplex_dim_l0, simplex_dim_l1, homology_reps_l0, homology_inds_l0, lvl):
    """
    Auxiliary function for reeb_complex()
    Compute lvl+1 facemaps between homology representatives going between heights lvl and lvl-1 in reeb complex 
    First computes lvl+1 facemaps between simplices going between heights lvl and lvl-1 in section complex 
    Then matches the faces of homology basis of lvl (fmap_hom_temp) with all vectors of homology vector space lvl-1

    Parameters
    ----------
    homology_l0 : list
        homology basis vectors of heights lvl-1
    homology_l1 : list
        homology basis vectors of heights lvl
    bisimplex_to_simplex_l0 : dictionary 
        map between bisimplices of section space and simplices of horizontal simplicial set lvl-1
    bisimplex_to_simplex_l1 : dictionary
        map between bisimplices of section space and simplices of horizontal simplicial set lvl
    simplex_dim_l0 : 
        simplices of dimension dim going between heights lvl-1
    simplex_dim_l1 : 
        simplices of dimension dim going between heights lvl
    homology_reps_l0 : 
        homology vector space representatives 
    homology_inds_l0 : list
        basis vector indices corresponding to vector space representatives 
    lvl : int
        number heights

    Returns
    ----------
    list
        list of facemaps  
    """
    homology_reps_l1 = [v for v in homology_l1]          
    bisimplex_l1 = list(bisimplex_to_simplex_l1.keys())
    simplex_l1 = list(bisimplex_to_simplex_l1.values())
    fmap_reeb = []    
    for ind_face in range(lvl+1): 
        fmap_lvl = np.zeros([len(simplex_dim_l0), len(simplex_dim_l1)])
        for ind_l1 in range(len(simplex_dim_l1)): 
            ind_s_l1 = simplex_l1.index(simplex_dim_l1[ind_l1])
            bs_l1 = bisimplex_l1[ind_s_l1].horizontal_face(ind_face)
            if isinstance(bs_l1, Bisimplex): 
                ind_l0 = simplex_dim_l0.index(bisimplex_to_simplex_l0[bs_l1])
                fmap_lvl[ind_l0][ind_l1] = 1      

        fmap_hom_lvl = np.zeros([len(homology_l0), len(homology_l1)])   
        fmap_hom_temp = np.matmul(fmap_lvl, np.transpose(homology_reps_l1)).T
        fmap_hom_temp = np.mod(np.rint(fmap_hom_temp),2) 
        for i, fmap_basis_l1 in enumerate(fmap_hom_temp):
            for j, hom_basis_l0 in enumerate(homology_reps_l0):
                if np.sum(np.abs(fmap_basis_l1-hom_basis_l0)) == 0: 
                    fmap_hom_lvl[homology_inds_l0[j],i] = 1
        fmap_reeb.append(fmap_hom_lvl)
    return fmap_reeb

def reeb_complex(sset_h, dim, num_lvls):
    """
    Compute the boundary facemaps of the dim-th Reeb complex (with Z_2-coefficients) of sections traversing num_lvls heights given a SimplicialSetWithHeight 
    Note: degenerate faces are omitted but they are irrelevant in homology anyways
    1. Translate sset_h to bisimplicial set s_h. 
    2. For each number of height levels, obtain horizontal simplicial sets going between lvl and lvl-1 heights (sset_l1 and sset_l0).
    3. Translate horizontal simplicial sets to chain complex and compute homology
    4. Find all vectors in homology vector space of lvl-1 
    5. Iterate over all sublevels and make boundary facemaps between lvl and lvl-1 and append to reeb facemaps

    Parameters
    ----------
    sset_h : SimplicialSetWithHeight
        simplicial set to be analysed
    dim : int
        homology dimension
    num_lvls : int
        number of height levels  
    
    Returns
    ----------
    list
        list of lists with the horizontal facemaps in homology
    """    
    facemaps = []     
    s_h = sset_to_sh(sset_h)[0] 
    sset_l0, bisimplex_to_simplex_l0 = s_h.horizontal_simplicial_set(0)
    homology_l0, boundaries_l0 = sset_to_ch(sset_l0).homology(dim)
    simplex_dim_l0 = sset_l0.simplexes[dim]
    for lvl in range(1, num_lvls+1):         
        sset_l1, bisimplex_to_simplex_l1  = s_h.horizontal_simplicial_set(lvl)
        homology_l1, boundaries_l1 = sset_to_ch(sset_l1).homology(dim)
        simplex_dim_l1 = sset_l1.simplexes[dim]
        homology_reps_l0, homology_inds_l0 = homology_vector_space(homology_l0, boundaries_l0)
        facemaps.append(reeb_facemap(homology_l0, homology_l1, bisimplex_to_simplex_l0, bisimplex_to_simplex_l1,
            simplex_dim_l0, simplex_dim_l1, homology_reps_l0, homology_inds_l0, lvl))
        sset_l0, bisimplex_to_simplex_l0, homology_l0, boundaries_l0, simplex_dim_l0 = (sset_l1, bisimplex_to_simplex_l1, 
            homology_l1, boundaries_l1, simplex_dim_l1)
    return facemaps