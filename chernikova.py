# Copyright 2021 Andrés Goens <andres.goens@barkhauseninstitut.org>

#This implements Chernikova's algorithm as described in:
#Felipe Fernandez, Patrice Quinton. Extension of Chernikova’s algorithm for solving general mixed linear programming problems. [Research Report] RR-0943, 1988

#with the extension described in:
#Hervé Le Verge. A Note on Chernikova’s algorithm. [Research Report] RR-1662, INRIA. 1992. inria-00074895

import numpy as np
from typing import List

def noop(*args):
    pass
#debug = noop
#debug = print

# a x >= b
class Constraint():
    def __init__(self,coeff : np.ndarray, inhomogeneity : np.float64 = 0,
             equality = False):
        self.n = coeff.shape[0]
        assert coeff.shape == (self.n,)
        self.a = coeff
        self.b = inhomogeneity
        self.eq = equality

    def is_saturated_by_y(self,y : np.ndarray) -> bool:
        return np.allclose(a*y,np.zeroes(self.n))

    def is_equality(self) -> bool:
        return self.eq

    def __repr__(self):
        res = ""
        for i in range(self.n):
            if self.a[i] != 0:
                if len(res) != 0:
                    res += "+ "
                if self.a[i] != 1:
                    res += f"{self.a[i]} * "
                res += f"x_{i+1} "
        if self.eq:
            res += f"= {self.b}"
        else:
            res += f">= {self.b}"
        return res

#This class is used to return rays,
#not internally for Chernikova's algorithm.
class Generator():

    def __init__(self,vector : np.ndarray, vertex = False, bidirectional = False):
        self.value = vector #.copy()
        self.vertex = vertex
        self.bidirectional = bidirectional

    def is_unidirectional_ray(self) -> bool:
        if vertex:
            return False
        else:
            return not self.bidirectional

    #Line
    def is_bidirectional_ray(self) -> bool:
        if vertex:
            return False
        else:
            return self.bidirectional

    def is_vertex(self) -> bool:
        return self.vertex

    def get_value(self) -> np.ndarray:
        return self.vertex.copy()

    def __repr__(self):
        if self.vertex:
            res = "Vertex("
        elif self.bidirectional:
            res = "Line("
        else:
            res = "Ray("
        res += str(self.value)
        res += ")"
        return res

class Cone():
    def __init__(self, constraints : List[Constraint]):
        if len(constraints) == 0:
            self.n = 0
        else:
            self.n = constraints[0].n
        eqs = []
        ineqs = []
        b_eqs = []
        b_ineqs = []

        for c in constraints:
            if c.is_equality():
                eqs.append(c.a)
                b_eqs.append(c.b)
            else:
                ineqs.append(c.a)
                b_ineqs.append(c.b)


        self.B = np.array(eqs)
        self.D = np.array(ineqs)
        self.B_b = np.array(b_eqs)
        self.D_b = np.array(b_ineqs)


    def to_uniform_system(self) -> np.ndarray:
        B_hom = homogenize_system(self.B,self.B_b)
        D_hom = homogenize_system(self.D,self.D_b)

        top = np.hstack((np.atleast_2d(np.zeros(B_hom.shape[0])).T,B_hom))
        bot = np.hstack((np.array([[1]]*D_hom.shape[0]),D_hom))

        if top.shape[1] == 1:
            #trivial case: empty uniform system
            if bot.shape[1] == 1:
                #TODO: fix this (this will fail)
                return np.array([[]])
            else:
                A_constraints = bot.T
        elif bot.shape[1] == 1:
            A_constraints = top.T
        else:
            A_constraints = np.vstack((top,bot)).T

        #add lambda constraint (described as e) in Section 8 of the paper)
        lambda_constraint = np.zeros(A_constraints.shape[0])
        lambda_constraint[0] = 1
        lambda_constraint[A_constraints.shape[0]-1] = 1

        A = np.hstack((A_constraints,np.atleast_2d(lambda_constraint).T))
        return A

    def to_initial_chernikova_tableau(self) -> np.ndarray:
        A = self.to_uniform_system()
        R = np.identity( (self.n+2) )
        return np.hstack((R,A))

    #Chernikova's algorithm
    #Returns a set of irredundant extremal rays
    #or associated minimal proper faces generating the
    #cone.
    def generators(self) -> List[Generator]:
        tableau = self.to_initial_chernikova_tableau()
        #debug(f"initial tableau: \n {tableau} \n =============== \n")
        constraints = range(self.n+2,tableau.shape[1])
        for constraint in constraints:
            tableau = chernikova_iteration(tableau, constraint, self.n)
            tableau = chernikova_reduction(tableau, self.n)
            #debug(f" after constraint {constraint}: {decode_rays(tableau[1:,:self.n+2])} \n")
        #don't include first row (mu)
        rays = tableau[1:,:self.n+2]
        return decode_rays(rays)

def S(y : np.ndarray, n : int) -> List[int]:
    projections = y[n+2:]
    return np.argwhere(projections == 0)

def decode_rays(rays_tableau : np.ndarray) -> List[Generator]:
    rays = []
    #debug(f"decoding ray tableau: \n {rays_tableau}")
    for ray_idx in range(rays_tableau.shape[0]):
        bidirectional = np.isclose(rays_tableau[ray_idx,0],0)
        ray = dehomogenize_ray(rays_tableau[ray_idx,1:])
        if not ray.is_vertex():
            if bidirectional:
                ray.bidirectional = True
        rays.append(ray)
    return rays

def row_lin_combination_satisfying_constraint(row1 : np.ndarray, p1 : np.int64,
                                              row2 : np.ndarray, p2 : np.int64) -> np.ndarray:

    #bidirectional?
    if np.isclose(row1[0],0) or np.isclose(row2[0],0):
        #row1 is unidirectional
        if np.isclose(row1[0],1):
            #if projections are oposite, then unidirectional rows are as well
            row2 = -np.sign(p1*p2) * row2
            #make unidirectional
            row2[0] = 1
            #debug(f"made row2 unidirectional: {row2}")
        elif np.isclose(row2[0],1):
            #same as above, but reversed
            row1 = -np.sign(p1*p2) * row1
            row1[0] = 1
            #debug(f"made row1 unidirectional: {row1}")
        #both are bidirectional
        else:
            row1 = -np.sign(p1*p2)*row1
            #debug(f"both bidirectional: {row1}")
            #row2 = row2 #not needed (noop)

    res = row1 * -p2 + row2 * p1
    #debug(f"linear combination:  {-p2} * {row1} + {p1} * {row2} = {res}")
    if res.dtype == np.int64:
        gcd = np.gcd.reduce(res)
        res = res / gcd

    return res

#This computes the new tableau without removing irredundant rays
def chernikova_iteration(tableau : np.ndarray, column : int,
                         n : int, leverge: bool = True) -> np.ndarray:
    ineq = np.isclose(tableau[0,column],1)
    # From the end of Section 8:
    # "The coefficients of the matrix A.T represent the projection of the
    # corresponding oriented row ray vector (the positive part if it is a
    # bidirectional ray) on the corresponding oriented column hyperplane
    #(the positive part if it is bidirectional (or an equation))."
    projections = tableau[:,column]
    #debug(f"projections {projections}")

    #keep first row (mu)
    conserved_rays = [0]
    #constraint is equation (corresp. to half-space)
    if ineq:
        #keep first row (mu)
        for i in range(1,tableau.shape[0]):
            #H^1 : projections are non-negative
            #or ray is bidirectional
            #debug(f" (eqn) proj: {projections[i]}, tab[0,i]: {tableau[0,i]}")
            if projections[i] >= 0 or np.isclose(tableau[0,i],0):
                conserved_rays.append(i)
    #constraint is equation (corresp. to hyper-plane)
    else:
        #keep first row (mu)
        for i in range(1,tableau.shape[0]):
            #H^0: projections are null
            if np.isclose(projections[i],0):
                conserved_rays.append(i)

    #debug(f"conserved row indices: {conserved_rays}")
    new_tableau_rows = []
    for idx in conserved_rays:
        row = tableau[idx,:].copy()

        #if bidirectional, change
        if np.isclose(row[0],0):
            if projections[idx] > 0:
                #make unidirectional
                row[0] = 1
                #debug(f"making undirectional: {row}")

            elif projections[idx] < 0:
                #oposite unidirectional
                row = -row
                row[0] = 1
                #debug(f"oposite undirectional: {row}")
            #projection == 0 is kept bidirectional
            #else:
            #    #debug(f"kept bidirectional: {row}")
        new_tableau_rows.append(row)
    #debug(f"conserved rays (including directionality): \n {new_tableau_rows}")
    #TODO: do I have to consider vertices separately here?

    #new rays
    #don't count first row (mu)
    for i in range(1,n+2):
        #don't iterate twice over pairs
        for j in range(i+1,n+2):
            #each ray belongs to a different side of the hyperplane

            p1 = projections[i]
            p2 = projections[j]
            row1 = tableau[i,:]
            row2 = tableau[j,:]
            if p1*p2 < 0 or ((np.isclose(row1[0],0) or np.isclose(row2[0],0)) and not np.isclose(p1*p2,0)):
                new_row = row_lin_combination_satisfying_constraint(row1,p1,row2,p2)
                if leverge:
                    S_set = S(new_row,n)
                    #debug(f"S set: {S_set}")
                    if len(S_set) <= n - 3:
                        #debug(f"leverge: skipping {new_row}")
                        continue
                new_tableau_rows.append(new_row)
    return np.array(new_tableau_rows)

def chernikova_reduction(tableau : np.ndarray, n : int) -> np.ndarray:
    #See the end of Section 8 (quoted above)
    #for projections: rows = rays, columns = constraints
    projections = tableau[:,n+2:tableau.shape[1]]

    #debug(tableau)
    #ignore first (mu)
    ray_directionalities = tableau[1:,0]

    #add 1 because of ignored (mu)
    unidirectional_rays = np.nonzero(ray_directionalities)[0]+1
    #debug(f"unidirectional rays: {unidirectional_rays}")

    to_remove = set()
    for ray1 in unidirectional_rays:
        for ray2 in unidirectional_rays:
            if ray1 == ray2:
                continue
            remove = True
            for constraint in range(n+2,tableau.shape[1]):
                c_idx = constraint - n - 2
                #debug(f"constraint {constraint}, r1 {ray1}, r2 {ray2}")
                two_projections = set([projections[ray1,c_idx],
                                       projections[ray2,c_idx]])
                #debug(f" two projections: {two_projections}")
                if 0 in two_projections:
                    two_projections.remove(0)
                    if len(two_projections) == 0 or two_projections.pop() > 0:
                        #debug(f"{ray1} and {ray2} are irredundant")
                        remove = False
                        break

            #not sure about this constraint (ray2 not in to_remove)
            if remove and ray2 not in to_remove:
                to_remove.add(ray1)

    #debug(f"removing rows: {to_remove}")
    irredundant = list(range(tableau.shape[0]))
    #debug(f" irredundant start: {irredundant}")
    for redundant in to_remove:
        irredundant.remove(redundant)
    #debug(f" irredundant: {irredundant}")
    new_tableau = tableau[np.ix_(irredundant,range(tableau.shape[1]))]
    return new_tableau



def homogenize_system(A : np.ndarray, b : np.ndarray):
    assert A.shape[0] == b.shape[0]
    if A.shape[0] == 0:
        return np.array([[]])
    A_mat = np.atleast_2d(A)

    #interpret as column, not row vector
    if A_mat.shape[1]  == 1:
        A_mat = A_mat.T

    return np.hstack((A_mat, np.atleast_2d(-b).T))

def dehomogenize_ray(y : np.ndarray) -> Generator:
    n = len(y)
    vec = y[0:(n-1)]
    last = y[-1]
    #debug(f"dehomogenizing {vec} ({last})")
    if np.isclose(last,0):
        return Generator(vec)
    else:
        return Generator(1/last * vec,vertex = True)

def to_bidirectional_coordinates(rays : List[Generator]) -> List[Generator]:
    result = []
    for ray in rays:
        if ray.is_unidirectional():
            r = Ray(np.hstack((0,ray.as_ndarray())))
            result.append(r)
        else:
            r = Ray(np.hstack((1,ray.as_ndarray())))
            result.append(r)
    return result

### Unneccessary?


# Hyperplanes represent equations:
# H = {x | ax = 0}
class Hyperplane():
    pass



def Qequals():
    pass

def Qgreater():
    pass

def Qless():
    pass

