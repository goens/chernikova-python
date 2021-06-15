#This implements Chernikova's algorithm as described in:
#Felipe Fernandez, Patrice Quinton. Extension of Chernikova’s algorithm for solving general mixed linear programming problems. [Research Report] RR-0943, 1988

#with the extension described in:
#Hervé Le Verge. A Note on Chernikova’s algorithm. [Research Report] RR-1662, INRIA. 1992. inria-00074895

import numpy as np
from typing import List

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

    def __str__(self):
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
class Ray():

    def is_extremal(self):
        pass #needed? (p.5)

    #Ray
    def is_unidirectional(self) -> bool:
        return self.unidirectional

    #Line
    def is_bidirectional(self) -> bool:
        return not self.unidirectional

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
    def generators() -> List[Ray]:
        pass


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
        elif np.isclose(row2[0],1):
            #same as above, but reversed
            row1 = -np.sign(p1*p2) * row1
            row1[0] = 1
        #both are bidirectional
        else:
            row1 = -np.sign(p1*p2)*row1
            #row2 = row2

    res = row1 * -p2 + row2 * p1
    if res.dtype == np.int64:
        gcd = np.gcd.reduce(res)
        res = res / gcd

    return res


#This computes the new tableau without removing irredundant rays
def chernikova_iteration(tableau : np.ndarray, column : int, n : int) -> np.ndarray:
    constraint = tableau[:,column]
    ineq = np.isclose(constraint[0],1)
    projections = np.matmul(np.atleast_2d(constraint) , tableau[:,0:n+2])
    assert projections.shape == (1,n+2)

    #conserved rays
    conserved_rays = []
    if ineq:
        for i in range(n+2):
            #H^1 : projections are non-negative
            if projections[0,i] >= 0:
                conserved_rays.append(i)
    else:
        for i in range(n+2):
            #H^0: projections are null
            if np.isclose(projections[0,i],0):
                conserved_rays.append(i)

    new_tableau_rows = []
    for idx in conserved_rays:
        row = tableau[idx,:]

        #if bidirectional, change
        if np.isclose(row[0],0):
            if projections[0,idx] > 0:
                #make unidirectional
                row[0] = 1
            elif projections[0,idx] < 0:
                #oposite unidirectional
                row[0] = 1
                row = -row
            #projection == 0 is kept bidirectional
        new_tableau_rows.append(row)

    #new rays
    for i in range(n+2):
        #don't iterate twice over pairs
        for j in range(i+1,n+2):
            #each ray belongs to a different side of the hyperplane

            p1 = projections[0,i]
            p2 = projections[0,j]
            row1 = tableau[i,:]
            row2 = tableau[j,:]
            if p1*p2 < 0 or np.isclose(row[0],0) or np.isclose(row[0],0):
                new_row = row_lin_combination_satisfying_constraint(row1,p1,row2,p2)
                new_tableau_rows.append(new_row)
    return np.array(new_tableau_rows)

def chernikova_reduction(tableau : np.ndarray) -> np.ndarray:
    pass

def chernikova_reduction_leverge(tableau : np.ndarray) -> np.ndarray:
    pass

def homogenize_system(A : np.ndarray, b : np.ndarray):
    assert A.shape[0] == b.shape[0]
    if A.shape[0] == 0:
        return np.array([[]])
    A_mat = np.atleast_2d(A)

    #interpret as column, not row vector
    if A_mat.shape[1]  == 1:
        A_mat = A_mat.T

    return np.hstack((A_mat, np.atleast_2d(-b).T))

def dehomogenize_ray(y : np.ndarray):
    n = y.shape[1]
    if np.isclose(y[0,n],0):
        return Ray(y[0,0:(n-1)])
    else:
        return Vertex(y)

def to_bidirectional_coordinates(rays : List[Ray]) -> List[Ray]:
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

#this can probably be vectorized (using something like scipy.linalg.null_space?)
def S(y : np.ndarray, A : Cone) -> List[Constraint]:
    result = []
    for constraint in A.constraints:
        if constraint.is_saturated_by_y(y):
            result.append(y)
    return result
