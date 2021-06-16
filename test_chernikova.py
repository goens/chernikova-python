#Copyright 2021 Andrés Goens <andres.goens@barkhauseninstitut.org>
#import pytest
from chernikova import *

#Example from Section 8 in the paper
def example_8():
    equation = Constraint(np.array([0,0,1]),1,equality = True)
    ineq1 = Constraint(np.array([1,0,0]))
    ineq2 = Constraint(np.array([-1,0,0]),-1)
    ineq3 = Constraint(np.array([0,1,0]),-1)
    return [equation,ineq1,ineq2,ineq3]

#Example from Section 9 in the paper
def example_9():
    ineq1 = Constraint(np.array([1,-1]))
    ineq2 = Constraint(np.array([1,0]))
    return [ineq1,ineq2]

def test_constraints():
    constraints = example_8()
    for c in constraints:
        print(c)

def test_cone():
    cone = Cone(example_8())
    assert np.allclose(cone.B,
                       np.array([[0,0,1]]))
    assert np.allclose(cone.B_b,
                       np.array([1]))

    assert np.allclose(cone.D,
                       np.array([[1,0,0],[-1,0,0],[0,1,0]]))
    assert np.allclose(cone.D_b,
                       np.array([0,-1,-1]))

    cone = Cone(example_9())
    assert np.allclose(cone.B,
                       np.array([[]]))
    assert np.allclose(cone.B_b,
                       np.array([]))

    assert np.allclose(cone.D,
                       np.array([[1,-1],[1,0]]))
    assert np.allclose(cone.D_b,
                       np.array([0,0]))


def test_uniform():
    cone = Cone(example_8())
    uniform = cone.to_uniform_system()
    assert np.allclose(uniform,
                       np.array([[ 0,  1,  1,  1,  1],
                                 [ 0,  1, -1,  0,  0],
                                 [ 0,  0,  0,  1,  0],
                                 [ 1,  0,  0,  0,  0],
                                 [-1,  0,  1,  1,  1]]
                       ))

    cone = Cone(example_9())
    uniform = cone.to_uniform_system()

def test_chernikova_initial_tableau():
    cone = Cone(example_8())
    chernikova_tableau = cone.to_initial_chernikova_tableau()
    assert np.allclose(chernikova_tableau,
                       np.array([[1, 0, 0, 0, 0, 0,  1,  1,  1,  1],
                                 [0, 1, 0, 0, 0, 0,  1, -1,  0,  0],
                                 [0, 0, 1, 0, 0, 0,  0,  0,  1,  0],
                                 [0, 0, 0, 1, 0, 1,  0,  0,  0,  0],
                                 [0, 0, 0, 0, 1,-1,  0,  1,  1,  1]]
                       ))

    cone = Cone(example_9())
    chernikova_tableau = cone.to_initial_chernikova_tableau()
    assert np.allclose(chernikova_tableau,
                       np.array([[1, 0, 0, 0,  1, 1, 1],
                                 [0, 1, 0, 0,  1, 1, 0],
                                 [0, 0, 1, 0, -1, 0, 0],
                                 [0, 0, 0, 1,  0, 0, 1]]
                       ))


def test_trivial_system():
    cone = Cone([])
    print(cone.to_initial_chernikova_tableau())

def test_chernikova_iteration():
    cone = Cone(example_9())
    tableau = cone.to_initial_chernikova_tableau()
    step_example_9 = chernikova_iteration(tableau,4,cone.n)
    #print(step_example_9)
    assert np.allclose(step_example_9,
                       np.array(
                           [[ 1,  0,  0,  0,  1,  1,  1],
                            [ 1,  1,  0,  0,  1,  1,  0],
                            [ 1,  0, -1,  0,  1,  0,  0],
                            [ 0,  0,  0,  1,  0,  0,  1],
                            [ 0,  1,  1,  0,  0,  1,  0]]
                           ))
    #print(chernikova_reduction(step_example_9,cone.n))

def test_chernikova_full():
    cone = Cone(example_9())
    result = cone.generators()
    #TODO: write a test that ensures a) generators indeed generate polytope, and b) they are irredundant
    #print(result)


if __name__ == "__main__":
    #test_constraints()
    #test_trivial_system() #not working
    test_cone()
    test_uniform()
    test_chernikova_initial_tableau()
    test_chernikova_iteration()
    test_chernikova_full()
