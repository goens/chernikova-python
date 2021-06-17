# Copyright 2021 Andrés Goens <andres.goens@barkhauseninstitut.org>

import islpy
import numpy as np
from chernikova import *
from typing import List

#<class 'islpy._isl.Set'>
#Ein Beispiel einer Dimension von GEMM (simpel)
gemm_ex = islpy._isl.BasicSet("{ [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12] : i3 > 0 and i8 >= i0 - i2 + i5 and i8 >= i5 and i9 >= i1 - i4 + i6 and i9 >= i6 and i10 >= i7 and i12 > i5 + i6 + i7 - i8 - i9 - i10 + i11 }")
#Von Nussinov, anfängliche Dimension (mittel)
nussinov_ex_mid = islpy._isl.BasicSet("{ [i0, i1, i2, i3, i4, 0, i2, i7, i2, i9, i2, i11, i2, i13, i2, i15, 0, i17, i18, i19, i20, i21, i22, i19, i20, i19, i20, i19, i20, i19, i20, i19, i20, i33, i34, i35, i36, i37, i38, i39, i40] : i4 >= 0 and i7 >= i3 and i7 <= i11 <= i9 and 0 <= i15 <= i3 and i36 >= i34 and i37 >= i36 and i7 - i11 + i36 <= i38 <= 2 * i9 - i11 + i37 and i37 <= i39 <= i9 + i37 and i39 <= i2 + i37 and i40 > -i15 + i39 and -2 * i2 - 2 * i15 + i39 < i40 <= 2 * i3 - i15 + i34 }")
#Von Nussinov, spätere Dimension (schwer)
nussinov_ex_hard = islpy._isl.BasicSet("{ [i0, i1, i2, i3, i4, 0, i6, i7, i8, i9, i10, i11, i8, i13, i14, i15, 0, i17, i18, i19, i20, i21, i22, i23, i20, i25, i20, i27, i20, i25, i20, i31, i20, i33, i34, i35, i36, i37, i38, i39, i40] : i4 >= 0 and i15 >= 0 and i15 >= -i14 and i23 >= i2 + i3 - i6 - i7 + i19 and i23 >= i2 - i6 + i19 and i23 >= i19 and i25 >= i6 - i8 + i23 and i25 >= i23 and i27 >= i6 + i7 - i10 - i11 + i23 and i27 >= i6 - i10 + i23 and i23 <= i27 <= i25 and i27 <= i8 - i10 + i25 and i27 <= i8 + i9 - i10 - i11 + i25 and i31 <= i19 and i31 <= i2 - i14 + i19 and i31 <= i2 + i3 - i14 - i15 + i19 and i36 >= i2 - i6 + 2 * i19 - 2 * i23 + i34 and i37 >= i6 - i8 + 2 * i23 - 2 * i25 + i36 and 2 * i6 + i7 - 2 * i10 - i11 + 3 * i23 - 3 * i27 + i36 <= i38 <= 3 * i8 + 2 * i9 - 3 * i10 - i11 + 4 * i25 - 4 * i27 + i37 and i39 >= i6 - i8 + 2 * i23 - 2 * i25 + i36 and i37 <= i39 <= i9 + i37 and i39 <= i8 + i37 and i40 <= 3 * i2 + 2 * i3 - 3 * i14 - i15 + 4 * i19 - 4 * i31 + i34 }")


def isl_Constraint_to_Constraint(isl_constraint : islpy._isl.Constraint,
                                 assume_ints = True) -> Constraint:
    eq = isl_constraint.is_equality()
    variables = isl_constraint.get_var_names(islpy.dim_type.set)
    coeffs_dict = isl_constraint.get_coefficients_by_name()
    coeffs = np.zeros(len(variables))
    for i,var in enumerate(variables):
        if var in coeffs_dict:
            if assume_ints:
                coeffs[i] = np.int64(coeffs_dict[var].to_python())
            else:
                coeffs[i] = np.float64(coeffs_dict[var].to_python())
    if assume_ints:
        const = np.int64(isl_constraint.get_constant_val().to_python())
    return Constraint(coeffs,const,eq)

def isl_ConstraintList_to_Cone(constraint_list : islpy._isl.ConstraintList) -> Cone:
    constraints = []
    for constraint in constraint_list:
        constraints.append(isl_Constraint_to_Constraint(constraint))
    return Cone(constraints)


def isl_BasicSet_to_Generator_list(basicset : islpy._isl.BasicSet) -> List[Generator]:
    cone = isl_ConstraintList_to_Cone(basicset.get_constraints())
    return cone.generators()


print(isl_BasicSet_to_Generator_list(gemm_ex))
