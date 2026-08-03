from pipeline import _constant_fold_readable, _sympy_canon

tests = [
    ('(age + 0.0)', 'identity +'),
    ('(0.0 + age)', 'identity 0+'),
    ('(age - 0.0)', 'identity -'),
    ('(age * 1.0)', 'identity *1'),
    ('(1.0 * age)', 'identity 1*'),
    ('(age / 1.0)', 'identity /1'),
    ('(age * 0.0)', 'annihil *0'),
    ('(0.0 * age)', 'annihil 0*'),
    ('IF(CustodyStatus_Parole, 5.0, 5.0)', 'IF same'),
    ('NOT(0.0)', 'NOT 0'),
    ('NOT(1.0)', 'NOT 1'),
    ('(IF(Agency_Text_Probation, 10.0, 4.0) + 0.0)', 'complex + 0'),
    ('(IF(LegalStatus_Deferred_Sentencing, 1.0, 1.0) + 10.11294126565182)', 'IF same + N'),
    ('max(IF(CustodyStatus_Parole, 5.0, 5.0), 6.0)', 'max(IF same, C)'),
    ('((ScoreText_FTA - 0.0) * 5.0)', 'nested - 0 * N'),
    ('(0.0 * IF(LegalStatus_Probation_Violator, 0.399, 0.0))', '0 * IF'),
    ('(region_South_East_Region / 1.0)', 'var / 1'),
    ('NOT(IF(LegalStatus_Conditional_Release, 0.0, 0.0))', 'NOT(IF same 0)'),
    # From real runs
    ('((|ScoreText_Violence| * 0.4) - 0.0)', '|x|*c - 0'),
    ('(IF(LegalStatus_Pretrial, 1.0, 1.0) + 10.11294126565182)', 'IF same + const'),
    ('(IF(LegalStatus_Pretrial, AssessmentReason, AssessmentType) / (((10.0 + ScoreText_Violence) + (4.0 * ScoreText_FTA)) - (DecileScore_Violence / IF(MaritalStatus_Divorced, Sex_Code_Text, ScoreText_FTA))))', 'complex no change'),
    # The original problematic expression
    ('((((region_London_Region + 2.0) + ((2.0 * -1.317) * 1.0)) / (((0.0 + 0.0) * (0.0 * 1.0)) - (2.0 * (0.6499 / 0.2955)))) * (0.0 - (((0.0 / 0.0) * 2.0) + ((code_module_BBB * 2.0) * 1.0)))) < 0', '0/0 expression'),
]

sympy_tests = [
    # Stage 2: additive constant on LHS should move to RHS
    ('2.0*Education_Num - 15.42*Hours_per_week + 0.09196 < -16.1', 'sympy lhs const'),
    # Pure numeric: should collapse to 0.0 or 1.0
    ('3.0 + 2.0 < 4.0', 'sympy const false'),
    ('1.0 + 1.0 < 4.0', 'sympy const true'),
]

alg_inv_tests = [
    # Algebraic inversion: (C1 - X) op C2 → (X flip_op C1-C2)
    ('(6.0 - age) ≤ 5.0', 'alg-inv case1 le'),
    ('(10.0 - score) < 3.0', 'alg-inv case1 lt'),
    # Algebraic inversion: C2 op (C1 - X) → (X op C1-C2)
    ('5.0 ≤ (16.67 - score)', 'alg-inv case2 le'),
    ('3.0 < (10.0 - age)', 'alg-inv case2 lt'),
    # Mid-constant chained ≤ split + inversion
    ('(age ≤ 5.0 ≤ score)', 'chained mid-const split'),
]

print("=== Constant fold tests ===")
for t, name in tests:
    r = _constant_fold_readable(t)
    r2 = _constant_fold_readable(_sympy_canon(r))
    status = 'OK' if r2 != t else 'SAME'
    print(f"{status} [{name}]")
    if r2 != t:
        print(f"  IN:  {t}")
        print(f"  OUT: {r2}")
    print()

print("=== SymPy canon tests ===")
for t, name in sympy_tests:
    r = _sympy_canon(t)
    status = 'OK' if r != t else 'SAME'
    print(f"{status} [{name}]")
    print(f"  IN:  {t}")
    print(f"  OUT: {r}")
    print()

chained_general_tests = [
    # Variable in middle
    ('260.0 ≤ index6040 ≤ (96.0 + (max(15.0, fulltime) * 25.0))', 'var in middle le-le'),
    # Expression in middle
    ('1.0 ≤ (age + score) ≤ 10.0', 'expr in middle le-le'),
    # Mixed operators
    ('0.0 < age ≤ 65.0', 'lt-le mixed'),
    # Constant in middle (not covered by mid-const rule which only does ≤ C ≤)
    ('age < 30.0 < score', 'const in middle lt-lt'),
]

print("=== General chained inequality tests ===")
for t, name in chained_general_tests:
    r = _constant_fold_readable(t)
    status = 'OK' if r != t else 'SAME'
    print(f"{status} [{name}]")
    print(f"  IN:  {t}")
    print(f"  OUT: {r}")
    print()

print("=== Algebraic inversion tests ===")
for t, name in alg_inv_tests:
    r = _constant_fold_readable(t)
    status = 'OK' if r != t else 'SAME'
    print(f"{status} [{name}]")
    print(f"  IN:  {t}")
    print(f"  OUT: {r}")
    print()
