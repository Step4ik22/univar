from operator import mul as _opmul
from functools import partial, reduce

def summul_gen():
    _compose_star = lambda f, g: lambda *args: f(g(*args))
    return reduce(_compose_star, (sum, partial(map, _opmul)))  # "Vectorized" dot-product

def dotcols(A, i):
    return [summul(A[i], A[j]) for j in range(len(A))]       # Dot prod all cols per row

summul = summul_gen()