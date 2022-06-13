import numpy as np
from ptorru_matmul import __version__
from ptorru_matmul import ptorru_matmul

def test_version():
    assert __version__ == '0.1.0'


def test_matmul():
    sides = 4
    a = np.arange(sides*sides).reshape(sides,sides)
    b = np.arange(sides*sides).reshape(sides,sides)
    c = ptorru_matmul(a,b)
    assert np.array_equal(c, np.matmul(a,b))
    print(a,b)
    print(c)
