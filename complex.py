import numpy as np
import cmath
import sys

""" to_polarとto_rectで処理方法を合わせたほうが良いような気がする。 """
""" 極形式にするのはnumpyでもできるんだけどね。cmathで合わせるほうが安心じゃん？ """
def to_polar(comp):
    if len(comp.shape) > 2:
        sys.exit("The input array shape is too large. Please put it in two dimensions.")

    # z = np.sqrt(pow(comp.real, 2) + pow(comp.imag, 2)).astype(np.float32)
    # z = np.abs(comp)
    # angle = np.angle(comp).astype(np.float32)

    z = []
    angle = []
    for c in comp:
        for _c in c:
            _z, _a = cmath.polar(_c)
            z.append(_z)
            angle.append(_a)

    z = np.array(z, dtype=np.float32).reshape(comp.shape)
    angle = np.array(angle, dtype=np.float32).reshape(comp.shape)

    return z, angle


def to_rect(z, angle, target_shape):
    comp = []
    for (Z, Angle) in zip(z, angle):
        for (_z, _angle) in zip(Z, Angle):
            comp.append(cmath.rect(_z, _angle))

    comp = np.array(comp, dtype=np.complex64).reshape(target_shape)

    return comp
