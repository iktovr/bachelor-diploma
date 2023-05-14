from numpy import sin, cos, pi, abs


# 2D

def linear(t, u, a=-0.1, b=2, c=-2, d=-0.1):
    x1, x2 = u
    return [
        a * x1 + b * x2,
        c * x1 + d * x2
    ]
linear.u0 = (0.1, 0.1)


def hopf_normal_form(t, u, mu, w, A):
    x1, x2 = u
    return [
        mu * x1 + w * x2 + A * (x1 ** 2 + x2 ** 2),
        -w * x1 + mu * x2 + A * (x1 ** 2 + x2 ** 2)
    ]


# не автономная
def van_der_pol(t, u, a1=3, a2=2.5, a3=2.7):
    x1, x2 = u
    return [
        x2,
        -x1 + a1 * x2 * (1 - x1 * x1) + a2 * cos(a3 * t)
    ]
van_der_pol.u0 = (0.1, 0.1)


# 3D

def lorenz_attractor(t, u, s=10, r=28, b=8/3):
    x1, x2, x3 = u
    return [
        s * (x2 - x1),
        x1 * (r - x3) - x2,
        x1 * x2 - b * x3
    ]
lorenz_attractor.u0 = (-8, 7, 27)
lorenz_attractor.terms = (('x1', 'x2'), ('x1', 'x2', 'x1 x3'), ('x3', 'x1 x2'))


def mean_field_model(t, u, mu, w, A, l):
    x1, x2, x3 = u
    return [
        mu * x1 + w * x2 + A * x1 * x3,
        w * x1 + mu * x2 + A * x2 * x3,
        -l * (x3 - x1 ** 2 - x2 ** 2)
    ]


def moore_spiegel(t, u, a1=0.8, a2=50):
    x1, x2, x3 = u
    return [
        x2,
        x3 - x1 * (1 - a1),
        -x3 / a2 + x2 * (1 - a1 * x1 * x1)
    ]
moore_spiegel.u0 = (0.1, 0.2, 0.3)
moore_spiegel.terms = (('x2',), ('x1', 'x3'), ('x2', 'x3', 'x1 x1 x2'))


def fabrab(t, u, a1=0.05, a2=0.1):
    x1, x2, x3 = u
    return [
        x2 * (-1 + x1 * x1 + x3) + x1 * a1,
        x1 * (1 - x1 * x1 + 3 * x3) + x2 * a1,
        -2 * x3 * (a2 + x1 * x2)
    ]
fabrab.u0 = (0.1, -0.1, 0.1)
fabrab.terms = (('x1', 'x2', 'x1 x1 x2', 'x2 x3'), ('x1', 'x2', 'x1 x1 x1', 'x1 x3'), ('x3', 'x1 x2 x3'))


def vallis(t, u, a1=60.175, a2=3):
    x1, x2, x3 = u
    return [
        a1 * x2 - a2 * x1,
        x1 * x3 - x2,
        -x1 * x2 - x3 + 1
    ]
vallis.u0 = (1, 0, 1)
vallis.terms = (('x1', 'x2'), ('x2', 'x1 x3'), ('x3', 'x1 x2'))


def simple_3d_model(t, u, a=7):
    x1, x2, x3 = u
    return [
        1 + a * x2 * x3,
        x1 - x2,
        1 - x1 * x2
    ]
simple_3d_model.u0 = (1, 1, 0)
simple_3d_model.terms = (('1', 'x2 x3'), ('x1', 'x2'), ('1', 'x1 x2'))


def torus(t, u, a1=1, a2=2.5, a3=0.5):
    def dd(x):
        return -a3 * x + 0.5 * (a3 + 1) * (abs(x + 1) - abs(x - 1))
    x1, x2, x3 = u
    return [
        (a1 - 1) * dd(x1) / a2 - x3 / a2,
        -a1 * dd(x1) / a2,
        a2 * (x1 + x2)
    ]
torus.u0 = (0.3499, 0.7683, -4.2814)


def double_scroll(t, u, a1=5.132, a2=7, a3=0.14, a4=-0.28):
    def dd(x):
        return a3 * x + 0.5 * (a4 - a3) * (abs(x + 1) - abs(x - 1))
    x1, x2, x3 = u
    return [
        a1 * (x2 - dd(x1)),
        x1 - x2 + x3,
        -a2 * x2
    ]
double_scroll.u0 = (1, 1, 1)


def chua(t, u, a1=0.7, a2=7, a3=-0.5, a4=-0.8, a5=0.111):
    def dd(x):
        return a3 * x + 0.5 * (a4 - a3) * (abs(x + 1) - abs(x - 1))
    x1, x2, x3 = u
    return [
        (a1 * (x2 - x1) - dd(x1)) / a5,
        a1 * (x1 - x2) + x3,
        -a2 * x2
    ]
chua.u0 = (0.1, 0.2, 0.3)


# не автономная
def aritmic(t, u, a1=0.5, a2=0.5, a3=0.2, a4=0.03):
    x1, x2, x3 = u
    return [
        x3,
        -a1 * x2 * (3 * x1 * x1 + x2 * x2) / 8 + a4,
        a3 * cos(t) - a2 * x1 - x1 * (3 * x2 * x2 + x1 * x1) / 8
    ]
aritmic.u0 = (0.1, 0.1, 0.1)


# не автономная
def generator1(t, u, a1=1, a2=3, a3=2*pi, a4=0.3):
    x1, x2, x3 = u
    return [
        a1 * x1 + x2 - x1 * x3 + a2 * sin(a3 * t),
        -x1,
        a4 * ((x1 > 0) * x1 * x1 - x3)
    ]
generator1.u0 = (0.1, 0.1, 0.1)


# не автономная
def generator2(t, u, a1=15, a2=1, a3=0.15, a4=0.5):
    x1, x2, x3 = u
    return [
        a1 * x1 + x2 - x1 * x3 + a2 * sin(a3 * t),
        -x1,
        a4 * (x1 * x1 - x3)
    ]
generator2.u0 = (1, 2, 3)


# 4D

def rikitaki(t, u, a1=0.5, a2=0.004, a3=0.002):
    x1, x2, x3, x4 = u
    return [
        -a1 * x1 + x2 * x3,
        -a1 * x2 + x1 * x4,
        1 - x1 * x2 - a2 * x3,
        1 - x1 * x2 - a3 * x4
    ]
rikitaki.u0 = (0.1, 0.3, 0.2, 0.4)


def henon_heiles(t, u):
    x1, x2, x3, x4 = u
    return [
        x2,
        -x1 * (1 + 2 * x3),
        x4,
        -(x1 * x1 + x3 - x3 * x3)
    ]
henon_heiles.u0 = (0.55, 0, 0, 0)


def yang_mills(t, u):
    x1, x2, x3, x4 = u
    return [
        x3,
        x4,
        -x1 * x2 * x2,
        -x2 * x1 * x1
    ]
yang_mills.u0 = (2, 1, 1, 1)


# не автономная
def child_infection(t, u, a1=0.02, a2=1800, a3=100, a4=0.28, a5=6.283, a6=35.84):
    x1, x2, x3, x4 = u
    return [
        a1 * (1 - x1) - x1 * x3 * x4,
        -(a1 + a6) * x2 + x1 * x3 * x4,
        a6 * x2 - (a1 + a3) * x3,
        -a2 * a4 * a5 * sin(a5 * t)
    ]
child_infection.u0 = (0.1, 0.1, 0.1, 2304)
