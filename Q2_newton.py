import numpy as np #type:ignore

# Ideal solar configuration vector c
c = np.array([
    25,   # tilt angle
    180,  # azimuth angle
    2,    # panel height
    5,    # row spacing
    0.9,  # module efficiency tuning
    0.8,  # inverter operating point
    1.5,  # cable thickness factor
    12,   # cleaning frequency coefficient
    0.95, # thermal derating factor
    0.5   # shading-loss compensation
], dtype=float)

q = np.array([0.1, 0.2, 0.15, 0.1, 0.05, 0.3, 0.1, 0.2, 0.25, 0.1], dtype=float)


def f(x):
    return np.sum((x - c)**4 + (x - c)**2 + q * x**2)


def grad(x):
    return 4*(x - c)**3 + 2*(x - c) + 2*q*x


def hess(x):
    return np.diag(12*(x - c)**2 + 2 + 2*q)


def newton(x0, tol, max_iter):
    x = x0.astype(float)

    print("Iter |      f(x)         |   ||grad||")
    print("------------------------------------------")

    for k in range(max_iter):
        g = grad(x)
        H = hess(x)
        grad_norm = np.linalg.norm(g)
        print(f"{k:4d} | {f(x):14.4f} | {grad_norm:10.4f}")
        if grad_norm < tol:
            break
        p = -np.linalg.solve(H, g)
        x = x + p
    return x


x0 = np.zeros(10)
xmin = newton(x0, pow(10,-13),1000)
print("Initial f is:", f(x0))
print("\nOptimal design vector x* =\n", xmin)
print("\nMinimum objective f(x*) =", f(xmin))