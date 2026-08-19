def derivative(f, x0, h=1e-5):
    """Return the derivative of f at x0 using central difference approximation."""
    return (f(x0 + h) - f(x0)) / h


def second_derivative(f, x0, h=1e-5):
    """Return the second derivative of f at x0 using central difference approximation."""
    def prime(x):
        return derivative(f, x, h)

    return derivative(prime, x0, h)


# optimize function using Newton's method
def optimize(f, x0, tol=1e-7, max_iter=100):
    """Optimize the function f using Newton's method starting from x0."""
    x = x0
    for i in range(max_iter):
        df = derivative(f, x)
        df2 = second_derivative(f, x)
        x_new = x - df / df2
        if abs(x - x_new) < tol:
            return x_new
        x = x_new

    return x
