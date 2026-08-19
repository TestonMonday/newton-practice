def derivative(f, x0, h=1e-5):
    return (f(x0+h) - f(x0)) / h

def second_derivative(f, x0, h=1e-5):
    def prime(x):
        return derivative(f, x, h)
    return derivative(prime, x0, h)


#optimize function using Newton's method
def optimize(f, x0, tol=1e-7, max_iter=100):
    x=x0
    for i in range(max_iter):
        df=derivative(f,x)
        df2=second_derivative(f,x)
        x_new=x-df/df2
        if abs(x-x_new)< tol:
            return x_new
        x=x_new

    return x
