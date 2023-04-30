import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    # чьто?     
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
#     print("analitic grads", analytic_grad)
#     print("x", x)
#     print("iterator", it)
    
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0
        # print("ix", ix)
        # print("analytic_grad_at_ix", analytic_grad_at_ix)
        
        
        # TODO compute value of numeric gradient of f to idx
  
        x[ix] += delta
        fx_plus = f(x)
        x[ix] -= 2*delta
        fx_minus = f(x)
        numeric_grad_at_ix = np.divide((fx_plus[0] - fx_minus[0]), 2*delta)

        # print("fx_plus_d", fx_plus_d)
        # print("fx_minus_d", fx_minus_d)
        # print("subs", fx_plus_d - fx_minus_d)
        # print("2*delta", 2*delta)
        # print("numeric_grad_at_ix", numeric_grad_at_ix)
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
