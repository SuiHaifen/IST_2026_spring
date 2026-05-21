import numpy as np
from numpy.linalg import LinAlgError
import scipy
try:
    from scipy.optimize import scalar_search_wolfe2
except ImportError:
    try:
        from scipy.optimize.linesearch import scalar_search_wolfe2
    except ImportError:
        def scalar_search_wolfe2(phi, derphi, alpha0=1.0, c1=1e-4, c2=0.9, maxiter=100):
            return None, None, None, None
from datetime import datetime
from collections import defaultdict
import time


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo' -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
        
        elif self._method == 'Armijo':
            # Starting alpha
            if previous_alpha is not None:
                alpha = previous_alpha * 2.0
            else:
                alpha = self.alpha_0
            
            # Get initial values
            phi_0 = oracle.func_directional(x_k, d_k, 0.0)
            phi_prime_0 = oracle.grad_directional(x_k, d_k, 0.0)
            
            # Backtracking
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                alpha = alpha / 2.0
                if alpha < 1e-16:
                    return None
        
        elif self._method == 'Wolfe':
            # Try Wolfe conditions using scipy
            phi = lambda a: oracle.func_directional(x_k, d_k, a)
            derphi = lambda a: oracle.grad_directional(x_k, d_k, a)
            
            # Use previous_alpha as starting point if provided
            alpha0 = previous_alpha if previous_alpha is not None else self.alpha_0
            
            alpha, phi_alpha, phi_prime_alpha, task = scalar_search_wolfe2(
                phi, derphi, alpha0, c1=self.c1, c2=self.c2
            )
            
            if alpha is not None:
                return alpha
            
            # If Wolfe fails, fall back to Armijo backtracking
            phi_0 = phi(0.0)
            phi_prime_0 = derphi(0.0)
            alpha = self.alpha_0
            
            while True:
                phi_alpha = phi(alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                alpha = alpha / 2.0
                if alpha < 1e-16:
                    return None
        
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = time.time()
    previous_alpha = None
    
    # Initial gradient norm for stopping criterion
    grad_k = oracle.grad(x_k)
    grad_norm0_sq = np.dot(grad_k, grad_k)
    
    for iteration in range(max_iter):
        # Check stopping criterion
        grad_norm_sq = np.dot(grad_k, grad_k)
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break
        
        # Compute descent direction (negative gradient)
        d_k = -grad_k
        
        # Line search to find step size
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        # Check for computational error
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            message = 'computational_error'
            break
        
        # Update x_k
        x_new = x_k + alpha * d_k
        
        # Check for NaN or Inf
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            message = 'computational_error'
            break
        
        # Update history if trace is True
        if trace:
            elapsed_time = time.time() - start_time
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Update for next iteration
        x_k = x_new
        grad_k = oracle.grad(x_k)
        previous_alpha = alpha
        
        # Display info if requested
        if display and iteration % 100 == 0:
            print(f"Iter {iteration}: f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_sq):.6e}, alpha={alpha:.6e}")
    else:
        # Loop completed without break - max iterations exceeded
        message = 'iterations_exceeded'
    
    return x_k, message, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = time.time()
    previous_alpha = None
    
    # Initial gradient norm for stopping criterion
    grad_k = oracle.grad(x_k)
    grad_norm0_sq = np.dot(grad_k, grad_k)
    
    for iteration in range(max_iter):
        # Check stopping criterion
        grad_norm_sq = np.dot(grad_k, grad_k)
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break
        
        # Compute Hessian
        hess_k = oracle.hess(x_k)
        
        # Solve linear system H * d = -grad using Cholesky decomposition
        try:
            # Try Cholesky decomposition for positive definite matrix
            cho_factor = scipy.linalg.cho_factor(hess_k)
            d_k = scipy.linalg.cho_solve(cho_factor, -grad_k)
        except (LinAlgError, ValueError):
            # If Cholesky fails, try general solver
            try:
                d_k = scipy.linalg.solve(hess_k, -grad_k)
            except LinAlgError:
                message = 'newton_direction_error'
                break
        
        # Check for NaN or Inf
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            message = 'newton_direction_error'
            break
        
        # For Newton's method, always try alpha=1 first
        # Force the line search to start with alpha=1
        # Use a separate line search tool with alpha_0=1
        if hasattr(line_search_tool, '_method'):
            if line_search_tool._method == 'Armijo':
                # Override alpha_0 to 1 for this iteration
                original_alpha_0 = line_search_tool.alpha_0
                line_search_tool.alpha_0 = 1.0
                alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
                line_search_tool.alpha_0 = original_alpha_0
            elif line_search_tool._method == 'Wolfe':
                # For Wolfe, use alpha=1 as starting point
                alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
            else:
                # Constant step size
                alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        else:
            alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        # Check for computational error
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            message = 'computational_error'
            break
        
        # Update x_k
        x_new = x_k + alpha * d_k
        
        # Check for NaN or Inf
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            message = 'computational_error'
            break
        
        # Update history if trace is True
        if trace:
            elapsed_time = time.time() - start_time
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Update for next iteration
        x_k = x_new
        grad_k = oracle.grad(x_k)
        previous_alpha = alpha
        
        # Display info if requested
        if display and iteration % 10 == 0:
            print(f"Iter {iteration}: f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_sq):.6e}, alpha={alpha:.6e}")
    else:
        # Loop completed without break - max iterations exceeded
        message = 'iterations_exceeded'
    
    return x_k, message, history