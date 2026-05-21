import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute b_i * (a_i^T x)
        # We need to compute log(1 + exp(-b_i * a_i^T x))
        # Using np.logaddexp for numerical stability: log(1 + exp(t)) = logaddexp(0, t)
        # Here t = -b_i * a_i^T x
        t = -self.b * Ax
        
        # Use logaddexp(0, t) to compute log(1 + exp(t)) numerically stable
        # Note: logaddexp(0, t) = log(exp(0) + exp(t)) = log(1 + exp(t))
        logistic_loss = np.sum(np.logaddexp(0, t)) / len(self.b)
        
        # Add L2 regularization
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return logistic_loss + reg_term

    def grad(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute sigmoid: 1 / (1 + exp(b_i * a_i^T x))
        # Actually we need: 1 / (1 + exp(-b_i * a_i^T x)) = expit(b_i * a_i^T x)
        # But let's compute carefully:
        # derivative of log(1 + exp(-b_i * a_i^T x)) w.r.t x is:
        # = -b_i * a_i * (1 / (1 + exp(b_i * a_i^T x)))
        # = -b_i * a_i * expit(-b_i * a_i^T x)
        # Let's use the formula: gradient = -1/m * A^T * (b * sigmoid(-b * Ax))
        t = -self.b * Ax
        # expit(t) = 1 / (1 + exp(-t))
        sigmoid_t = expit(t)
        
        # Compute A^T * (b * sigmoid_t)
        b_sigmoid = self.b * sigmoid_t
        grad_logistic = -self.matvec_ATx(b_sigmoid) / len(self.b)
        
        # Add regularization gradient
        grad_reg = self.regcoef * x
        
        return grad_logistic + grad_reg

    def hess(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute s_i = sigmoid(b_i * a_i^T x) * sigmoid(-b_i * a_i^T x)
        # But we need: p_i = 1 / (1 + exp(-b_i * a_i^T x)) = expit(b_i * a_i^T x)
        # The derivative: sigma_i = p_i * (1 - p_i)
        t = -self.b * Ax
        sigmoid_t = expit(t)
        # sigma_i = sigmoid(t) * (1 - sigmoid(t))
        sigma = sigmoid_t * (1 - sigmoid_t)
        
        # Hessian = 1/m * A^T * diag(sigma) * A + regcoef * I
        hess_logistic = self.matmat_ATsA(sigma) / len(self.b)
        hess_reg = self.regcoef * np.eye(len(x))
        
        return hess_logistic + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        # Cache for precomputed values
        self._cached_Ax = None
        self._cached_x = None
        self._cached_Ad = None
        self._cached_d = None
        self._cached_Ax_alpha = None
        self._cached_alpha_xd = None

    def _get_Ax(self, x):
        """Get Ax with caching"""
        if self._cached_x is not None and np.array_equal(self._cached_x, x):
            return self._cached_Ax
        self._cached_x = x.copy()
        self._cached_Ax = self.matvec_Ax(x)
        return self._cached_Ax

    def func(self, x):
        # Clear cached values for directional methods when x changes
        if self._cached_x is not None and not np.array_equal(self._cached_x, x):
            self._cached_Ax_alpha = None
            self._cached_alpha_xd = None
        Ax = self._get_Ax(x)
        t = -self.b * Ax
        logistic_loss = np.sum(np.logaddexp(0, t)) / len(self.b)
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        return logistic_loss + reg_term

    def grad(self, x):
        Ax = self._get_Ax(x)
        t = -self.b * Ax
        sigmoid_t = expit(t)
        b_sigmoid = self.b * sigmoid_t
        grad_logistic = -self.matvec_ATx(b_sigmoid) / len(self.b)
        grad_reg = self.regcoef * x
        return grad_logistic + grad_reg

    def hess(self, x):
        Ax = self._get_Ax(x)
        t = -self.b * Ax
        sigmoid_t = expit(t)
        sigma = sigmoid_t * (1 - sigmoid_t)
        hess_logistic = self.matmat_ATsA(sigma) / len(self.b)
        hess_reg = self.regcoef * np.eye(len(x))
        return hess_logistic + hess_reg

    def func_directional(self, x, d, alpha):
        # Compute x + alpha*d
        x_alpha = x + alpha * d
        
        # Check if we can use cached value
        if (self._cached_alpha_xd is not None and 
            np.array_equal(self._cached_alpha_xd, x_alpha)):
            Ax_alpha = self._cached_Ax_alpha
        else:
            # Compute Ax for the new point
            if self._cached_x is not None and np.array_equal(self._cached_x, x):
                # We have Ax cached, compute Ad
                if self._cached_d is not None and np.array_equal(self._cached_d, d):
                    Ad = self._cached_Ad
                else:
                    Ad = self.matvec_Ax(d)
                    self._cached_d = d.copy()
                    self._cached_Ad = Ad
                Ax_alpha = self._cached_Ax + alpha * Ad
            else:
                # Fallback to direct computation
                Ax_alpha = self.matvec_Ax(x_alpha)
            
            # Cache for future use
            self._cached_alpha_xd = x_alpha.copy()
            self._cached_Ax_alpha = Ax_alpha
        
        t = -self.b * Ax_alpha
        logistic_loss = np.sum(np.logaddexp(0, t)) / len(self.b)
        reg_term = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return logistic_loss + reg_term

    def grad_directional(self, x, d, alpha):
        x_alpha = x + alpha * d
        
        # Check if we can use cached value
        if (self._cached_alpha_xd is not None and 
            np.array_equal(self._cached_alpha_xd, x_alpha)):
            Ax_alpha = self._cached_Ax_alpha
        else:
            if self._cached_x is not None and np.array_equal(self._cached_x, x):
                if self._cached_d is not None and np.array_equal(self._cached_d, d):
                    Ad = self._cached_Ad
                else:
                    Ad = self.matvec_Ax(d)
                    self._cached_d = d.copy()
                    self._cached_Ad = Ad
                Ax_alpha = self._cached_Ax + alpha * Ad
            else:
                Ax_alpha = self.matvec_Ax(x_alpha)
            self._cached_alpha_xd = x_alpha.copy()
            self._cached_Ax_alpha = Ax_alpha
        
        t = -self.b * Ax_alpha
        sigmoid_t = expit(t)
        # grad_directional = (grad_f(x_alpha))^T * d
        # grad_f(x_alpha) = -1/m * A^T * (b * sigmoid(-b*Ax_alpha)) + regcoef * x_alpha
        b_sigmoid = self.b * sigmoid_t
        grad_logistic = -self.matvec_ATx(b_sigmoid) / len(self.b)
        grad_total = grad_logistic + self.regcoef * x_alpha
        return np.dot(grad_total, d)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        # s is a 1D array of size m
        # Returns A.T * diag(s) * A
        # For sparse matrices, we need to handle efficiently
        if scipy.sparse.issparse(A):
            # For sparse matrices, use dot product
            return A.T.dot(A * s.reshape(-1, 1))
        else:
            # For dense matrices, use einsum or dot
            return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    grad = np.zeros(n)
    f0 = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_plus = func(x + e_i)
        grad[i] = (f_plus - f0) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    H = np.zeros((n, n))
    f0 = func(x)
    
    # Precompute f at points with single perturbations
    f_i_plus = []
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_i_plus.append(func(x + e_i))
    
    # Compute Hessian elements
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n)
            e_i[i] = eps
            e_j = np.zeros(n)
            e_j[j] = eps
            
            f_ij_plus = func(x + e_i + e_j)
            
            if i == j:
                H[i, j] = (f_ij_plus - 2 * f_i_plus[i] + f0) / (eps * eps)
            else:
                # For off-diagonal, we need f at x + eps*e_i and x + eps*e_j separately
                H[i, j] = (f_ij_plus - f_i_plus[i] - f_i_plus[j] + f0) / (eps * eps)
    
    return H