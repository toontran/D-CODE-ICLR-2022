import numpy as np
import scipy.integrate
import sympy
import abc

# X0 = sympy.Symbol('X0', positive=True)
# X1 = sympy.Symbol('X1', positive=True)
# X2 = sympy.Symbol('X2', positive=True)
# X3 = sympy.Symbol('X3', positive=True)
# C = sympy.Symbol('C', positive=True)
#
# VarDict = {
#     'X0': X0,
#     'X1': X1,
#     'X2': X2,
#     'X3': X3,
#     'C': C,
# }

def get_ode(ode_name, param, env=0, data=None):
    if data is not None:
        ode = IpadODE(param, data=data)
    elif ode_name == 'SineWave':
        ode = SineWave(param)
    elif ode_name == 'VdpODE':
        ode = VdpODE(param)
    elif ode_name == 'MMODE':
        ode = MMODE(param)
    elif ode_name == 'GompertzODE':
        ode = GompertzODE(param)
    elif ode_name == 'LogisticODE':
        ode = LogisticODE(param)
    elif ode_name == 'HillODE':
        ode = HillODE(param)
    elif ode_name == 'SirODE':
        ode = SirODE(param, env=env)
    elif ode_name == 'LvODE':
        ode = LvODE(param, env=env)
    elif ode_name == 'ThetaModel':
        ode = ThetaModel(param)
    elif ode_name == 'LIF':
        ode = LIF(param)
    elif ode_name == 'HopfODE':
        ode = HopfODE(param)
    elif ode_name == 'SelkovODE':
        ode = SelkovODE(param)
    elif ode_name == 'Brusselator':
        ode = Brusselator(param)
    elif ode_name == 'FHN':
        ode = FHN(param)
    elif ode_name == 'Lorenz':
        ode = Lorenz(param, env=env)
    elif ode_name == 'FracODE':
        ode = FracODE(param)
    else:
        raise ValueError('{} is not a supported ode.'.format(ode_name))
    return ode

def get_var_pos():
    X0 = sympy.Symbol('X0', positive=True)
    X1 = sympy.Symbol('X1', positive=True)
    X2 = sympy.Symbol('X2', positive=True)
    X3 = sympy.Symbol('X3', positive=True)
    C = sympy.Symbol('C', positive=True)

    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'C': C,
    }
    return VarDict

def get_var_real():
    X0 = sympy.Symbol('X0', real=True)
    X1 = sympy.Symbol('X1', real=True)
    X2 = sympy.Symbol('X2', real=True)
    X3 = sympy.Symbol('X3', real=True)
    C = sympy.Symbol('C', positive=True)
    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'C': C,
    }
    return VarDict


class ODE(metaclass=abc.ABCMeta):
    def __init__(self, dim_x, param=None, env=0):
        self.dim_x = dim_x
        self.env = env
        self.has_coef = param is not None
        self.param = param if self.has_coef else self.get_default_param()
        self.T = 5
        self.init_high = 0.1
        self.init_low = 0
        self.std_base = 1.
        self.positive = True

    @abc.abstractmethod
    def get_default_param(self):
        pass

    @abc.abstractmethod
    def _dx_dt(self, *args):
        pass

    @staticmethod
    def get_var_dict():
        return get_var_pos()

    @abc.abstractmethod
    def get_expression(self):
        pass

    def dx_dt(self, t, x):
        arg_list = list()
        for i in range(self.dim_x):
            arg_list.append(x[i])
        return self._dx_dt(*arg_list)

    def dx_dt_batch(self, t, x):
        arg_list = list()
        for i in range(self.dim_x):
            arg_list.append(x[:, :, i])
        return np.stack(self._dx_dt(*arg_list), axis=-1)

    @abc.abstractmethod
    def functional_theta(self, theta):
        pass


class ODESolver:
    def __init__(self, ode, T, freq, return_list=False, integrator='lsoda'):
        self.ode = ode
        self.integrator = integrator
        # if T > freq:
        #     assert T % freq == 0
        self.T = T
        self.freq = freq
        n_step = int(T * freq)
        self.n_step = n_step
        self.dt = 1 / freq
        self.t = np.arange(0, T + self.dt, self.dt)
        self.return_list = return_list

    def solve_one(self, init):
        ode = scipy.integrate.ode(self.ode.dx_dt).set_integrator(self.integrator)
        ode.set_initial_value(init, 0)

        res_list = [init]

        while ode.successful() and ode.t < self.T:
            res = ode.integrate(ode.t + self.dt)
            res_list.append(res)
        res = np.stack(res_list, axis=-1)
        # D, T
        return res

    def solve(self, init_list):
        res_list = []
        for init in init_list:
            res_list.append(self.solve_one(init))
        # B, D, T -> T, B, D
        # print({arr.shape for arr in res_list})
        if not self.return_list:
            res = np.stack(res_list, axis=0).transpose((2, 0, 1))
            res_t = res.shape[0]
            true_t = len(self.t)
            if res_t > true_t:
                res = res[(res_t - true_t):, ...]
            return res
        else:
            return res_list

class InferredODE(ODE):
    def __init__(self, dim_x, param=None, f_hat_list=None, T=None):
        super().__init__(dim_x, param)
        assert len(f_hat_list) == dim_x
        self.f_hat_list = f_hat_list
        self.T = T

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, X):
        return None

    def get_expression(self):
        return None

    def functional_theta(self, theta):
        return None

    def dx_dt(self, t, x):
        return [f(x[None, :]) for f in self.f_hat_list]



class LinearODE(ODE):
    def __init__(self, dim_x, param=None):
        super().__init__(dim_x, param)
        self.beta = self.param[0]
        assert self.beta.shape[0] == self.beta.shape[1]
        assert self.beta.shape[0] == dim_x

    def get_default_param(self):
        return [np.array([[0, 1], [-1, 0]])]

    def _dx_dt(self, t, x):
        pass

    def dx_dt(self, t, x):
        x = np.array(x)
        return np.matmul(x, self.beta)

    def dx_dt_batch(self, t, x):
        # x: T, B, D
        # beta: D, D
        # out: T, B, D
        return np.matmul(x, self.beta)

    def functional_theta(self, theta):
        new_ode = LinearODE(self.dim_x, theta)
        return new_ode.dx_dt_batch

    def get_expression(self):
        raise NotImplementedError


class SineWave(LinearODE):
    def __init__(self, param=None):
        # param: [1.0]
        dim_x = 2
        if param is not None:
            param = [np.array([[0, 1], [-1 * param[0], 0]])]

        super().__init__(dim_x, param)
        self.has_coef = param is not None

    def get_default_param(self):
        return [np.array([[0, 1], [-1, 0]])]

    @staticmethod
    def get_var_dict():
        return get_var_real()

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']

        if self.has_coef:
            eq1 = X1
            eq2 = -1 * C * X0
        else:
            eq1 = X1
            eq2 = -1 * X0
        return [eq1, eq2]


class VdpODE(ODE):
    """
    van der pol equation
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Forced_Van_der_Pol_oscillator
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.mu = self.param[0]
        self.init_high = 1.
        if not self.has_coef:
            self.T = 25

    def get_default_param(self):
        return [1.]

    def _dx_dt(self, x0, x1):
        dx = 1 / self.mu * (x1 - 1.0 / 3.0 * x0 ** 3 + x0)
        dy = -x0
        return [dx, dy]

    @staticmethod
    def get_var_dict():
        return get_var_real()

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']

        if self.has_coef:
            eq1 = C * (X1 - C * X0 * X0 * X0 + X0)
        else:
            eq1 = X1 - C * X0 * X0 * X0 + X0
        eq2 = -1 * X0
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) in [1]
        new_ode = VdpODE(theta[0])
        return new_ode.dx_dt_batch


class MMODE(ODE):
    """
    Michaelis–Menten kinetics
    https://https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics
    """

    def __init__(self, param=None):
        super().__init__(4, param)
        self.k_f, self.k_r, self.k_cat = self.param
        self.init_high = 1.

    def _dx_dt(self, x0, x1, x2, x3):
        dedt = -1 * self.k_f * x0 * x1 + self.k_r * x2 + self.k_cat * x2
        dsdt = -1 * self.k_f * x0 * x1 + self.k_r * x2
        desdt = self.k_f * x0 * x1 - self.k_r * x2 - self.k_cat * x2
        dpdt = self.k_cat * x2
        return [dedt, dsdt, desdt, dpdt]

    def get_default_param(self):
        return 1., 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = -1 * C * X0 * X1 + C * X2
            eq2 = -1 * C * X0 * X1 + C * X2
            eq3 = X0 * X1 - C * X2
            eq4 = C * X2
        else:
            eq1 = -1 * X0 * X1 + C * X2
            eq2 = -1 * X0 * X1 + X2
            eq3 = X0 * X1 - C * X2
            eq4 = X2
        return eq1, eq2, eq3, eq4

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = MMODE(theta)
        return new_ode.dx_dt_batch


class GompertzODE(ODE):
    """
    Gompertz
    https://en.wikipedia.org/wiki/Gompertz_function
    """

    def __init__(self, param=None):
        super().__init__(1, param)
        self.a, self.b = self.param
        self.init_high = 0.01
        if self.has_coef:
            self.T = 4
        else:
            self.T = 5
        self.name = 'GompertzODE'
        self.std_base = 0.23405202469895242

    def _dx_dt(self, X):
        dxdt = -1 * self.a * X * np.log(self.b * X)
        return [dxdt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        C = var_dict['C']
        if self.has_coef:
            # eq1 = -1 * C * X0 * sympy.log(X0) - C * X0
            eq1 = -1 * C * X0 * sympy.log(C * X0)
        else:
            eq1 = -1 * X0 * sympy.log(X0)
        return [eq1]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = GompertzODE(theta)
        return new_ode.dx_dt_batch


class LogisticODE(ODE):
    """
    Logistic
    https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation
    """

    def __init__(self, param=None):
        super().__init__(1, param)
        self.a, self.k = self.param
        self.T = 10
        self.has_coef = True
        self.name = 'LogisticODE'
        self.std_base = 0.31972985438694346

    def _dx_dt(self, X):
        dxdt = self.a * (1 - np.power(X, self.k)) * X
        return [dxdt]

    def get_default_param(self):
        return 1., 0.5

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 - X0 ** C
        else:
            eq1 = (1 - X0) * X0
        return [eq1]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = LogisticODE(theta)
        return new_ode.dx_dt_batch


class HillODE(ODE):
    """
    Hill_equation_(biochemistry)
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.n, self.k, self.ka, self.ky = self.param
        self.init_high = 10.

    def _dx_dt(self, X, Y):
        dxdt = self.k * np.power(Y, self.n) / (self.ka + np.power(Y, self.n))
        dydt = -1 * self.ky * Y
        return [dxdt, dydt]

    def get_default_param(self):
        return 1., 1., 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * sympy.Pow(X1, C) / (C + sympy.Pow(X1, C))
            eq2 = -1 * C * X1
        else:
            eq1 = X1 / (C + X1)
            eq2 = -1 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = HillODE(theta)
        return new_ode.dx_dt_batch


class ThetaModel(ODE):
    """
    Theta model
    https://en.wikipedia.org/wiki/Theta_model#General_equations
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b, self.k = self.param
        self.init_high = 1.
        self.T = 10

    def get_default_param(self):
        return [1., 1, 0]

    def _dx_dt(self, x, y):
        # dxdt = 1 - np.cos(self.a * x) + (1 + np.cos(self.a * x)) * self.k
        dxdt = 1 - np.cos(self.a * x) + (1 + np.cos(self.a * x)) * np.sin(self.b * y) * self.k
        dydt = 1
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - sympy.cos(C * X0) + (1 + sympy.cos(C * X0)) * sympy.sin(C * X1) * C
        else:
            eq1 = 1 - sympy.cos(X0)
        eq2 = 1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = ThetaModel(theta)
        return new_ode.dx_dt_batch


class LIF(ODE):
    """
    Leaky integrate and fire
    https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.c, self.r = self.param
        print(self.param)

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, x, y):
        dxdt = self.c * y - self.r * x
        dydt = 0
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * X1 - C * X0
        else:
            eq1 = X1 - X0
        eq2 = 0
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = LIF(theta)
        return new_ode.dx_dt_batch


class HopfODE(ODE):
    """
    2D Hopf normal form
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.mu, self.omega, self.A = self.param
        self.T = 10

    def get_default_param(self):
        return 1., 1., 0.

    def _dx_dt(self, x, y):
        dxdt = self.mu * x + self.omega * y - self.A * (x ** 2 + y ** 2)
        dydt = -1 * self.omega * x + self.mu * y - self.A * (x ** 2 + y ** 2)
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * X0 + C * X1 - C * (X0 ** 2 + X0 ** 2)
            eq2 = -1 * C * X0 + C * X1 - C * (X0 ** 2 + X0 ** 2)
        else:
            eq1 = X0 + X1 - (X0 ** 2 + X0 ** 2)
            eq2 = -1 * X0 + X1 - (X0 ** 2 + X0 ** 2)
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = HopfODE(theta)
        return new_ode.dx_dt_batch


class SelkovODE(ODE):
    """
    Selkov model for Glycolysis
    https://services.math.duke.edu/ode-book/xppaut/ch4-selkov.pdf
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.rho, self.sigma = self.param
        self.has_coef = True
        self.name = 'SelkovODE'
        self.std_base = 0.5641061
        self.T = 15

    def get_default_param(self):
        return 0.75, 0.1

    def _dx_dt(self, x, y):
        dxdt = self.rho - self.sigma * x - x * y * y
        dydt = -1 * y + self.sigma * x + x * y * y
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - C * X0 - X0 * X1 * X1
            eq2 = -1 * X1 + C * X0 + X0 * X1 * X1
        else:
            eq1 = 1 - X0 - X0 * X1 * X1
            eq2 = -1 * X1 + X0 + X0 * X1 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = SelkovODE(theta)
        return new_ode.dx_dt_batch



class FracODE(ODE):
    def __init__(self, param=None):
        super().__init__(2, param)
        self.rho = self.param
        self.has_coef = True
        self.name = 'FracODE'
        self.std_base = 2.5388
        self.T = 3
        self.init_high = 1.
        self.init_low = 0.8

    def get_default_param(self):
        return 1.

    def _dx_dt(self, x, y):
        dxdt = 1 / (y + self.rho) - x
        dydt = 1
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 ** C / (X1 + C), X0 ** 2 / (X1 + C), X0 ** 2 / (X1 + 1)
            eq2 = C
        else:
            eq1 = X0 ** 2 / X1
            eq2 = C
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = FracODE(theta)
        return new_ode.dx_dt_batch


class Brusselator(ODE):
    """
    Brusselator
    https://www.bibliotecapleyades.net/archivos_pdf/brusselator.pdf
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b = self.param
        self.T = 15
        self.init_high = 1.

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, x, y):
        dxdt = 1. - (self.b + 1) * x + self.a * x * x * y
        dydt = self.b * x - self.a * x * x * y
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - C * X0 + C * X0 * X0 * X1
            eq2 = C * X0 - C * X0 * X0 * X1
        else:
            eq1 = C - C * X0 + X0 * X0 * X1
            eq2 = X0 - X0 * X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = Brusselator(theta)
        return new_ode.dx_dt_batch


class FHN(ODE):
    """
        FitzHugh–Nagumo model
        https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b = self.param
        self.init_high = 2.
        self.T = 25

    def get_default_param(self):
        return 1., 0.

    def _dx_dt(self, x, y):
        dxdt = x - 1. / 3. * x * x * x - y
        dydt = x + self.a - self.b * y
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 - C * X0 * X0 * X0 - X1
            eq2 = X0 + C - C * X1
        else:
            eq1 = X0 - C * X0 * X0 * X0 - X1
            eq2 = X0 + C
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = FHN(theta)
        return new_ode.dx_dt_batch


class RealODEPlaceHolder:
    def __init__(self):
        self.name = 'real'
        self.std_base = 0.281234968452
        self.positive = False

    @staticmethod
    def get_var_dict():
        X0 = sympy.Symbol('X0', real=True)
        X1 = sympy.Symbol('X1', positive=True)
        C = sympy.Symbol('C', positive=True)

        VarDict = {
            'X0': X0,
            'X1': X1,
            'C': C,
        }
        return VarDict

class SirODE(ODE):
    """
    "random_params_base": [0.010, 0.050],
    "default_params_list": [
        [0.010, 0.050],
        [0.011, 0.040],
        [0.012, 0.043],
        [0.013, 0.045],
        [0.014, 0.047],
        [0.009, 0.060],
        [0.008, 0.058],
        [0.007, 0.056],
        [0.014, 0.054],
        [0.015, 0.052],
    ],
    "random_y0_base": [50.0, 40.0, 10.0],
    "default_y0_list": [
        [50.0, 40.0, 10.0],
        [50.5, 41.0, 8.5],
        [55.0, 40.0, 5.0],
        [47.5, 48.5, 4.0],
        [48.0, 49.0, 3.0],
        [43.0, 51.0, 6.0],
        [47.0, 46.0, 7.0],
        [48.5, 43.5, 8.0],
        [49.0, 42.0, 9.0],
        [30.0, 55.0, 15.0],
    ],
    "truth_ode_format": [
        "-{0}*x*y",
        "{0}*x*y-{1}*y",
        "{1}*y",
    ]
    dy = np.asarray([
            - beta * x[0] * x[1],
            beta * x[0] * x[1] - gamma * x[1],
            gamma * x[1]
        ])
    SIR model
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    """

    def __init__(self, param=None, env=0):
        super().__init__(3, param, env)
        self.b, self.k = self.param
        self.init_high = 1.
        self.T = 96
        self.name = "SirODE"
        self.has_coef = True
        self.positive = False

    def _dx_dt(self, x0, x1, x2):
        dsdt = -1 * self.b * x0 * x1
        didt = self.b * x0 * x1 - self.k * x1
        drdt = self.k * x1
        return [dsdt, didt, drdt]

    def get_default_param(self):
#         return 1., 1.
        default_param = [[0.010, 0.050],
                            [0.011, 0.040],
                            [0.012, 0.043],
                            [0.013, 0.045],
                            [0.014, 0.047],]
        return default_param[self.env]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (-1 * C * X0 * X1,
                   -1 * C * X0 * X1+C,
                  -1 * C * X0 * X1-C,
                   C * X0 * X1,
                  C * X0 * X1+C,
                  C * X0 * X1-C)
            eq2 = (C * X0 * X1 - C * X1,
                   C * X0 * X1 - C * X1+C,
                   C * X0 * X1 - C * X1-C,
                  -1*C * X0 * X1 + C * X1,
                  -1*C * X0 * X1 + C * X1+C,
                  -1*C * X0 * X1 + C * X1-C,)
            eq3 = (C * X1,
                   C * X1+C,
                   C * X1-C,
                  -1*C * X1,
                  -1*C * X1+C,
                  -1*C * X1-C,)
        else:
            eq1 = -1 * X0 * X1
            eq2 = X0 * X1 - X1
            eq3 = X1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = SirODE(theta)
        return new_ode.dx_dt_batch
    
class LvODE(ODE):
    """
    "random_params_base": [1.00, 0.30, 3.00, 0.10],
    "default_params_list": [
        [1.00, 0.30, 3.00, 0.10],
        [1.20, 0.39, 2.80, 0.09],
        [1.30, 0.42, 3.20, 0.08],
        [1.10, 0.51, 3.10, 0.11],
        [0.90, 0.39, 2.90, 0.12],
        [0.85, 0.35, 2.85, 0.13],
        [1.15, 0.28, 3.30, 0.14],
        [1.25, 0.40, 3.05, 0.07],
        [0.95, 0.43, 2.65, 0.06],
        [1.40, 0.26, 3.35, 0.15],
    ],
    "random_y0_base": [10.00, 5.00],
    "default_y0_list": [
        [10.00, 5.00],
        [9.60, 4.30],
        [10.10, 5.10],
        [10.95, 4.85],
        [8.90, 6.10],
        [11.10, 5.45],
        [8.75, 5.85],
        [11.40, 4.95],
        [10.05, 5.35],
        [8.85, 5.20],
    ],
    "truth_ode_format": ["{0}*x-{1}*x*y", "{3}*x*y-{2}*y"],
    alpha, beta, gamma, delta = iter(self.params[env_id])
        dy = np.asarray([
            alpha * x[0] - beta * x[0] * x[1],
            delta * x[0] * x[1] - gamma * x[1],
        ])
    Lotka-Volterra equations
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    """

    def __init__(self, param=None, env=0):
        super().__init__(2, param, env)
#         self.a, self.c, self.gamma = self.param
        self.alpha, self.beta, self.gamma, self.delta = self.param
        self.init_high = 1.
        self.T = 16
        self.name = 'LvODE'
        self.has_coef = True
        self.positive = False

    def _dx_dt(self, X, Y):
#         dxdt = self.a * X - self.a * X * Y
#         dydt = -1. * self.c * Y + self.gamma * X * Y
        dxdt = self.alpha * X - self.beta * X * Y
        dydt = -1. * self.gamma * Y + self.delta * X * Y
        return [dxdt, dydt]

    def get_default_param(self):
#         return 1., 1., 1
        default_params = [
            [1.00, 0.30, 3.00, 0.10],
            [1.20, 0.39, 2.80, 0.09],
            [1.30, 0.42, 3.20, 0.08],
            [1.10, 0.51, 3.10, 0.11],
            [0.90, 0.39, 2.90, 0.12],
        ]
        return default_params[self.env]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C * X0 - C * X0 * X1,
                   C * X0 - C * X0 * X1+C,
                   C * X0 - C * X0 * X1-C,
                   -1*C * X0 + C * X0 * X1,
                   -1*C * X0 + C * X0 * X1+C,
                   -1*C * X0 + C * X0 * X1-C)
            eq2 = (-1 * C * X1 + C * X0 * X1,
                   -1 * C * X1 + C * X0 * X1+C,
                   -1 * C * X1 + C * X0 * X1-C,
                   C * X1 - C * X0 * X1,
                  C * X1 - C * X0 * X1+C,
                  C * X1 - C * X0 * X1-C,)
        else:
            eq1 = X0 - X0 * X1
            eq2 = -1 * X1 + X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = LvODE(theta)
        return new_ode.dx_dt_batch
    
    
class Lorenz(ODE):
    """
    "random_params_base": [28.00, 10.00, 2.67],
    "default_params_list": [
        [28.00, 10.00, 2.67],
        [13.00, 9.90, 2.70],
        [15.00, 10.20, 2.74],
        [14.00, 10.50, 2.54],
        [19.00, 8.90, 3.00],
        [24.00, 9.70, 2.84],
        [18.00, 9.30, 2.60],
        [25.00, 9.80, 2.67],
        [17.00, 10.10, 2.76],
        [15.50, 10.40, 2.61],
    ],
    "random_y0_base": [6.00, 6.00, 15.00],
    "default_y0_list": [
        [6.00, 6.00, 15.00],
        [5.00, 7.00, 12.00],
        [5.80, 6.30, 17.00],
        [6.05, 6.40, 14.00],
        [6.25, 6.50, 11.00],
        [6.30, 6.10, 10.00],
        [6.20, 6.80, 18.00],
        [6.10, 6.90, 19.00],
        [5.90, 6.60, 20.00],
        [5.80, 5.80, 12.50],
    ],
    "truth_ode_format": [
        "{1}*y-{1}*x",
        "{0}*x-x*z-y",
        "x*y-{2}*z",
    ]
    rho, sigma, beta = iter(self.params[env_id])
        dy = np.asarray([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
        Lorenz System
    """
    def __init__(self, param=None, env=0):
        super().__init__(3, param, env)
        self.rho, self.sigma, self.beta = self.param
        self.T = 3
        self.has_coef = True
        self.init_high = 10
        self.std_base = 8.55515291
        self.name = 'Lorenz'
        self.positive = False

    def get_default_param(self):
#         return 10, 28, 8/3
        default_params = [
            [28.00, 10.00, 2.67],
            [13.00, 9.90, 2.70],
            [15.00, 10.20, 2.74],
            [14.00, 10.50, 2.54],
            [19.00, 8.90, 3.00],
        ]
        print(default_params[self.env])
        return default_params[self.env]

    def _dx_dt(self, x, y, z):
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = C * (X1 - X0)
            eq2 = (X0 * (C - X2) - X1, 
                   X0*(C - X2) + X0 - X1,
                   C*X0 - C*X1 - X0*X2, 
                   -C*X1 + X0*(C - X2) + X0, 
                   -C*X1 + X0*(C - X2) + C*X0)
            eq3 = X0 * X1 - C * X2
        else:
            eq1 = X1 - X0
            eq2 = -1. * X0 * X2 - X1
            eq3 = X0 * X1 - X2
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = Lorenz(theta)
        return new_ode.dx_dt_batch

class IpadODE(ODE):
    """
    Placeholder for IPAD
    """
    def __init__(self, param=None, env=0, data=None):
        super().__init__(3, param, env)
        if not data:
            raise Exception("Need to provide data")
        # self.rho, self.sigma, self.beta = self.param
        self.data = data
        self.T = data['params_config']['t_max']
        self.has_coef = True
        self.init_high = -1
        self.name = data['params_config']['task']
        self.expression = data['params_config']['truth_ode_format']

#     def get_default_param(self):
# #         return 10, 28, 8/3
#         default_params = [
#             [28.00, 10.00, 2.67],
#             [13.00, 9.90, 2.70],
#             [15.00, 10.20, 2.74],
#             [14.00, 10.50, 2.54],
#             [19.00, 8.90, 3.00],
#         ]
#         print(default_params[self.env])
#         return default_params[self.env]

    def get_expression(self):
        return self.expression

    def _dx_dt(self, x, y, z):
        return

    def functional_theta(self, theta):
        return

    def get_default_param(self):
        return

# class Vortex(ODE):
#     """
#     Vortex shedding
#     """
#
#     def __init__(self, param=None):
#         super().__init__(3, param)
#         self.mu, self.omega, self.A, self.lam = self.param
#
#     def get_default_param(self):
#         return 1., 1., 1., 1.
#
#     def _dx_dt(self, x, y, z):
#         dxdt = self.mu * x - self.omega * y + self.A * x * z
#         dydt = self.omega * x + self.mu * y + self.A * y * z
#         dzdt = x * x + y * y - z
#         return [dxdt, dydt, dzdt]
#
#     def get_expression(self):
#         if self.has_coef:
#             eq1 = C * X0 - C * X1 + C * X0 * X2
#             eq2 = C * X0 + C * X1 + C * X1 * X2
#             eq3 = C * (X0 * X0 + X1 * X1 - X2)
#         else:
#             eq1 = X0 - X1 + X0 * X2
#             eq2 = X0 + X1 + X1 * X2
#             eq3 = X0 * X0 + X1 * X1 - X2
#         return [eq1, eq2, eq3]
#
#     def functional_theta(self, theta):
#         assert len(theta) == 3
#         new_ode = Vortex(theta)
#         return new_ode.dx_dt_batch




