import numpy as np
import equations
import pickle

import random

def get_train_test_total_list(train_test_total: str, num_env: int, seed=None):
    """
    Args:
        train_test_total: supports three types of argument:
            (1) one integer, like 500, indicating it's a balanced dataset;
            (2) integers split by "/", like 500/400/300/200/100. There would be an error if its length mismatches the num_env;
            (3) a string "default_x", following a built-in dict as below.
        num_env: an integer, number of environment
        seed: an integer for generating random ordering list
    Returns:
        a list of environment size, e.g., [500, 400, 300, 200, 100].
    """
    default_dic = {
        "default_0": "500/500/500/500/500",
        "default_1": "20/20/20/20/20",
        "default_2": "100/100/100/100/100",
        "default_3": "500/400/40/20/10",
        "default_4": "500/80/40/20/10",
        "default_5": "100/50/50/20/10",
        "default_6": "500/400/300/200/100",
        "default_7": "500/400/300/200/10",
        "default_8": "500/400/300/20/10",
        "default_10": "200/200/200/200/200",
        "default_11": "500/400/40/40/20",
        "default_12": "500/200/200/50/50",
        "default_13": "500/125/125/125/125",
    }

    if train_test_total.isdigit():
        train_test_total_list = [int(train_test_total) for i in range(num_env)]
    else:
        if "/" not in train_test_total:
            assert train_test_total in default_dic, "Error: key error in get_train_test_total_list: " + str(train_test_total)
            string = default_dic[train_test_total]
        else:
            string = train_test_total
        parts = string.split("/")
        train_test_total_list = [int(item) for item in parts]
    assert len(train_test_total_list) == num_env, "Error: mismatching between " + str(train_test_total) + " and " + str(num_env)
    # print(train_test_total_list)
    one_order = generate_random_order(num_env, seed)
    train_test_total_list_swapped = reseat(train_test_total_list, one_order)
    print(f"Swap dataset size: {train_test_total_list} -> {train_test_total_list_swapped}")
    return train_test_total_list_swapped


def generate_random_order(n, seed=None):
    if seed is not None:
        random.seed(seed)
    numbers = list(range(n))
    random.shuffle(numbers)
    return numbers


def reseat(one_list, one_order):
    assert len(one_list) == len(one_order)
    return [one_list[one_order[i]] for i in range(len(one_order))]


class DataGenerator:
    def __init__(self, ode, ode_name, T, freq, n_sample, noise_sigma, init_low=0., init_high=1., return_list=False, env=0, seed=None, dataset="default_1"):
        self.ode = ode
        self.T = T
        self.noise_sigma = noise_sigma
        self.return_list = return_list

        if isinstance(ode, equations.IpadODE):
            self.init_cond = ode.data['data_train']["y0_list"][env]
            self.yt = ode.data['data_train']["y_noise"][env]
            self.xt = ode.data['data_train']["y"][env]
            print(self.yt.shape)
            self.freq = 1 / (ode.data['t_series_list'][0][1] - ode.data['t_series_list'][0][0])
            return
        elif ode_name == "Lorenz":
            default_init = [[6.00, 6.00, 15.00],
                            [5.00, 7.00, 12.00],
                            [5.80, 6.30, 17.00],
                            [6.05, 6.40, 14.00],
                            [6.25, 6.50, 11.00],]
            solver_default_freq = 1000
        elif ode_name == "SirODE":
            default_init = [[50.0, 40.0, 10.0],
                            [50.5, 41.0, 8.5],
                            [55.0, 40.0, 5.0],
                            [47.5, 48.5, 4.0],
                            [48.0, 49.0, 3.0],]
            solver_default_freq = 100
        elif ode_name == "LvODE":
            default_init = [[10.00, 5.00],
                            [9.60, 4.30],
                            [10.10, 5.10],
                            [10.95, 4.85],
                            [8.90, 6.10],]
            solver_default_freq = 100
        else:
            raise NotImplemented
        
        # TODO: set seed for DataGenerator
        # with size of number of envs
        sizes = get_train_test_total_list(dataset, 5, seed=seed)
        self.freq = sizes[env] / T     
        print("T, Freq:", T, self.freq)
        # self.freq = freq
        solver_freq = self.freq * int(solver_default_freq/self.freq)# should be high enough to represent true data
        print("Solver Freq and multiplier:", solver_freq, int(solver_default_freq/self.freq))
        self.solver = equations.ODESolver(ode, T+1/self.freq, solver_freq, return_list=return_list)
        
        
        self.init_cond = np.array([default_init[env]])
        print(self.init_cond)

        self.xt = self.solver.solve(self.init_cond)[::int(solver_default_freq/self.freq), :, :][:-1,:,:]
        print("xt:", self.xt.shape)
        # print(self.solver.t.shape)
        # print(self.solver.t[::int(solver_default_freq/self.freq)])
        # raise
        if not self.return_list:
            self.eps = np.random.randn(*self.xt.shape) * noise_sigma
            self.yt = self.xt + self.eps
#             self.eps = np.random.random(self.xt.shape) * noise_sigma
#             self.yt = self.xt + self.eps*self.xt
        self.solver = equations.ODESolver(ode, T+1/self.freq-1e-6, self.freq, return_list=return_list)
        print(self.solver.t, self.solver.t.shape)

    def generate_data(self):
        return self.yt

class DataGeneratorReal:
    def __init__(self, dim_x, n_train):

        with open('data/real_data_c1.pkl', 'rb') as f:
            y_total = pickle.load(f)

        with open('data/real_data_mask_c1.pkl', 'rb') as f:
            mask = pickle.load(f)

        if dim_x == 1:
            self.yt = y_total[:, :, 0:1]
        else:
            self.yt = y_total

        self.mask = mask

        self.xt = self.yt.copy()

        self.yt_train = self.yt[:, :n_train, :].copy()
        self.yt_test = self.yt[:, n_train:, :].copy()

        self.mask_test = self.mask[:, n_train:].copy()

        # self.T = y_total.shape[0] - 1
        # self.solver = equations.ODESolver(equations.RealODEPlaceHolder(), self.T, 1.)

        self.T = 1.
        self.solver = equations.ODESolver(equations.RealODEPlaceHolder(), self.T, 364)
        self.noise_sigma = 0.001
        self.freq = 364

    def generate_data(self):
        return self.yt_train
