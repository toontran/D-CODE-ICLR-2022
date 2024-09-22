import sympy
import argparse
import numpy as np

import equations
import data
from gp_utils import run_gp_ode
from interpolate import get_ode_data
import pickle
import os
import time
import basis

import datetime
import os
import pandas as pd
import pytz
import sys

current_dir = os.getcwd()
sys.path.insert(0, os.path.join(current_dir, 
                                'Invariant_Physics'))


def get_now_string(time_string="%Y%m%d_%H%M%S_%f"):
    # return datetime.datetime.now().strftime(time_string)
    est = pytz.timezone('America/New_York')

    # Get the current time in UTC and convert it to EST
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)

    # Return the time in the desired format
    return est_now.strftime(time_string)



def run(ode_name, ode_param, x_id, freq, n_sample, noise_ratio, seed, n_basis, basis_str, ipad_data, args):

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    import pandas as pd
    try:
        df = pd.read_csv(f"{save_dir}/summary_{args.dataset}.csv", header=None,
                        names=['end_time', 'status', 'ODE', 'x_id', 'correct', 'noise', 'env', 'seed', "time_elapsed"]
                        )
        df = df[df["status"] == "End"]
        df = df.astype({
        #     'term_success_predicted_rate': 'float32',
        #     'correct': 'bool',
            'env': 'int32',
            'seed': 'int32',
            'x_id': 'float32',
            'noise': 'float32',
        })
        df["correct"] = df["correct"].apply(lambda x: True if x=="True" else False)
        current_entry = df[(df.seed==seed) & (df.env==args.env) & (df.x_id==x_id) & (df.noise==noise_ratio) &  (df["ODE"]==ode_name)]
        if len(current_entry) == 0:
            pass
        else:
            print(current_entry)
            print("Entry already seen, skipping..")
            return
    except:
        print("Can't find summary in", save_dir)
        pass
    
    np.random.seed(seed)

    ode = equations.get_ode(ode_name, ode_param, args.env, data=ipad_data)
    T = ode.T
    init_low = 0
    init_high = ode.init_high

    if basis_str == 'sine':
        basis_obj = basis.FourierBasis
    else:
        basis_obj = basis.CubicSplineBasis

    noise_sigma = ode.std_base * noise_ratio  
    
    dg = data.DataGenerator(ode, ode_name, T, freq, n_sample, noise_sigma, init_low, init_high, False, args.env, seed=seed, dataset=args.dataset)
    yt = dg.generate_data()
    
    
    print("yt.shape", yt.shape)
    if ipad_data:
        t = ipad_data['t_series_list'][args.env]
    else:
        t = dg.solver.t[:yt.shape[0]]
    ode_data, X_ph, y_ph, t_new = get_ode_data(yt, x_id, t, dg, ode, n_basis, basis_obj,
                                              env = args.env)

    path_base = 'results_vi/{}/noise-{}/sample-{}/freq-{}/n_basis-{}/basis-{}'.\
        format(ode_name, noise_ratio, n_sample, freq, n_basis, basis_str)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    # for s in range(seed, seed+1):
#         print(' ')
    s = seed
    print('Running with seed {}'.format(s))
    start = time.time()
#         print(X_ph.shape, y_ph.shape, x_id)
#         import pdb;pdb.set_trace()

    f_hat, est_gp = run_gp_ode(ode_data, X_ph, y_ph,  ode, x_id, s)

    if ipad_data:
        f_true = ipad_data['params_config']['truth_ode_format'][x_id]
        correct = None
    else:
        f_true = ode.get_expression()[x_id]
        if not isinstance(f_true, tuple):
            correct = sympy.simplify(f_hat - f_true) == 0
        else:
            correct_list = [sympy.simplify(f_hat - f) == 0 for f in f_true]
            correct = max(correct_list) == 1
    print(f_hat, f_true)
#         print(correct_list)
    # results/${ode}/noise-${noise}-seed-${seed}-env-${env}.txt
    # s, f_hat, f_true, x_id
    log_path = f"{save_dir}/summary_{args.dataset}.csv"
    log_end_time = get_now_string()
    end = time.time()
    with open(log_path, "a") as f:
        f.write(f"{log_end_time},{ode_name},{x_id},{correct},{noise_ratio:.6f},{args.env},{s},{end-start},{f_hat},{f_true}\n")
        # f.write(f"{log_end_time},truth,{str(f_true)}\n")
        # f.write(f"{log_end_time},prediction,{str(f_hat)}\n")

    if x_id == 0:
        path = path_base + 'grad_seed_{}.pkl'.format(s)
    else:
        path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)
    

#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'model': est_gp._program,
#                 'ode_data': ode_data,
#                 'seed': s,
#                 'correct': correct,
#                 'f_hat': f_hat,
#                 'ode': ode,
#                 'noise_ratio': noise_ratio,
#                 'noise_sigma': noise_sigma,
#                 'dg': dg,
#                 't_new': t_new,
#                 'time': end - start,
#             }, f)

    print(correct)
    print("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode_name", help="name of the ode", type=str)
    parser.add_argument("--ode_param", help="parameters of the ode (default: None)", type=str, default=None)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--freq", help="sampling frequency", type=float, default=10)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--noise_ratio", help="noise level (default 0)", type=float, default=0.)
    parser.add_argument("--n_basis", help="number of basis function", type=int, default=50)
    parser.add_argument("--basis", help="basis function", type=str, default='sine')

    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--env", help="random seed", type=int, default=1)
    parser.add_argument("--dataset", help=" ", type=str, default="default_1")
    parser.add_argument("--save_dir", help=" ", type=str, default="./result")
    parser.add_argument("--load_ipad_data", help=" ", type=str, default="")
    
    args = parser.parse_args()
    print('Running with: ', args)

    if args.ode_param is not None:
        param = [float(x) for x in args.ode_param.split(',')]
    else:
        param = None
        
    if args.load_ipad_data:
        with open(args.load_ipad_data, 'rb') as file:
            ipad_data = pickle.load(file)
        ode_name = ipad_data['args'].task
        param = None
        x_id = ipad_data['args'].task_ode_num - 1
        n_sample = 1
        noise_ratio = ipad_data['args'].noise_ratio
        seed = ipad_data['args'].seed
    else:
        ode_name = args.ode_name
        param = param
        x_id = args.x_id
        n_sample = args.n_sample
        noise_ratio = args.noise_ratio
        seed = args.seed
        ipad_data = None

    
    run(args.ode_name, param, args.x_id, args.freq, args.n_sample, args.noise_ratio, seed=args.seed, n_basis=args.n_basis, basis_str=args.basis, ipad_data=ipad_data, args=args)
