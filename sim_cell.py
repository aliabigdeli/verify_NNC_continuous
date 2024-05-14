import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import onnxruntime as ort
import math
import h5py
from scipy.integrate import RK45
import argparse
from scipy.io import loadmat
import os
import sys


onnx_model_path = 'models/papermodel/full_mlp_supervised.onnx'
sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
onnx_model_path_all = 'models/papermodel/AllInOne.onnx'
sess_all = ort.InferenceSession(onnx_model_path_all, providers=['CPUExecutionProvider'])



def nncontroller_der(t, y):
    '''derivative function of the neural network controller(cGAN+ImgaeController+P-Controller)'''
    v = 5
    L = 5
    pval, tval = y[0], y[1]

    # Run inference and get the model's output
    ub, lb = 0.8, -0.8
    l_var1 = np.random.uniform(lb, ub) 
    l_var2 = np.random.uniform(lb, ub)
    # l_var1, l_var2 = 0.0, 0.0
    pval, tval = y[0], y[1]
    input_data = np.array([l_var1, l_var2, pval/6.36615, tval/17.247995], dtype=np.float32)
    # input_data = np.array([l_var1, l_var2, pval/6.36615, tval/17.247995], dtype=np.float32).reshape(1, 4)
    output = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: input_data})

    # phi = output[0][0,0]
    pval, tval = output[0][0], output[0][1]

    phi = -0.74*pval - 0.44*tval    # phi = steering angle (control command)
    
    der_p = v*np.sin(np.pi/180 *tval)
    der_t = 180/np.pi * (v/L)*np.tan(np.pi/180 *phi)
    
    return [der_p, der_t]

def next_state(s, dt, t_end=1):
    p, theta = s
    y0 = [p, theta]
    t0 = 0

    integrator = RK45(nncontroller_der, t0, y0, t_bound=t_end, max_step=dt)

    t_values = [t0]
    y_values = [y0]

    # Integrate the ODE
    while integrator.status == 'running':
        integrator.step()
        t_values.append(integrator.t)
        y_values.append(integrator.y)

    ps = [pt[0] for pt in y_values]
    ts = [pt[1] for pt in y_values]

    s_ = [ps[-1], ts[-1]]
    return s_, ps, ts


def main(bloutcoef, cell_idx_p, cell_idx_theta, uncerteainty_bloat=0.1, freqmode="inf"):
    # cell partitions
    p_range = [-10, 10]
    p_num_bin = 128
    theta_range = [-30, 30]
    theta_num_bin = 128
    p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:],dtype=np.float32)
    theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:],dtype=np.float32)


    ##### random sample states in specific cell
    p_lb, p_ub = p_lbs[cell_idx_p], p_ubs[cell_idx_p]
    theta_lb, theta_ub = theta_lbs[cell_idx_theta], theta_ubs[cell_idx_theta]
    print(f"p_lb, p_ub: {p_lb, p_ub}")
    print(f"theta_lb, theta_ub: {theta_lb, theta_ub}")
    ps = np.random.uniform(p_lb, p_ub, [300,1])
    thetas = np.random.uniform(theta_lb, theta_ub, [300,1])
    ##### add four corners
    ps[0, 0] = p_lb
    ps[1, 0] = p_ub
    ps[-2, 0] = p_lb
    ps[-1, 0] = p_ub
    thetas[0, 0] = theta_lb
    thetas[1, 0] = theta_lb
    thetas[-2, 0] = theta_ub
    thetas[-1, 0] = theta_ub
    states = np.concatenate([ps, thetas], axis=1)

    random_seed_value = 2
    np.random.seed(random_seed_value)
    t_end = 1
    dt = t_end          # dt(max time step) = t_end, cuz we don't need trajectories
    states_ = []
    trajectory_p = []
    trajectory_t = []
    for s in states:
        s_, trp, trt = next_state(s, dt=dt, t_end=t_end)
        states_.append(s_)
        trajectory_p.append(trp)
        trajectory_t.append(trt)
    states_ = np.array(states_)

    f_p_max = np.max(states_[:, 0])
    f_p_min = np.min(states_[:, 0])
    f_t_max = np.max(states_[:, 1])
    f_t_min = np.min(states_[:, 1])
    print(f"f_p_min, f_p_max: {f_p_min, f_p_max}")
    print(f"f_t_min, f_t_max: {f_t_min, f_t_max}")


    # find valid region max and min
    v_p_max = np.max([f_p_max, p_ub])
    v_p_min = np.min([f_p_min, p_lb])
    v_t_max = np.max([f_t_max, theta_ub])
    v_t_min = np.min([f_t_min, theta_lb])


    # blout valid region by 'bloutcoef'
    blout_p = (v_p_max-v_p_min)*bloutcoef
    blout_t = (v_t_max-v_t_min)*bloutcoef

    print(f'blouted ranges')
    print(f'p: {v_p_min-blout_p, v_p_max+blout_p}')
    print(f't: {v_t_min-blout_t, v_t_max+blout_t}')

    if freqmode == "inf":
        fitLinear(v_p_min-blout_p, v_p_max+blout_p, v_t_min-blout_t, v_t_max+blout_t, f_p_min, f_p_max, f_t_min, 
                  f_t_max, cell_idx_p, cell_idx_theta, uncerteainty_bloat)
    elif freqmode == "fixed":
        fitLinear(p_lb, p_ub, theta_lb, theta_ub, f_p_min, f_p_max, f_t_min, f_t_max, cell_idx_p, cell_idx_theta, uncerteainty_bloat)
    else:
        print(f"Invalid freqmode: {freqmode}")
    
 # Function to calculate coefficients using the normal equation
def calculate_coefficients(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

# Function to make predictions
def predict(X, coefficients):
    return X.dot(coefficients)
    

def fitLinear(p_lb, p_ub, theta_lb, theta_ub, f_p_min, f_p_max, f_t_min, f_t_max, p_cell_idx, theta_cell_idx, uncerteainty_bloat):
    n_samples = 10000
    ps = np.random.uniform(p_lb, p_ub, [n_samples,1])
    thetas = np.random.uniform(theta_lb, theta_ub, [n_samples,1])
    # add four corners
    ps[0, 0] = p_lb
    ps[1, 0] = p_ub
    ps[-2, 0] = p_lb
    ps[-1, 0] = p_ub
    thetas[0, 0] = theta_lb
    thetas[1, 0] = theta_lb
    thetas[-2, 0] = theta_ub
    thetas[-1, 0] = theta_ub
    states = np.concatenate([ps, thetas], axis=1)
    outputs = []
    for idx, s in enumerate(states):
        s_np = np.array(s)/[6.36615, 17.247995] # normalize
        s_np = s_np.reshape(-1, 2)
        z = np.random.uniform(-.8, .8, size=(1, 2))
        # z = np.array([0.0, 0.0]).reshape(-1, 2)
        x = np.concatenate([z, s_np], axis=1)
        x = x.astype(np.float32)
        output = sess_all.run([sess_all.get_outputs()[0].name], {sess_all.get_inputs()[0].name: x})
        outputs.append(output)
    outputs = np.array(outputs).squeeze()
    print(f'outputs.shape: {outputs.shape}')


    # Add a bias term to X
    X_bias = np.c_[np.ones((states.shape[0], 1)), states]
    
    # Calculate coefficients
    coefficients = calculate_coefficients(X_bias, outputs)

    print(f'coefficients: {coefficients}')

    predictions = predict(X_bias, coefficients)

    # Get the coefficients (slope and intercept)
    slope = coefficients[1:]
    print(f'slope: {slope}')
    print(f'slope.shape: {slope.shape}')
    intercept = coefficients[0]
    print(f'intercept: {intercept}')
    print(f'intercept.shape: {intercept.shape}')

    diff = outputs - predictions
    # print(f'diff min: {diff.min()}, diff max: {diff.max()}')
    print(f'diff min: {diff.min()}, diff max: {diff.max()}')
    print(f'fit range : {diff.max() - diff.min()}')
    model_range = [diff.min(), diff.max()] 
    model_range = [diff.min() * (1 + uncerteainty_bloat), diff.max() * (1 + uncerteainty_bloat)] # add uncerteainty_bloat to the uncertainty range 
    print('#############################################')

    models_folder = 'models'
    Linear_folder = os.path.join(models_folder, 'Linear')
    if not os.path.exists(Linear_folder):
        os.makedirs(Linear_folder)
    # create an HDF5 file
    with h5py.File(f'models/Linear/singlecell_{p_cell_idx}_{theta_cell_idx}.h5', 'w') as f:
        # create a dataset in the file
        models_slopehd5 = f.create_dataset('modelslopes', data=slope)
        models_intercepthd5 = f.create_dataset('modelintercept', data=intercept)
        rangeshd5 = f.create_dataset('ranges', data=model_range)
        validphd5 = f.create_dataset('validp', data=[p_lb, p_ub])
        validthd5 = f.create_dataset('validtheta', data=[theta_lb, theta_ub])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--bloat', type=float, default=0.2)
    parser.add_argument('--pidx', type=int, default=1)
    parser.add_argument('--tidx', type=int, default=1)
    parser.add_argument('--ubloat', type=float, default=0.1)
    parser.add_argument('--freqmode', type=str, default="inf")  # options: "inf", "fixed"
    args = parser.parse_args()
    bloutcoef = args.bloat/2    # bloat the valid range by 2*'bloutcoef'
    cell_idx_p = args.pidx
    cell_idx_theta = args.tidx
    uncerteainty_bloat = args.ubloat
    freqmode = args.freqmode

    main(bloutcoef, cell_idx_p, cell_idx_theta, uncerteainty_bloat, freqmode)