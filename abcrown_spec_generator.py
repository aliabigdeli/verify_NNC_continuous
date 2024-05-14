import onnx
from onnx import numpy_helper
import numpy as np
import torch.nn as nn
import torch
import struct
import warnings
import h5py
import os
import argparse


data_type_tab = {
    1: ['f', 4],
    2: ['B', 1],
    3: ['b', 1],
    4: ['H', 2],
    5: ['h', 2],
    6: ['i', 4],
    7: ['q', 8],
    10: ['e', 2],
    11: ['d', 8],
    12: ['I', 4],
    13: ['Q', 8]
}

def unpack_weights(initializer):
    ret = {}
    for i in initializer:
        name = i.name
        dtype = i.data_type
        shape = list(i.dims)
        if dtype not in data_type_tab:
            warnings("This data type {} is not supported yet.".format(dtype))
        fmt, size = data_type_tab[dtype]
        if len(i.raw_data) == 0:
            if dtype == 1:
                data_list = i.float_data
            elif dtype == 7:
                data_list = i.int64_data
            else:
                warnings.warn("No-raw-data type {} not supported yet.".format(dtype))
        else:
            data_list = struct.unpack('<' + fmt * (len(i.raw_data) // size), i.raw_data)
        t = torch.tensor(data_list)
        if len(shape) != 0:
            t = t.view(*shape)
        ret[name] = t
    return ret

class ONNXNet(nn.Module):
    def __init__(self):
        super(ONNXNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),

            nn.Linear(8, 2),
        )
        self.controller = nn.Linear(2, 1, bias=False)
        self.linear_weight = nn.Linear(2, 1, bias=False)
        self.linear_constraint = nn.Linear(2, 2, bias=False)
    
    def forward(self, x):
        output1 = self.controller(self.main(x))
        output2 = self.linear_weight(x[:, 2:])
        concatenated = torch.cat((output1, output2), dim=1)
        output3 = self.linear_constraint(concatenated)
        return output3

def save_vnnlib(input_bounds: np.ndarray, output_bounds: np.ndarray, spec_path: str):

    """
    Saves the property derived as vnn_lib format.
    """

    with open(spec_path, "w") as f:

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(output_bounds.shape[0]):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write(f"(assert (or")
        f.write(f"\n")
        for i in range(output_bounds.shape[0]):
            f.write(f"    (and (>= Y_{i} {output_bounds[i]}))\n")
        f.write(f"))")
        f.write("\n")

def create_yml_file(folder_path, cell_idx_p, cell_idx_theta):
    # create yml file for alpha-beta-crown verifier
    yml_path = folder_path + f'/reach_config_{cell_idx_p}_{cell_idx_theta}.yml'
    content = f"""# Example of verifying an ONNX model with VNNLIB general specifications
model:
    # Assuming you have cloned the vnncomp2021 repository: https://github.com/stanleybak/vnncomp2021
    onnx_path: complete_verifier/models/reach/AllInOne_{cell_idx_p}_{cell_idx_theta}.onnx
specification:
    # VNNLIB file specification.
    vnnlib_path: complete_verifier/models/reach/prop_{cell_idx_p}_{cell_idx_theta}.vnnlib
# solver:
    # batch_size: 2048  # Number of subdomains to compute in parallel in bound solver. Decrease if you run out of memory.
bab:
    # timeout: 360  # Timeout (in second) for verifying one image/property.
    override_timeout: 1000
    """
    with open(yml_path, "w") as f:
        f.write(content)

def check_cell_spec_based_on_inputs(cell_idx_p, cell_idx_theta):
    p_range = [-10, 10]
    p_num_bin = 128
    theta_range = [-30, 30]
    theta_num_bin = 128
    p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)


    p_lb, p_ub = p_lbs[cell_idx_p], p_lbs[cell_idx_p+1]
    theta_lb, theta_ub = theta_lbs[cell_idx_theta], theta_lbs[cell_idx_theta+1]

    print(f"p_lb, p_ub: {p_lb, p_ub}")
    print(f"theta_lb, theta_ub: {theta_lb, theta_ub}")

    # Open the .h5 file
    file_path = f'models/Linear/singlecell_{cell_idx_p}_{cell_idx_theta}.h5'

    # Reading the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Read each dataset
        models_slope = file['modelslopes'][:]
        models_intercept = file['modelintercept'][()]
        ranges = file['ranges'][:]
        validp = file['validp'][:]
        validtheta = file['validtheta'][:]

    print(f'models_slope: {models_slope}')
    print(f'models_intercept: {models_intercept}')
    print(f'ranges: {ranges}')
    print(f'validp: {validp}')
    print(f'validtheta: {validtheta}')


    # add skip connection to the NN model to pass the input to the output and then add linear layer to create linear constraints for the specification
    normalization_factor = np.array([6.36615, 17.247995])
    model_path = 'models/papermodel/AllInOne.onnx'
    net = ONNXNet()
    onnx_model = onnx.load(model_path)
    weights = unpack_weights(onnx_model.graph.initializer)
    weights_keys = list(weights.keys())
    for o_weight, p_weight in zip(weights_keys[:-1], net.main.parameters()):
        p_weight.data = weights[o_weight]
    net.controller.weight.data = torch.FloatTensor([[-0.74, -0.44]])
    net.linear_weight.weight.data = torch.from_numpy((models_slope * normalization_factor).reshape(1, -1).astype(np.float32))
    net.linear_constraint.weight.data = torch.FloatTensor([[1.0, -1.0], [-1.0, 1.0]])

    folder_path = 'spec'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(net.state_dict(), folder_path + f"/AllInOne_{cell_idx_p}_{cell_idx_theta}.pth")

    x = torch.randn(1, 4)

    torch.onnx.export(net, 
                        x,
                        folder_path + f"/AllInOne_{cell_idx_p}_{cell_idx_theta}.onnx",
                        input_names = ['input'],   # the model's input names
                        output_names = ['output']) # the model's output names


    

    input_bounds = np.array([[-0.8, 0.8], [-0.8, 0.8], [p_lb, p_ub]/normalization_factor[0], [theta_lb, theta_ub]/normalization_factor[1]], dtype=np.float32)
    # Diff = 'AllInOne'model's output – (A*states + b) -> Diff_max = ub, Diff_min = lb -> ub >= Y'_0 - (Y'_1 + b) , lb <= Y'_0 - (Y'_1 + b)
    # output constraint (safe condition, considering linear model: A*states + b + [lb, ub], and A*states = Y'_1, Y'= output of a layer before the last layer): Y'_1 + b + lb <= Y'_0 <= Y'_1 + b + ub
    # output constraint (unsafe condition): Y'_0 - ub >= Y'_1 + b or Y'_1 + b >= Y'_0 - lb -> Y'_0 - Y'_1 >= b + ub or -Y'_0 + Y'_1 >= -lb - b
    # continue: (unsafe condition, considering Y'_0 - Y'_1 = Y_0 or -Y'_0 + Y'_1 = Y_1):  Y_0 >= b + ub or Y_1 >= -lb - b
    output_bounds = np.array([ models_intercept + ranges[1], - ranges[0] - models_intercept], dtype=np.float32)

    spec_path = folder_path + f'/prop_{cell_idx_p}_{cell_idx_theta}.vnnlib'
    save_vnnlib(input_bounds, output_bounds, spec_path)
    # create yml file for alpha-beta-crown verifier
    create_yml_file(folder_path, cell_idx_p, cell_idx_theta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pidx', type=int, default=1)
    parser.add_argument('--tidx', type=int, default=1)
    args = parser.parse_args()
    cell_idx_p = args.pidx
    cell_idx_theta = args.tidx

    check_cell_spec_based_on_inputs(cell_idx_p, cell_idx_theta)