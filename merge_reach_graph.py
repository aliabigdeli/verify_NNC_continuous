import glob
import scipy.io
import numpy as np

file_paths = glob.glob("models/reachgraph/*.mat")

reach_graph_idx = np.zeros((128, 128, 2, 2), dtype=int)
reach_graph_range = np.zeros((128, 128, 2, 2))
for file_path in file_paths:
    data = scipy.io.loadmat(file_path)
    # Process the data from each .mat file here
    file_name = file_path.split("/")[-1]
    file_name = file_name[:-4]  # Remove ".mat" from the end of the file_name
    index1, index2 = file_name.split("_")[-2:]
    # Use index1 and index2 as needed
    reach_graph_idx[int(index1), int(index2)] = data['reachable_range_idx']

np.save('reach_graph_idx.npy', reach_graph_idx)