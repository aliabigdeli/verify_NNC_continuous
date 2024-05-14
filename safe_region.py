import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import argparse

class NextState:
    def __init__(self, use_mat=True):
        self.use_mat = use_mat
        self.data = scipy.io.loadmat('reachGraph.mat') if use_mat else np.load('reach_graph_idx.npy')

    def get_next_state(self, p_idx, t_idx):
        if self.use_mat:
            return self.data['data'][p_idx,t_idx][0, 0]
        else:
            return self.data[p_idx, t_idx]

def main(use_mat):
    ns = NextState(use_mat)

    previous_states = [[ set() for _ in range(128)] for _ in range(128)]
    for i in range(128):
        for j in range(128):
            next_idx = ns.get_next_state(i, j)
            
            start_p = max(next_idx[0][0] -1, 0)
            end_p = next_idx[0][1] -1 if next_idx[0][1] != -1 else 127
            start_t = max(next_idx[1][0] -1, 0)
            end_t = next_idx[1][1] -1 if next_idx[1][1] != -1 else 127
            for p_idx in range(start_p, end_p+1):
                for t_idx in range(start_t, end_t+1):
                    previous_states[p_idx][t_idx].add((i, j))

    isSafe = np.ones([128,128], dtype=int)
    new_unsafe_state = []

    # initial check: set the state whose next state is not safe to 0
    for i in range(128):
        for j in range(128):
            over_range_index = ns.get_next_state(i, j)
            if over_range_index[0][0] == -1 or over_range_index[1][0] == -1 or over_range_index[0][1] == -1 or over_range_index[1][1] == -1:
                isSafe[(i, j)] = 0
                new_unsafe_state.append((i, j))

    while new_unsafe_state:
        print(len(new_unsafe_state))
        temp = []
        for (i, j) in new_unsafe_state:
            for (_i, _j) in previous_states[i][j]:
                if isSafe[(_i, _j)] != 0:
                    temp.append((_i, _j))
                    isSafe[(_i, _j)] = 0
        new_unsafe_state = temp


    fig, ax = plt.subplots()
    graph = isSafe.transpose()
    paddings = np.zeros([128, 10])
    graph = np.concatenate([paddings, graph, paddings], axis=1)
    print(graph.shape)
    ax.imshow(graph, cmap="gray")
    ax.set_xticks([0+10, 127/4+10, 127*2/4+10, 127*3/4+10, 127+10])
    ax.set_xticklabels([-10, -5, 0, 5, 10])
    ax.set_xlim([0, 147])

    ax.set_yticks([127/60*10, 127/60*30, 127/60*50])
    ax.set_yticklabels([-20, 0, 20])
    ax.set_ylim([0, 127])
    ax.set_xlabel(r"$p$ (m)")
    ax.set_ylabel(r"$\theta$ (degrees)")

    

    # Check if "closedLoop" folder exists in "Fig" folder
    if not os.path.exists("Fig/closedLoop"):
        # Create "closedLoop" folder
        os.makedirs("Fig/closedLoop")
    plt.savefig("Fig/closedLoop/safe_region.png")
    plt.close()

    # count nonzero elements of isSafe
    with open("Fig/closedLoop/safe_region.txt", "w") as f:
        f.write(f'safe cells: {np.count_nonzero(isSafe)}\n')
        f.write(f'unsafe cells: {128*128-np.count_nonzero(isSafe)}\n')
        f.write(f'percentage of safe cells: {np.count_nonzero(isSafe)/128/128*100}%\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', action='store_true')
    args = parser.parse_args()
    use_mat = args.mat

    main(use_mat)
