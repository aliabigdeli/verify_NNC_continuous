import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import argparse
import os


def get_p_theta_idx(p_lb_, p_ub_, theta_lb_, theta_ub_, p_lbs, p_ubs, theta_lbs, theta_ubs):
    if p_lb_ < p_lbs[0]:
        p_index_lb = -1
    else:
        p_index_lb = np.searchsorted(p_lbs, p_lb_, "right")-1
    if p_ub_ > p_ubs[-1]:
        p_index_ub = len(p_ubs)
    else:
        p_index_ub = np.searchsorted(p_ubs, p_ub_)

    if theta_lb_ < theta_lbs[0]:
        theta_index_lb = -1
    else:
        theta_index_lb = np.searchsorted(theta_lbs, theta_lb_, "right")-1
    if theta_ub_ > theta_ubs[-1]:
        theta_index_ub = len(theta_ubs)
    else:
        theta_index_ub = np.searchsorted(theta_ubs, theta_ub_)
    return p_index_lb, p_index_ub, theta_index_lb, theta_index_ub

def plot_states(states, p_lbs, p_ubs, theta_lbs, theta_ubs, axis=None, color='blue', color_fill='lightblue'):
    for state in states:
        p_idx, t_idx = state[0], state[1]
        pt1 = [p_lbs[p_idx], theta_lbs[t_idx]]
        pt2 = [p_ubs[p_idx], theta_lbs[t_idx]]
        pt3 = [p_ubs[p_idx], theta_ubs[t_idx]]
        pt4 = [p_lbs[p_idx], theta_ubs[t_idx]]
        vertices = np.array([pt1, pt2, pt3, pt4])
        # Close the rectangle by repeating the first vertex
        vertices = np.vstack([vertices, vertices[0]])

        # Extract x and y coordinates
        x, y = zip(*vertices)

        if axis:
            # Plot the rectangle
            axis.plot(x, y, color=color)
            # Fill the rectangle with color
            axis.fill(x, y, color=color_fill)
        else:
            # Plot the rectangle
            plt.plot(x, y, color=color)
            # Fill the rectangle with color
            plt.fill(x, y, color=color_fill)


def main(p_lb, p_ub, theta_lb, theta_ub, finalt, withoutinit):
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

    flag = 0
    if os.path.exists('reach_graph_idx.npy'):
        data = np.load('reach_graph_idx.npy')
        flag = 2
    elif os.path.exists('reachGraph.mat'):
        data = scipy.io.loadmat('reachGraph.mat')
        flag = 1
    else:
        print('No reach graph data found')
        return

    p_lb_, p_ub_, theta_lb_, theta_ub_ = get_p_theta_idx(p_lb, p_ub, theta_lb, theta_ub, p_lbs, p_ubs, theta_lbs, theta_ubs)

    current_states = set()
    for i in range(p_lb_, p_ub_):
        for j in range(theta_lb_, theta_ub_):
            current_states.add((i, j))

    print(f'len(initial_states): {len(current_states)}')

    fig_folder = 'Fig'
    closed_loop_folder = os.path.join(fig_folder, 'closedLoop')
    if not os.path.exists(closed_loop_folder):
        os.makedirs(closed_loop_folder)
    
    outfile = open(f'{closed_loop_folder}/cellcount.txt', 'w')
    outfile.write(f'len(initial_states): {len(current_states)}\n')
    
    # plot initial states
    plot_states(current_states, p_lbs, p_ubs, theta_lbs, theta_ubs)
    plt.xlim(-10, 10)
    plt.ylim(-30, 30)
    plt.title('Initial states')
    # plt.show()
    save_fig_path = 'Fig/closedLoop/initial_states.png'
    plt.savefig(save_fig_path)

    timestep = 1
    nsteps = int(finalt/timestep)

    if withoutinit:
        list_of_states = []
    else:
        list_of_states = [current_states.copy()]
    flag_converge = False
    converge_idx = 0
    for i in range(nsteps):
        new_states = set()
        n_current_states = len(current_states)
        while current_states:
            state = current_states.pop()
            p_idx, t_idx = state[0], state[1]
            
            if flag == 1:
                p_start = max(data['data'][p_idx,t_idx][0, 0][0][0] -1, 0)
                p_end = data['data'][p_idx,t_idx][0, 0][0][1] -1 if data['data'][p_idx,t_idx][0, 0][0][1] != -1 else 127
                t_start = max(data['data'][p_idx,t_idx][0, 0][1][0] -1, 0)
                t_end = data['data'][p_idx,t_idx][0, 0][1][1] -1 if data['data'][p_idx,t_idx][0, 0][1][1] != -1 else 127
            elif flag == 2:
                p_start = max(data[p_idx, t_idx, 0, 0] -1, 0)
                p_end = data[p_idx, t_idx, 0, 1] -1 if data[p_idx, t_idx, 0, 1] != -1 else 127
                t_start = max(data[p_idx, t_idx, 1, 0] -1, 0)
                t_end = data[p_idx, t_idx, 1, 1] -1 if data[p_idx, t_idx, 1, 1] != -1 else 127
            for p_i in range(p_start, p_end+1):
                for t_i in range(t_start, t_end+1):
                    new_states.add((p_i, t_i))
        current_states = new_states
        if i % int(5/timestep) == 0:
            list_of_states.append(new_states.copy())
        n_new_states = len(new_states)
        if n_current_states == n_new_states and (not flag_converge):
            flag_converge = True
            converge_idx = i
            print(f'converge after {i} steps, or {i*timestep} seconds')
            outfile.write(f'converge after {i} steps, or {i*timestep} seconds\n')

    # plot final states
    print(f'len(final_states): {len(current_states)}')
    outfile.write(f'len(final_states): {len(current_states)}\n')
    outfile.close()
    # print(f'current_states: {current_states}') 
    plt.clf()
    plot_states(current_states, p_lbs, p_ubs, theta_lbs, theta_ubs)
    plt.xlim(-10, 10)
    plt.ylim(-30, 30)
    plt.title(f'Final states after {finalt} seconds')
    # plt.show()
    save_fig_path = 'Fig/closedLoop/final_states.png'
    plt.savefig(save_fig_path)
    plt.clf()

    # plot states over time
    fig, axs = plt.subplots(len(list_of_states), 1, figsize=(8, 5*len(list_of_states)))

    for i, states in enumerate(list_of_states):
        axs[i].set_xlim(-10, 10)
        axs[i].set_ylim(-30, 30)
        axs[i].set_title(f'Reachable states after {(i)*5} seconds')
        plot_states(states, p_lbs, p_ubs, theta_lbs, theta_ubs, axs[i])

    plt.tight_layout()
    save_fig_path = 'Fig/closedLoop/states_overtime.png'
    plt.savefig(save_fig_path)
    # plt.show()

    # plot states over time
    if withoutinit:
        list_of_states2 = list_of_states[:3] + [list_of_states[-1]]
    else:
        list_of_states2 = list_of_states[:4] + [list_of_states[-1]]
    fig, axs = plt.subplots(1, len(list_of_states2), figsize=(4.25*len(list_of_states2), 4))

    fontsize = 14
    for i, states in enumerate(list_of_states2):
        axs[i].set_xlim(-10, 10)
        axs[i].set_xticks([-10, -5, 0, 5, 10])
        axs[i].set_xticklabels([-10, -5, 0, 5, 10], fontsize=fontsize)
        axs[i].set_ylim(-30, 30)
        axs[i].set_yticks([-20, 0, 20])
        axs[i].set_yticklabels([-20, 0, 20], fontsize=fontsize)

        if withoutinit:
            if i != 3:
                axs[i].set_title(f'Reachable states after {(i+1)*5} seconds', fontsize=fontsize)
            else:
                axs[i].set_title(f'Converged', fontsize=fontsize)
        else:
            if i != 4:
                axs[i].set_title(f'Reachable states after {(i)*5} seconds', fontsize=fontsize)
            else:
                axs[i].set_title(f'Converged', fontsize=fontsize)
        plot_states(states, p_lbs, p_ubs, theta_lbs, theta_ubs, axs[i])

    # plt.tight_layout()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    save_fig_path = 'Fig/closedLoop/states_overtime_horiz.png'
    plt.savefig(save_fig_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--plb', type=int, default=-9)
    parser.add_argument('--pub', type=int, default=9)
    parser.add_argument('--tlb', type=int, default=-10)
    parser.add_argument('--tub', type=int, default=10)
    parser.add_argument('--finalt', type=int, default=30)
    parser.add_argument('--withoutinit', action='store_true')
    args = parser.parse_args()
    p_lb = args.plb
    p_ub = args.pub
    theta_lb = args.tlb
    theta_ub = args.tub
    finalt = args.finalt
    without_init = args.withoutinit

    main(p_lb, p_ub, theta_lb, theta_ub, finalt, without_init)