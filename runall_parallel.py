import sim_cell
import time
import matlab.engine
import matlab
import argparse
import time
import threading


def run_task(engine_id, start_idx, end_idx, theta_start_idx=0, theta_end_idx=128, initial_bloat=0.2, ubloat=0.1, freqmode="inf"):
    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        print(f"%%%%%% Engine {engine_id} started.")
        
        for pidx in range(start_idx, end_idx):
            for tidx in range(theta_start_idx, theta_end_idx):
                start_time = time.time()  # Start measuring time

                sim_cell.main(bloutcoef=initial_bloat, cell_idx_p=pidx, cell_idx_theta=tidx, uncerteainty_bloat=ubloat, freqmode=freqmode)
                result = eng.localLinContCellDegParallel(pidx, tidx, freqmode)
                bloat = initial_bloat
                while not result:
                    bloat *= 1.5
                    print(f"bloat factor for ({pidx}, {tidx}) is currently: {bloat}")
                    sim_cell.main(bloutcoef=bloat, cell_idx_p=pidx, cell_idx_theta=tidx, uncerteainty_bloat=ubloat, freqmode=freqmode)
                    result = eng.localLinContCellDegParallel(pidx, tidx, freqmode)
                print(f"###### Loop finished. Final value of bloat factor for ({pidx}, {tidx}) is {bloat}")

                end_time = time.time()  # Stop measuring time
                time_taken = end_time - start_time
                print(f"###### Time taken for iteration ({pidx}, {tidx}): {time_taken} seconds")
        
    except Exception as e:
        print(f"%%%%%% Error on engine {engine_id}: {e}")
    finally:
        # Stop MATLAB engine
        eng.quit()
        print(f"%%%%%% Engine {engine_id} stopped.")

def main(start_end_list, initial_bloat=0.2, ubloat=0.1, freqmode="inf"):
    # List to hold threads
    threads = []

    thread_idx = 0
    for start_idx, end_idx in start_end_list:
        # Create and start a thread for each MATLAB script
            
        thread = threading.Thread(target=run_task, args=(thread_idx, start_idx, end_idx, 0, 128, initial_bloat, ubloat, freqmode))
        threads.append(thread)
        thread.start()
        thread_idx += 1
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pidx', type=int, default=None)
    parser.add_argument('--tidx', type=int, default=None)
    parser.add_argument('--pstart', type=int, default=None)
    parser.add_argument('--pend', type=int, default=None)
    parser.add_argument('--tstart', type=int, default=None)

    parser.add_argument('--bloat', type=float, default=0.2) # initial bloating factor
    parser.add_argument('--ubloat', type=float, default=0.1)    # uncertainty bloating factor
    parser.add_argument('--freqmode', type=str, default="inf")  # options: "inf", "fixed"
    args = parser.parse_args()
    if (args.pidx is not None) and (args.tidx is not None): # Single cell
        p_idx = args.pidx
        theta_idx = args.tidx
        run_task(0, p_idx, p_idx+1, theta_start_idx=theta_idx, theta_end_idx=theta_idx+1, initial_bloat=args.bloat, ubloat=args.ubloat, freqmode=args.freqmode)
    elif (args.pidx is not None) and (args.tstart is not None): # Range of cells (only theta variable)
        p_idx = args.pidx
        theta_start = args.tstart
        run_task(0, p_idx, p_idx+1, theta_start_idx=theta_start, theta_end_idx=128, initial_bloat=args.bloat, ubloat=args.ubloat, freqmode=args.freqmode)
    elif (args.pstart is not None) and args.pend: # Range of cells
        start = args.pstart
        end = args.pend
        assert start < end
        if end - start > 5:
            start_end_list = []
            step_size = 4
            for i in range(0, end-start, step_size):
                if start+i+step_size > end:
                    start_end_list.append((start+i, end))
                else:
                    start_end_list.append((start+i, start+i+step_size))
            main(start_end_list, initial_bloat=args.bloat, ubloat=args.ubloat, freqmode=args.freqmode)
        else:
            start_end_list = [(start, end)]
            main(start_end_list, initial_bloat=args.bloat, ubloat=args.ubloat, freqmode=args.freqmode)
    else:
        start_end_list = [(0, 15), (15, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 95), (95, 100), (100, 105), (105, 115), (115, 128)]
        main(start_end_list, initial_bloat=args.bloat, ubloat=args.ubloat, freqmode=args.freqmode)
