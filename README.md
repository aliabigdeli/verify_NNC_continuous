# verify_NNC_continuous

Companion code for "Verification of Neural Network Control Systems in Continuous Time"

A. ArjomandBigdeli, A. Mata, and S. Bak


Installation and requirement
----------------------

```bash
conda env create --name nnccontinuous -f environment.yml
conda activate nnccontinuous
```

How to Run
----------------------

### Run using computed linear models weights and abstract graph
Unzip the linear models' weights and abstract transitions for a specific setting in 'models/archive' and place them in the 'models/Linear' and 'models/reachgraph' folders, respectively.
To create a abstract graph from abstract transitions run the following:

```bash
python merge_reach_graph.py
```
After obtaining the abstract graph, forward and backward reachability analysis can be performed.
For forward reachability, run the following:

```bash
python closed_loop.py
```
You can change the default initial set by `--plb`, `--pub`, `--tlb`, `--tub` options.

For backward reachability, run the following:

```bash
python safe_region.py
```

### Recomputing linear models weights and abstract graph
For computing the abstract transitions and linear models weights, run the following:

```bash
python runall_parallel.py
```
Then you should go with the procedure mentioned earlier.

