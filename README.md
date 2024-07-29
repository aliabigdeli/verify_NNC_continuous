# Verification of Neural Network Control Systems in Continuous Time

Companion code for ["Verification of Neural Network Control Systems in Continuous Time"](https://arxiv.org/abs/2406.00157)


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


## Citing ##
The following citation can be used:

```
@inbook{ArjomandBigdeli_2024,
   title={Verification of Neural Network Control Systems in Continuous Time},
   ISBN={9783031651120},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-031-65112-0_5},
   DOI={10.1007/978-3-031-65112-0_5},
   booktitle={Lecture Notes in Computer Science},
   publisher={Springer Nature Switzerland},
   author={ArjomandBigdeli, Ali and Mata, Andrew and Bak, Stanley},
   year={2024},
   pages={100–115} }
```

