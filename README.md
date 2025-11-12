# 🦎 GECO 🦎

Official repository for the 2025 ICCV paper "Fast Globally Optimal and Geometrically Consistent 3D Shape Matching" by Paul Roetzer and Florian Bernard. 
Visit our [webpage](https://paulroetzer.github.io/publications/2025-06-26-geco.html) for more details.


## ⚙️ Installation

Note: this respository contains the code to generate the Geco optimisation problem (for computing features etc. please also follow installation instructions in [Spider-Match](https://github.com/paul0noah/spider-match) repository).

1. Download [Spider-Match](https://github.com/paul0noah/spider-match) repository via `git clone https://github.com/paul0noah/spider-match.git`
2. Navigate into folder of downloaded repository `cd spider-match-main`
3. Follow installation instructions of [Spider-Match](https://github.com/paul0noah/spider-match) repository.
4. With the conda environment `spidermatch` activated, run `pip install git+https://github.com/paul0noah/GeCo`
5. Download `geco_example.py` via `curl -OL https://raw.githubusercontent.com/paul0noah/GeCo/refs/heads/main/geco_example.py`
6. Done. You can run the example now via `python geco_example.py`

## ✨ Usage

Either directly run `geco_example.py` (after following installation instructions above) or run the below python 🐍 code (at least after installing geco and gurobi).
Note: all matrices are assumed to be numpy type.

```python
import numpy as np
import igl
import gurobipy as gp
from gurobipy import GRB
import time
import scipy.sparse as sp
from utils.misc import robust_lossfun
from geco import product_graph_generator, get_surface_cycles

## TODO: load your own shapes (numpy arrays) with per-vertex features
vx, fx, feat_x = ..., ...
vy, fy, feat_y = ..., ...

## settings
resolve_coupling = True
time_limit = 60 * 60
max_depth = 2
lp_relax = True

## extract data
feature_difference = np.zeros((len(vx), len(vy)))
for i in range(0, len(vx)):
    diff = feat_y[vy2VY, :] - feat_x[vx2VX[i], :]
    feature_difference[i, :] = np.sum(to_numpy(robust_lossfun(torch.from_numpy(diff.astype('float64')),
                                                      alpha=torch.tensor(2, dtype=torch.float64),
                                                      scale=torch.tensor(0.3, dtype=torch.float64))), axis=1)
fx, _ = igl.orient_outward(vx, fx, np.ones_like(fx)[:, 0])
ey = igl.edges(fy)
ey = np.row_stack((ey, ey[:, [1, 0]]))
ex = get_surface_cycles(fx)
pg = product_graph_generator(vy, ey, vx, ex, feature_difference)
pg.set_resolve_coupling(resolve_coupling)
pg.set_max_depth(max_depth)
pg.generate()
product_space = pg.get_product_space()
E = pg.get_cost_vector()
I, J, V = pg.get_constraint_matrix_vectors()
RHS = pg.get_rhs().flatten()



## setup gurobi problem
m = gp.Model("GECO")
if lp_relax:
    x = m.addMVar(shape=E.shape[0], vtype=GRB.CONTINUOUS, name="x")
    m.addConstr(x >= 0, name="cx0")
    m.addConstr(x <= 1, name="cx1")
else:
    x = m.addMVar(shape=E.shape[0], vtype=GRB.BINARY, name="x")
    

obj = E.transpose() @ x
m.setObjective(obj, GRB.MINIMIZE)

rows = RHS.shape[0]
A = sp.csr_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(rows, E.shape[0]))
m.addConstr(A @ x == RHS, name="c")

## optimise
start_time = time.time()
m.setParam('TimeLimit', time_limit)
m.setParam('Crossover', 4)
m.optimize()
end_time = time.time()
result_vec = x.X

## retrieve results
# point map is [indices_in_x, corresponding_indices_in_y]
point_map = np.unique(product_space[result_vec.astype('bool'), :-1][:, [0, 2]], axis=0)
feature_difference = feature_difference.T
cost = feature_difference[point_map[:, 1], point_map[:, 0]]
point_map = point_map[np.lexsort((point_map[:, 0], cost))]
_, idx = np.unique(point_map[:, 0], return_index=True)
point_map = point_map[idx, :]
```

# 🎓 Attribution
When using this code in your own projects please cite the following

```bibtex
@inproceedings{roetzer2025geco,
    author    = {Paul Roetzer and Florian Bernard},
    title     = {Fast Globally Optimal and Geometrically Consistent 3D Shape Matching},
    booktitle = {ICCV},
    year      = 2025
}
```
