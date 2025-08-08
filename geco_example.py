from geco import product_graph_generator, get_surface_cycles
import gurobipy as gp
from gurobipy import GRB
import time
from utils.sm_utils import *
import scipy.sparse as sp
from utils.misc import robust_lossfun
from utils.vis_util import plot_match, get_cams_and_rotation

time_limit = 60 * 60
## Load data (change dataset accordingly when using files from different datasets)
dataset = "faust" # dt4d_inter, dt4d_intra, smal
filename1 = "datasets/FAUST_r/off/tr_reg_000.off"
filename2 = "datasets/FAUST_r/off/tr_reg_001.off"
shape_opts = {"num_faces": 100}


## Load and downsample shapes and compute spidercurve on shape X
VX, FX, vx, fx, vx2VX, VY, FY, vy, fy, vy2VY = shape_loader(filename1, filename2, shape_opts)
ey, ex = get_spider_curve(vx, fx, vy, fy)

## Comptue Features and edge cost matrix
feature_opts = get_feature_opts(dataset)
feat_x, feat_y = get_features(VX, FX, VY, FY, feature_opts)
feature_difference = np.zeros((len(vx), len(vy)))
for i in range(0, len(vx)):
    diff = feat_y[vy2VY, :] - feat_x[vx2VX[i], :]
    feature_difference[i, :] = np.linalg.norm(diff, axis=1)

## ++++++++++++++++++++++++++++++++++++++++
## ++++++++ Solve with 🦎 GECO 🦎 +++++++++
## ++++++++++++++++++++++++++++++++++++++++
resolve_coupling = True
time_limit = 60 * 60
max_depth = 2
lp_relax = True


fx, _ = igl.orient_outward(vx, fx, np.ones_like(fx)[:, 0])
ey = igl.edges(fy)
ey = np.row_stack((ey, ey[:, [1, 0]]))
ex = get_surface_cycles(fx)
pg = product_graph_generator(vy, ey, vx, ex, feature_difference.T)
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

## output results
# f_matching = pg.convert_matching_to_surface_matching(result_vec.astype('bool'))
# fy_sol_orig = f_matching[:, [3, 4, 5]]
# fx_sol_orig = f_matching[:, [0, 1, 2]]
# energy = E.transpose() @ result_vec

# point map is [indices_in_x, corresponding_indices_in_y]
point_map = np.unique(product_space[result_vec.astype('bool'), :-1][:, [0, 2]], axis=0)
print(f"Optimisation took {end_time - start_time}s")

## Visualise result
[cam, cams, rotationX, rotationY] = get_cams_and_rotation(dataset)
plot_match(vy, fy, vx, fx, point_map[:, [1, 0]], cam, "", offsetX=[1, 0, 0],
                            rotationShapeX=rotationX, rotationShapeY=rotationY)
