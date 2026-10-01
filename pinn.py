import logging
import time
import os
import sys
import numpy as np
import torch

from utils.plots import plot_mesh, plot_loss
from utils.fcn_module import FCN
from utils.gen_plot import generate_gif
from utils.gen_data import data_gen
from utils.utils import create_tests_folder
from pydoe import lhs # lhs (latin hypercube sampling)

# Device Configuration

torch.set_default_dtype(torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

# Tuning Parameter

steps = 5000
batch_size = 1000
w_bc = 100
w_pde = 1
lr = 1e-4
layers = np.array([2,50,50,50,50,1])
log_freq = 50

case = 'example_2'
if case == 'example_1':
    domain = [-1,1,0,1]
elif case == 'example_2':
    domain = [0,1,0,10]

N_test_x = 300
N_test_t = 300
N_train_x = 150
N_train_t = 150
N_bc = 1000

# Create test folder with the result

test_folder = create_tests_folder(parent_folder = 'results', prefix = f"_{case}")
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
logging.basicConfig(handlers= [logging.StreamHandler(sys.stdout), logging.FileHandler(test_folder + "/0_log.txt")], level = logging.INFO, format="%(message)s")
logger = logging.getLogger()

# Generate Data

DG = data_gen(domain, case)
x_train_pde = DG.get_PDE_dataset(N_train_x, N_train_t)
x_train_bc, y_train_bc = DG.get_BC_dataset(N_bc) # generate train dataset
y_real, x_test, y_test, X, T = DG.get_test_dataset(N_test_x, N_test_t)

#####
# Is it possible to use Latin Hypercube Sampling instead of the get_PDE_dataset function
# lb = x_test[0]  # first value
# ub = x_test[-1]  # last value
# x_train_pde = lb + (ub - lb) * lhs(2, N_train_x*N_test_t)  # Choose 20000 points, 2 as the inputs are x and t
#####

# If the device is CUDA (CUDA is installed), the data are sent to the GPU
x_train_bc = x_train_bc.float().to(device) # training points (bc)
y_train_bc = y_train_bc.float().to(device) # training points (boundary condition points)
x_train_pde = x_train_pde.float().to(device) # collocation points
X_test = x_test.float().to(device) # input the dataset 
Y_test = y_test.float().to(device) # input the real solution

# Crate model and optimizer
PINN = FCN(layers, case, domain)
PINN.to(device)
print(PINN)
optimizer = torch.optim.Adam(PINN.parameters(), lr = lr, amsgrad = False)

res_dict = {'loss':[], 'loss_bc':[], 'rela_err_l2':[], 'iteration':[]}
start_time_a = time.time()
count = 1
plot_mesh(X, T, y_real, case, name=f"0_real", error = 0, folder = test_folder)
for i in range(1, steps + 1):
    idx = np.random.choice(x_train_pde.shape[0], batch_size, replace = False) # select randomly a batch in x_train_pde
    loss = PINN.loss(x_train_bc, y_train_bc, x_train_pde[idx, :], w_bc = w_bc, w_pde = w_pde) # self, x_BC, y_BC, x_PDE, w_bc, w_pde.  use mean square error
    optimizer.zero_grad() # 'CLEAN' the Optimizer
    loss.backward() # compute the gradient with 'backpropagation'
    optimizer.step() # actualization the parameter

    # Create the plots with the solution with the current parameters, and the relative error.
    res_dict['loss'].append(loss.item())
    res_dict['iteration'].append(i)
    with torch.no_grad():
        err_l2_mean, error = PINN.error_l2_norm(X_test, Y_test)
        res_dict['rela_err_l2'].append(err_l2_mean.item())
    if i % log_freq == 0 or i == 1:
        u_predict = PINN(X_test)
        arr_y1 = u_predict.reshape(shape = [N_test_t, N_test_x]).transpose(1,0).detach().cpu()
        plot_mesh(X,T, arr_y1, case, name = f"approx_{count}", loss = loss.item(), folder = test_folder, iter=i)
        # error = error.reshape(shape=[N_test_t, N_test_x]).transpose(1, 0).detach()
        # plot_mesh(X, T, error, name=f"error_{count}", error=err_l2_mean, folder=test_folder, iter=i)
        if i % (log_freq*2) == 0 or i == 1:
            logger.info(f"| Iter: {i} | Loss: {loss.item():.4f} | Total_time: {(time.time() - start_time_a)/60:.1f} minutes")
        count += 1
    # path = test_folder + "/A_result_dict.npy"
    # np.save(path, np.asarray(res_dict, dtype = object))
print("Training Finished")
print('length of loss data: ', len(res_dict['loss']))
print('length of relative error L2: ', len(res_dict['rela_err_l2']))
print('length of iteration: ', len(res_dict['iteration']))
print('length of loss boundary conditions: ', len(res_dict['loss_bc']))
# Animation
generate_gif(test_folder, count, remove_imgs=True)
plot_loss(res_dict['iteration'], res_dict['loss'], 'Total Loss', 'loss_vs_iteration', case = case)
