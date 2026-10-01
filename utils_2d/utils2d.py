import os
import time
from matplotlib import gridspec, pyplot as plt
import numpy as np
import pytz
import torch

def create_tests_folder(parent_folder = "",prefix = "", postfix = ""):
    time_stamp = int(time.time())
    time_zone = pytz.timezone("Asia/Singapore")
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    tests_folder = os.getcwd() + f"/{parent_folder}/test{prefix}_{test_time}{postfix}"
    os.makedirs(tests_folder)
    print(f"\nWorking in folder {tests_folder}\n")
    return tests_folder

def get_PDE_dataset(domain, N_train_x, N_train_y, N_train_t):
    x_min, x_max = domain[0], domain[1]
    y_min, y_max = domain[2], domain[3]
    t_min, t_max = domain[4], domain[5]

    x = torch.linspace(x_min, x_max, N_train_x).view(-1,1)
    y = torch.linspace(y_min, y_max, N_train_y).view(-1,1)
    t = torch.linspace(t_min, t_max, N_train_t).view(-1,1)
    X, Y, T = torch.meshgrid(x.squeeze(1), y.squeeze(1), t.squeeze(1))
    X_pde = torch.hstack((X.transpose(1,0).flatten().view(-1,1), Y.transpose(1,0).flatten().view(-1,1), T.transpose(1,0).flatten().view(-1,1)))
    return X_pde


def get_BC_dataset(domain, N_bc, case):
    x_min, x_max = domain[0], domain[1]
    y_min, y_max = domain[2], domain[3]
    t_min, t_max = domain[4], domain[5]

    x_bc = torch.linspace(x_min, x_max, N_bc // 6).view(-1,1)
    y_bc = torch.linspace(y_min, y_max, N_bc // 6).view(-1,1)
    t_bc = torch.linspace(t_min, t_max, N_bc // 6).view(-1,1)

    X, Y, T = torch.meshgrid(x_bc.squeeze(1), y_bc.squeeze(1), t_bc.squeeze(1), indexing = 'ij')
    
    # Initial Condition. Left Edge. t = 0 
    init_cond = torch.hstack((
        X[:, :, 0].permute(1,0).flatten()[:, None],
        Y[:, :, 0].permute(1,0).flatten()[:, None],
        T[:, :, 0].permute(1,0).flatten()[:, None]
    ))

    # Boundary Conds
    # Left x = x_min, y_min =< y =< y_max, t_min =< t =< t_max
    left_x = torch.hstack((
        X[0, :, :].permute(1,0).flatten()[:, None],
        Y[0, :, :].permute(1,0).flatten()[:, None],
        T[0, :, :].permute(1,0).flatten()[:, None]
    ))

    # Right x = x_max, y_min =< y =< y_max, t_min =< t =< t_max
    right_x = torch.hstack((
        X[-1, :, :].permute(1,0).flatten()[:, None],
        Y[-1, :, :].permute(1,0).flatten()[:, None],
        T[-1, :, :].permute(1,0).flatten()[:, None]
    ))

    # Down x_min =< x =< x_max, y = y_min, t_min =< t =< t_max
    down_x = torch.hstack((
        X[:, 0, :].permute(1,0).flatten()[:, None],
        Y[:, 0, :].permute(1,0).flatten()[:, None],
        T[:, 0, :].permute(1,0).flatten()[:, None]
    ))

    # Top x_min =< x =< x_max, y = y_max, t_min =< t =< t_max 
    top_x = torch.hstack((
        X[:, -1, :].permute(1,0).flatten()[:, None],
        Y[:, -1, :].permute(1,0).flatten()[:, None],
        T[:, -1, :].permute(1,0).flatten()[:, None]
    ))

    # boundary conds
    # u(x_min,y,t) = 0
    U_xmin = torch.zeros(left_x.shape[0], 1)

    # u(x_max, y, t) = 0
    U_xmax = torch.zeros(right_x.shape[0], 1)

    # u(x, y_min, t) = 0
    U_ymin = torch.zeros(down_x.shape[0], 1)

    # u(x, y_max, t) = 0
    U_ymax = torch.zeros(top_x.shape[0], 1)

    if case == "example_1":
        # init_u: u(x,y,0) = sin(pi*x) * sin(pi*y) 
        f_init = ((torch.sin(np.pi*init_cond[:, 0]))* (torch.sin(np.pi*init_cond[:, 1]))).unsqueeze(1)
    elif case == 'example_2':
        # init_u: u(x,y,t) = (x**2)*(y**2)*((x-1)**2)*((y-1)**2)/4
        f_init = ((1/4)*(init_cond[:, 0]**2)*(init_cond[:, 1]**2)*((init_cond[:, 0]-1)**2)*((init_cond[:, 1] - 1)**2)).unsqueeze(1)

    X_train = torch.vstack([u_init, left_x, right_x, down_x, top_x])
    U_train = torch.vstack([f_init, U_xmin, U_xmax, U_ymin, U_ymax])
    return X_train, U_train




