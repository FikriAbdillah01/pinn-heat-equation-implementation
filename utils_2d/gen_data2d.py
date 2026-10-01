import torch
from utils_2d.fcn2d import u_real
import numpy as np

class data_gen:
    def __init__(self, domain, case):
        super().__init__()
        self.case = case
        self.x_min = domain[0]
        self.x_max = domain[1]
        self.y_min = domain[2]
        self.y_max = domain[3]
        self.t_min = domain[4]
        self.t_max = domain[5]
    
    def get_test_dataset(self, N_test_x, N_test_y, N_test_t):
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        t_min, t_max = self.t_min, self.t_max

        x = torch.linspace(x_min, x_max, N_test_x).view(-1,1)
        y = torch.linspace(y_min, y_max, N_test_y).view(-1,1)
        t = torch.linspace(t_min, t_max, N_test_t).view(-1,1)

        X, Y, T = torch.meshgrid(x.squeeze(1), y.squeeze(1), t.squeeze(1), indexing = 'ij')
        y_real = u_real(X, Y, T, self.case)

        x_test = torch.hstack((
            X.permute(2,1,0).flatten().view(-1,1),
            Y.permute(2,1,0).flatten().view(-1,1),
            T.permute(2,1,0).flatten().view(-1,1)
        ))

        y_test = y_real.permute(2,1,0).flatten().view(-1,1)

        return y_real, x_test, y_test, X, Y, T

    def get_PDE_dataset(self, N_train_x, N_train_y, N_train_t):
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        t_min, t_max = self.t_min, self.t_max

        x = torch.linspace(x_min, x_max, N_train_x + 2).view(-1,1)[1:-1]
        y = torch.linspace(y_min, y_max, N_train_y +2).view(-1,1)[1:-1]
        t = torch.linspace(t_min, t_max, N_train_t + 2).view(-1,1)[1:-1]
        X, Y, T = torch.meshgrid(x.squeeze(1), y.squeeze(1), t.squeeze(1), indexing = 'ij')
        X_pde = torch.hstack((X.transpose(1,0).flatten().view(-1,1), Y.transpose(1,0).flatten().view(-1,1), T.transpose(1,0).flatten().view(-1,1)))
        return X_pde
    
    def get_BC_dataset(self, N_bc):
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        t_min, t_max = self.t_min, self.t_max
        case = self.case

        x_bc = torch.linspace(x_min, x_max, N_bc).view(-1,1)
        y_bc = torch.linspace(y_min, y_max, N_bc).view(-1,1)
        t_bc = torch.linspace(t_min, t_max, N_bc).view(-1,1)
        X, Y, T = torch.meshgrid(x_bc.squeeze(1), y_bc.squeeze(1), t_bc.squeeze(1), indexing = 'ij')

        # initial conds
        init_cond = torch.hstack((
            X[:, :, 0].permute(1,0).flatten()[:, None],
            Y[:, :, 0].permute(1,0).flatten()[:, None],
            T[:, :, 0].permute(1,0).flatten()[:, None]
            ))
        
        if case == "example_1":
            # init_u: u(x,y,0) = sin(pi*x) * sin(pi*y) 
            f_init = ((torch.sin(np.pi*init_cond[:, 0]))* (torch.sin(np.pi*init_cond[:, 1]))).unsqueeze(1)
        elif case == 'example_2':
            # init_u: u(x,y,t) = (x - 1)**2 * (y-1)**2/4
            f_init = ((((init_cond[:,0] - 1)**2*(init_cond[:, 1] - 1)))**2*(init_cond[:, 0]*init_cond[:, 1]*1/8)).unsqueeze(1)
        
        # Boundary Conds Dataset
        ## Initial Condition. Left Edge. t = 0 
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

        X_train = torch.vstack([init_cond, left_x, right_x, down_x, top_x])
        U_train = torch.vstack([f_init, U_xmin, U_xmax, U_ymin, U_ymax])

        idx = np.random.choice(X_train.shape[0], N_bc, replace = False)
        X_train_BC = X_train[idx,:]
        U_train_BC = U_train[idx,:]
        return X_train_BC, U_train_BC