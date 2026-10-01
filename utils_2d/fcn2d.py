import torch
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCN2D(nn.Module):
    def __init__(self, layers, case, domain):
        super().__init__()

        # Activation Function
        self.layers = layers
        self.domain = domain
        self.case = case
        self.activation = nn.Tanh()

        # Loss Function
        self.loss_function = nn.MSELoss(reduction = 'mean')

        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range (len(self.layers) - 1)])
        self.iter = 0

        # Xavier Normal Initialization
        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain = 1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward (self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def lossbc(self, x_BC, u_BC):
        loss_BC = self.loss_function(self.forward(x_BC), u_BC)
        return loss_BC

    def lossPDE(self, x_PDE):
        x_min, x_max, t_min, t_max = self.domain[0], self.domain[1], self.domain[4], self.domain[5]
        y_min, y_max = self.domain[2], self.domain[3]
        x = x_PDE.clone()
        x.requires_grad = True
        u = self.forward(x)
        u_t = dde.grad.jacobian(u, x, i = 0, j = 2)
        u_xx = dde.grad.hessian(u, x, i = 0, j = 0)
        u_yy = dde.grad.hessian(u, x, i = 0, j = 1)

        x_coor = x[:, 0:1]
        y_coor = x[:, 1:2]
        t_coor = x[:, 2:]

        if self.case == "example_1":
            f = (1 - 2*np.pi**2) * torch.exp(-x[:, 2:3])*torch.sin(np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])
            u_sol = u_t - (u_xx + u_yy) - f
        elif self.case == "example_2":
            zt = 2 * (x_coor**2) * ((x_coor - x_max)**2) * (y_coor**2) * ((y_coor - y_max)**2) * (t_coor - 0.5)
            zxx = (12*(x_coor**2) - 12*x_max*x_coor + 2*(x_max**2)) * (y_coor**2) * ((y_coor - y_max)**2) * ((t_coor - 0.5)**2)
            zyy = (x_coor**2) * ((x_coor - x_max)**2) * (12*(y_coor**2) - 12*y_max*y_coor + 2*(y_max**2)) * ((t_coor - 0.5)**2)
            sources = zt - (zxx + zyy)
            u_sol = u_t - (u_xx + u_yy) - sources
        
        u_hat = torch.zeros(x.shape[0], 1).to(device)
        loss_pde = self.loss_function(u_sol, u_hat)
        return loss_pde


    def loss(self, x_BC, u_BC, x_PDE, w_bc, w_pde):
        loss_bc = self.lossbc(x_BC, u_BC)
        loss_pde = self.lossPDE(x_PDE)
        return w_bc * loss_bc + w_pde * loss_pde

    def error_l2_norm(self, x, u):
        error_u_predict = torch.norm(self.forward(x) - u, dim = 1, p = 2)
        error_u_predict_mean = error_u_predict.mean()
        return error_u_predict_mean, error_u_predict

    def relative_error_l2_norm(self, x, u):
        error_u_predict = torch.norm(self.forward(x) - u, dim = 1, p = 2)
        norm_u_real = torch.norm(u, dim = 1, p=2)
        rela_error = error_u_predict / (norm_u_real)
        rela_error_mean = rela_error.mean()
        return rela_error_mean

    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(X_train_bc, U_train_bc, X_train_Nf, w_bc = 1, w_pde = 1)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            loss2 = self.lossBC(X_test, U_test)
        return loss

def u_real(x, y, t, case):
    if case == "example_1":
        f = torch.exp(-t)*(torch.sin(np.pi*x) * torch.sin(np.pi*y))
    elif case == "example_2":
        f = (x**2)*(y**2)*((x-1)**2)*((y-1)**2)*((t - 0.5)**2)
    return f