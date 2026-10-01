from matplotlib import gridspec, pyplot as plt

def plot_loss(iteration, loss_total, t, name_fig, case, losspde = None, lossbc = None,  time_idx = 0):
    T_2d_init = t[:, :, time_idx]
    T = T_2d_init.detach().cpu()
    real_time_val = T[0, time_idx].item()
    plt.figure(figsize = (12,5), dpi=200)
    if losspde is not None:
        plt.plot(iteration, losspde, label = 'Loss PDE', linewidth = 2, zorder = 3)
    if lossbc is not None:
        plt.plot(iteration, lossbc, label = 'Loss BC', linewidth = 2)
    plt.plot(iteration, loss_total, label = "Total Loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss Score')
    plt.yscale('log')
    plt.grid(True, alpha = 0.5)
    plt.legend()
    if case == 'example_1':
        plt.title(f'Loss Score per iteration for Case 1 when time = {real_time_val}')
        plt.savefig(f'{name_fig}_{case}_{real_time_val:.1f}.png')
    elif case == 'example_2':
        plt.title(f'Loss Score per iteration for Case 2 when time t = {real_time_val}')
        plt.savefig(f'{name_fig}_{case}_{real_time_val:.1f}.png')
    plt.close()
