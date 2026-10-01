from matplotlib import gridspec, pyplot as plt

def plot_mesh(x, y, t, u, case, folder="", name = "", show = None, error = 0, loss = 0, iter = 0, time_idx = 0):
    X_2d_init = x[:, :, time_idx]
    Y_2d_init = y[:, :, time_idx]
    T_2d_init = t[:, :, time_idx]
    U_2d_init = u[:, :, time_idx]

    X = X_2d_init.detach().cpu()
    Y = Y_2d_init.detach().cpu()
    T = T_2d_init.detach().cpu()
    F_xt_init = U_2d_init.detach().cpu()
    real_time_val = T[0, time_idx].item()

    fig, ax = plt.subplots(1, 1, dpi=200)
    cp = ax.contourf(X, Y, F_xt_init, 20, cmap="coolwarm")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_xlabel("X")
    # ax.set_ylabel("x2")
    ax.set_ylabel("Y")

    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}\nTime: t = {real_time_val:.2f}, Rel error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}\nRel error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}")
    plt.savefig(f"{folder}/contour_{name}.png")  # Save the first image as an image file
    plt.close()

    plt.figure(dpi = 200)
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X.numpy(), Y.numpy(),F_xt_init.numpy(), cmap = "coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Temperature")

    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}\nTime: t = {real_time_val:.2f}, Rel error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}\nRel error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}, Time: t = {real_time_val:.2f}")
    
    if case == 'example_1':
        ax.set_zlim3d(-1,1)
    else:
        ax.set_zlim3d(0,1)

    plt.savefig(f"{folder}/{name}.png")  # Save the first image as an image file
    if show:
        plt.show()
    plt.close()

def plot_loss(iteration,loss_total, t, label_1, name_fig, case, time_idx = 0):
    T_2d_init = t[:, :, time_idx]
    T = T_2d_init.detach().cpu()
    real_time_val = T[0, time_idx].item()
    plt.figure(figsize = (12,5), dpi=200)
    plt.plot(iteration, loss_total, label = label_1)
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
