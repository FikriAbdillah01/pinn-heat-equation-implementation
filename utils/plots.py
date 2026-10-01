from matplotlib import gridspec, pyplot as plt

def plot_mesh(x, t, y, case, folder="", name = "", show = None, error = 0, loss = 0, iter = 0):
    X, T = x.cpu(), t.cpu()
    F_xt = y.cpu()

    fig, ax = plt.subplots(1, 1, dpi=200)
    cp = ax.contourf(X, T, F_xt, 20, cmap="coolwarm")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_xlabel("x")
    # ax.set_ylabel("x2")
    ax.set_ylabel("t")
    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f} \n  Relative error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter} \n Relative error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}")
    plt.savefig(f"{folder}/contour_{name}.png")  # Save the first image as an image file
    plt.close()

    plt.figure(dpi=200)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X.numpy(), T.numpy(), F_xt.numpy(), cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    
    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f} \n  Relative error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter} \n Relative error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}")
        
    if case=='example_1':
        ax.set_zlim3d(-1, 1) 
    else:
        ax.set_zlim3d(0, 6) 
        
    plt.savefig(f"{folder}/{name}.png")  # Save the first image as an image file
    if show:
        plt.show()
    plt.close()

def plot_loss(iteration,loss_total, label_1, name_fig, case):
    plt.figure(figsize = (12,5), dpi=200)
    plt.plot(iteration, loss_total, label = label_1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Score')
    plt.yscale('log')
    plt.grid(True, alpha = 0.5)
    plt.legend()
    if case == 'example_1':
        plt.title(f'Loss Score per iteration for Case 1')
        plt.savefig(f'{name_fig}{case}.png')
    elif case == 'example_2':
        plt.title(f'Loss Score per iteration for Case 2')
        plt.savefig(f'{name_fig}{case}.png')
    plt.close()
