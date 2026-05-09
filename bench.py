import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


# 1. Dataset Loader
def gen_testdata():
    
    data = np.load("Burgers.npz")
 

    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 2.LAAF (Locally Adaptive Activation)
class LAAF_Net(nn.Module):
    def __init__(self, layers, scale=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.a = nn.ParameterList()  # Scalable parameters for LAAF

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # One learnable parameter per layer for LAAF
            self.a.append(nn.Parameter(torch.ones(1) * scale))
            nn.init.xavier_normal_(self.layers[-1].weight)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            # LAAF formula: activation(n * a * layer(x))
            x = torch.tanh(self.a[i] * self.layers[i](x))
        return self.layers[-1](x)


# 3.RAR Logic
def compute_residual(model, x, nu=0.01 / np.pi):
    x.requires_grad_(True)
    u = model(x)

    grads = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]

    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    res = u_t + u * u_x - nu * u_xx
    return res


def get_rar_points(model, num_new_points=100):
    """Residual-based Adaptive Refinement (RAR)"""
    x_rand = torch.rand(2000, 1) * 2 - 1
    t_rand = torch.rand(2000, 1)
    X_cand = torch.cat([x_rand, t_rand], dim=1)

    res = compute_residual(model, X_cand)
    err = torch.abs(res).detach()

    _, idx = torch.topk(err.flatten(), num_new_points)
    return X_cand[idx]


# 4. Training
def train_model(X_data, u_data, use_stall_protection=True):
    layers = [2, 40, 40, 40, 1]
    model = LAAF_Net(layers)

    # Multi-Adam 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_pde = X_data.clone()  

    history = []
    nu = 0.01 / np.pi

    print(
        f"\nStarting {'Stall-Protected' if use_stall_protection else 'Standard'} Training..."
    )

    for it in range(3001):
        optimizer.zero_grad()

        u_pred = model(X_data)
        loss_data = torch.mean((u_pred - u_data) ** 2)

        res = compute_residual(model, X_pde, nu)
        weight = 10.0 if use_stall_protection else 1.0
        loss_pde = weight * torch.mean(res**2)

        total_loss = loss_data + loss_pde
        total_loss.backward()
        optimizer.step()

        if use_stall_protection and it % 1000 == 0 and it > 0:
            new_pts = get_rar_points(model, 100)
            X_pde = torch.cat([X_pde, new_pts], dim=0)
            print(f"    [RAR] Added 100 points. PDE Set size: {len(X_pde)}")

        if it % 500 == 0:
            history.append(total_loss.item())
            print(f"    Iter {it:4d} | Loss: {total_loss.item():.6e}")

    return model, history


# 5. Comparison
X, y = gen_testdata()
if X is not None:
    # Model 1: No Stall Protection (Baseline)
    model_baseline, hist_base = train_model(X, y, use_stall_protection=False)

    # Model 2: WF-PINN + RAR + LAAF (Stall Protected)
    model_stall, hist_stall = train_model(X, y, use_stall_protection=True)

    plt.figure(figsize=(10, 5))
    plt.semilogy(range(0, 3001, 500), hist_base, label="Standard PINN (Stalls)")
    plt.semilogy(range(0, 3001, 500), hist_stall, label="WF-PINN + RAR + LAAF")
    plt.title("Convergence Comparison: Stall vs. No Stall")
    plt.xlabel("Iterations")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

    # Final check
    u_base = model_baseline(X).detach()
    u_stall = model_stall(X).detach()

    l2_base = torch.norm(u_base - y) / torch.norm(y)
    l2_stall = torch.norm(u_stall - y) / torch.norm(y)

    print("\n" + "=" * 30)
    print(f"Baseline L2 Error: {l2_base.item():.6f}")
    print(f"Stall-Protected L2 Error: {l2_stall.item():.6f}")
    print("=" * 30)
