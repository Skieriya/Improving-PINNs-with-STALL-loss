import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Dataset
print("\n" + "="*60)
print("BURGERS EQUATION DATASET")
print("="*60)

data = np.load('Burgers.npz')
t_data, x_data, u_exact = data['t'], data['x'], data['usol']

print(f"t shape: {t_data.shape}, x shape: {x_data.shape}, u shape: {u_exact.shape}")

xx, tt = np.meshgrid(x_data.flatten(), t_data.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = u_exact.T.flatten()[:, None]


sample_x, sample_t = 4, 4
x_train = x_data.flatten()[::sample_x]
t_train = t_data.flatten()[::sample_t]
xx_train, tt_train = np.meshgrid(x_train, t_train)

X_data = np.vstack((np.ravel(xx_train), np.ravel(tt_train))).T
y_data = u_exact[::sample_x, :].T[::sample_t, :].flatten()[:, None]

nx_train = len(x_train)
nt_train = len(t_train)

print(f"Full grid: {X_full.shape[0]} points")
print(f"Training grid: {nx_train}x{nt_train} = {nx_train*nt_train} points")
print(f"Training x: {x_train.shape}, t: {t_train.shape}")

X_data_t = torch.tensor(X_data, dtype=torch.float32, device=device)
y_data_t = torch.tensor(y_data, dtype=torch.float32, device=device)

X_full_t = torch.tensor(X_full, dtype=torch.float32, device=device)
y_full_t = torch.tensor(y_full, dtype=torch.float32, device=device)

# Whitening parameters
mu_y = y_data_t.mean()
std_y = y_data_t.std() + 1e-8

print(f"Data range: [{y_data_t.min().item():.4f}, {y_data_t.max().item():.4f}]")
print(f"Whitening params: mu={mu_y.item():.4f}, std={std_y.item():.4f}")

# 2. WF-PINN
class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_features: int, scale: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class WF_Pinn(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_fourier: int = 32,
                 num_layers: int = 3):
        super().__init__()
        self.fourier = FourierFeatures(input_dim, num_fourier, scale=1.0)
        fourier_dim = input_dim + 2 * num_fourier

        layers = []
        layers.append(nn.Linear(fourier_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x_fourier = self.fourier(x)
        x_combined = torch.cat([x, x_fourier], dim=-1)
        return self.net(x_combined)


# 3. PDE RESIDUAL
def compute_pde_residual(model: nn.Module, x: torch.Tensor, nu: float = 0.01/np.pi) -> torch.Tensor:
    x.requires_grad_(True)
    u = model(x)

    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    u_xx = grad_u_x[:, 0:1]

    residual = u_t + u * u_x - nu * u_xx
    return residual


def compute_all_metrics(model: nn.Module, X: torch.Tensor, y_true: torch.Tensor,
                       nu: float = 0.01/np.pi) -> Tuple[Dict, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    metrics = {}
    metrics['L2_error'] = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    metrics['Max_abs_error'] = torch.max(torch.abs(y_pred - y_true)).item()
    metrics['Mean_abs_error'] = torch.mean(torch.abs(y_pred - y_true)).item()

    residual = compute_pde_residual(model, X.clone(), nu)
    metrics['PDE_mean_residual'] = torch.mean(torch.abs(residual)).item()
    metrics['PDE_max_residual'] = torch.max(torch.abs(residual)).item()
    metrics['PDE_rms_residual'] = torch.sqrt(torch.mean(residual ** 2)).item()

    boundary_mask = (torch.abs(X[:, 0] - (-1.0)) < 0.01) | (torch.abs(X[:, 0] - 1.0) < 0.01)
    if boundary_mask.any():
        metrics['BC_error'] = torch.mean(torch.abs(y_pred[boundary_mask])).item()
    else:
        metrics['BC_error'] = 0.0

    return metrics, y_pred


# 4. LOSS FUNCTIONS
def compute_loss_no_stall(model: nn.Module, X: torch.Tensor, y_true: torch.Tensor,
                          nu: float = 0.01/np.pi, lambda_pde: float = 1.0,
                          lambda_bc: float = 1.0) -> Tuple:
    x = X.clone()
    x.requires_grad_(True)

    y_pred = model(x)
    y_pred = torch.clamp(y_pred, min=-10, max=10)

    grad_u = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred),
                              create_graph=True, retain_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True)[0][:, 0:1]

    residual = u_t + y_pred * u_x - nu * u_xx

    loss_data = torch.mean((y_pred - y_true) ** 2)
    loss_pde = torch.mean(residual ** 2)

    # Boundary: u=0 at x=-1 and x=1
    bc_mask = (torch.abs(x[:, 0] + 1.0) < 0.01) | (torch.abs(x[:, 0] - 1.0) < 0.01)
    if bc_mask.any():
        loss_bc = torch.mean(y_pred[bc_mask] ** 2)
    else:
        loss_bc = torch.tensor(0.0, device=x.device)

    loss_data = torch.where(torch.isnan(loss_data), torch.zeros_like(loss_data), loss_data)
    loss_pde = torch.where(torch.isnan(loss_pde), torch.zeros_like(loss_pde), loss_pde)
    loss_bc = torch.where(torch.isnan(loss_bc), torch.zeros_like(loss_bc), loss_bc)

    return loss_data, loss_pde, loss_bc, loss_data + lambda_pde * loss_pde + lambda_bc * loss_bc


def compute_loss_stall(model: nn.Module, X: torch.Tensor, y_true: torch.Tensor,
                       mu: float, std: float, nu: float = 0.01/np.pi,
                       lambda_pde: float = 0.1, lambda_bc: float = 1.0) -> Tuple:
    x = X.clone()
    x.requires_grad_(True)

    y_pred = model(x)
    y_pred = torch.clamp(y_pred, min=-10, max=10)

    # Whiten
    y_pred_whitened = (y_pred - mu) / std
    y_true_whitened = (y_true - mu) / std

    grad_u = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred),
                              create_graph=True, retain_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True)[0][:, 0:1]

    residual = u_t + y_pred * u_x - nu * u_xx

    loss_likelihood = torch.mean((y_pred_whitened - y_true_whitened) ** 2)
    loss_pde = torch.mean(residual ** 2)

    bc_mask = (torch.abs(x[:, 0] + 1.0) < 0.01) | (torch.abs(x[:, 0] - 1.0) < 0.01)
    if bc_mask.any():
        loss_bc = torch.mean(y_pred[bc_mask] ** 2)
    else:
        loss_bc = torch.tensor(0.0, device=x.device)

    loss_likelihood = torch.where(torch.isnan(loss_likelihood), torch.zeros_like(loss_likelihood), loss_likelihood)
    loss_pde = torch.where(torch.isnan(loss_pde), torch.zeros_like(loss_pde), loss_pde)
    loss_bc = torch.where(torch.isnan(loss_bc), torch.zeros_like(loss_bc), loss_bc)

    return loss_likelihood, loss_pde, loss_bc, loss_likelihood + lambda_pde * loss_pde + lambda_bc * loss_bc


# 5. TRAINING
def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              use_stall: bool = False, mu: float = None, std: float = None,
              epochs: int = 3000, nu: float = 0.01/np.pi) -> Dict:

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = {'total': [], 'data': [], 'pde': [], 'bc': []}

    for it in range(epochs):
        optimizer.zero_grad()

        if use_stall:
            loss_lik, loss_pde, loss_bc, loss = compute_loss_stall(model, X, y, mu, std, nu)
            history['data'].append(loss_lik.item())
        else:
            loss_data, loss_pde, loss_bc, loss = compute_loss_no_stall(model, X, y, nu)
            history['data'].append(loss_data.item())

        if torch.isnan(loss):
            print(f"NaN at iter {it}, skipping...")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history['total'].append(loss.item())
        history['pde'].append(loss_pde.item())
        history['bc'].append(loss_bc.item())

        if it % 500 == 0:
            if use_stall:
                print(f"STALL [{it:4d}]: Loss={loss.item():.6f}, Lik={loss_lik.item():.6f}, PDE={loss_pde.item():.6f}")
            else:
                print(f"No-STALL [{it:4d}]: Loss={loss.item():.6f}, Data={loss_data.item():.6f}, PDE={loss_pde.item():.6f}")

    return history



print("\n" + "="*60)
print("TRAINING WF-PINN MODELS")
print("="*60)
print("Comparing: WF-PINN + STALL vs WF-PINN No-STALL")
print("-"*60)

nu = 0.01 / np.pi

# Models
wf_stall = WF_Pinn(input_dim=2, hidden_dim=64, num_fourier=32, num_layers=3).to(device)
wf_no_stall = WF_Pinn(input_dim=2, hidden_dim=64, num_fourier=32, num_layers=3).to(device)

print(f"WF-PINN parameters: {sum(p.numel() for p in wf_stall.parameters())}")

# Train with STALL
print("\n>>> Training WF-PINN + STALL <<<")
history_stall = train_model(wf_stall, X_data_t, y_data_t, use_stall=True,
                        mu=mu_y.item(), std=std_y.item(), epochs=3000, nu=nu)

# Train without STALL
print("\n>>> Training WF-PINN No-STALL <<<")
history_no_stall = train_model(wf_no_stall, X_data_t, y_data_t, use_stall=False, epochs=3000, nu=nu)


# 7. EVALUATION 
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

metrics_stall, pred_stall = compute_all_metrics(wf_stall, X_full_t, y_full_t, nu)
metrics_no_stall, pred_no_stall = compute_all_metrics(wf_no_stall, X_full_t, y_full_t, nu)

print(f"\n{'Metric':<30} {'STALL':>12} {'No-STALL':>12} {'Winner':>10}")
print("-"*60)
print(f"{'L2 Error':<30} {metrics_stall['L2_error']:>12.6f} {metrics_no_stall['L2_error']:>12.6f} ", end="")
print("STALL" if metrics_stall['L2_error'] < metrics_no_stall['L2_error'] else "No-STALL")

print(f"{'Max Abs Error':<30} {metrics_stall['Max_abs_error']:>12.6f} {metrics_no_stall['Max_abs_error']:>12.6f} ", end="")
print("STALL" if metrics_stall['Max_abs_error'] < metrics_no_stall['Max_abs_error'] else "No-STALL")

print(f"{'PDE RMS Residual':<30} {metrics_stall['PDE_rms_residual']:>12.6f} {metrics_no_stall['PDE_rms_residual']:>12.6f} ", end="")
print("STALL" if metrics_stall['PDE_rms_residual'] < metrics_no_stall['PDE_rms_residual'] else "No-STALL")

print(f"{'BC Error':<30} {metrics_stall['BC_error']:>12.6f} {metrics_no_stall['BC_error']:>12.6f} ", end="")
print("STALL" if metrics_stall['BC_error'] < metrics_no_stall['BC_error'] else "No-STALL")


# 8. VISUALIZATION
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

nx = x_data.flatten().shape[0]  # 256
nt = t_data.flatten().shape[0]   # 100

assert nx * nt == pred_stall.shape[0], f"Size mismatch: {nx}*{nt}={nx*nt} vs {pred_stall.shape[0]}"

fig = plt.figure(figsize=(18, 10))

# Ground truth - u_exact is (256, 100)
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.imshow(u_exact.T, extent=[x_data.min(), x_data.max(), t_data.max(), t_data.min()],
                 aspect='auto', cmap='viridis')
ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
plt.colorbar(im1, ax=ax1)

# STALL prediction - reshape to (nt, nx) = (100, 256)
ax2 = fig.add_subplot(2, 3, 2)
pred_stall_grid = pred_stall.cpu().reshape(nt, nx).detach().numpy()
im2 = ax2.imshow(pred_stall_grid, extent=[x_data.min(), x_data.max(), t_data.max(), t_data.min()],
                aspect='auto', cmap='viridis')
ax2.set_title(f'WF-PINN + STALL\nL2={metrics_stall["L2_error"]:.4f}', fontsize=12, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
plt.colorbar(im2, ax=ax2)

# No-STALL prediction
ax3 = fig.add_subplot(2, 3, 3)
pred_no_stall_grid = pred_no_stall.cpu().reshape(nt, nx).detach().numpy()
im3 = ax3.imshow(pred_no_stall_grid, extent=[x_data.min(), x_data.max(), t_data.max(), t_data.min()],
                aspect='auto', cmap='viridis')
ax3.set_title(f'WF-PINN No-STALL\nL2={metrics_no_stall["L2_error"]:.4f}', fontsize=12, fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
plt.colorbar(im3, ax=ax3)

# Error STALL
ax4 = fig.add_subplot(2, 3, 4)
err_stall = (pred_stall_grid - u_exact.T)
im4 = ax4.imshow(err_stall, extent=[x_data.min(), x_data.max(), t_data.max(), t_data.min()],
                aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.2)
ax4.set_title('STALL Error', fontsize=12, fontweight='bold')
ax4.set_xlabel('x')
ax4.set_ylabel('t')
plt.colorbar(im4, ax=ax4)

# Error No-STALL
ax5 = fig.add_subplot(2, 3, 5)
err_no_stall = (pred_no_stall_grid - u_exact.T)
im5 = ax5.imshow(err_no_stall, extent=[x_data.min(), x_data.max(), t_data.max(), t_data.min()],
                aspect='auto', cmap='RdBu', vmin=-0.2, vmax=0.2)
ax5.set_title('No-STALL Error', fontsize=12, fontweight='bold')
ax5.set_xlabel('x')
ax5.set_ylabel('t')
plt.colorbar(im5, ax=ax5)

# Loss history
ax6 = fig.add_subplot(2, 3, 6)
ax6.semilogy(history_stall['total'], label='STALL', alpha=0.8, linewidth=2)
ax6.semilogy(history_no_stall['total'], label='No-STALL', alpha=0.8, linewidth=2)
ax6.set_title('Training Loss', fontsize=12, fontweight='bold')
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Loss (log)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wf_pinn_burgers_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: wf_pinn_burgers_comparison.png")


# 9. SUMMARY
print("\n" + "="*70)
print("FINAL SUMMARY: WF-PINN STALL vs NO-STALL (Full Grid Evaluation)")
print("="*70)
print(f"""
Models compared:
  - WF-PINN + STALL: Fourier Features + Whitened Likelihood
  - WF-PINN No-STALL: Fourier Features + Standard MSE

Training data: {X_data_t.shape[0]} points (subsampled by {sample_x}x{sample_t})
Full evaluation grid: {X_full_t.shape[0]} points

WF-PINN + STALL:
  - L2 Error: {metrics_stall['L2_error']:.6f}
  - PDE RMS: {metrics_stall['PDE_rms_residual']:.6f}
  - BC Error: {metrics_stall['BC_error']:.6f}

WF-PINN No-STALL:
  - L2 Error: {metrics_no_stall['L2_error']:.6f}
  - PDE RMS: {metrics_no_stall['PDE_rms_residual']:.6f}
  - BC Error: {metrics_no_stall['BC_error']:.6f}

Winner: {'STALL' if metrics_stall['L2_error'] < metrics_no_stall['L2_error'] else 'No-STALL'}
""")