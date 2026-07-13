import os
import json
import time
import sys
import torch
import torch.nn as nn
import numpy as np

# Progress bar fallback if tqdm is not installed
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc="", total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total or len(iterable)
            self.postfix = ""
            
        def __iter__(self):
            total = self.total
            desc_str = f"{self.desc}: " if self.desc else ""
            for i, item in enumerate(self.iterable):
                yield item
                percent = int(100 * (i + 1) / total)
                bar_len = 20
                filled_len = int(bar_len * (i + 1) // total)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                postfix_str = f" | {self.postfix}" if self.postfix else ""
                sys.stdout.write(f"\r{desc_str}[{bar}] {i+1}/{total} ({percent}%){postfix_str}")
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            
        def set_postfix(self, **kwargs):
            self.postfix = ", ".join(f"{k}: {v}" for k, v in kwargs.items())

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
ALPHA = 0.01  # Thermal diffusivity
N_COLL = 2000 # Number of collocation points
N_IC = 200    # Number of initial condition points
N_BC = 200    # Number of boundary condition points
STEPS = 600   # Number of training steps
LR = 1e-3     # Learning rate

# Model definition
class PINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# Initial condition: sum of two Gaussian bumps
def u_ic_fn(x):
    # Centered at 0.3 and 0.7 with standard deviation 0.08
    bump1 = torch.exp(-((x - 0.3) / 0.08) ** 2)
    bump2 = 0.5 * torch.exp(-((x - 0.7) / 0.08) ** 2)
    return bump1 + bump2

def main():
    start_time = time.time()
    
    # Ensure frames output directory exists
    os.makedirs("frames", exist_ok=True)
    
    # Use CPU explicitly as requested
    device = torch.device("cpu")
    
    model = PINN(hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Grid for evaluation (100x100)
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 1, 100)
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid, indexing='ij')
    
    x_eval_t = torch.tensor(x_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_eval_t = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    loss_log = {}
    
    pbar = tqdm(range(STEPS + 1), desc="Training PINN")
    for step in pbar:
        if step > 0:
            model.train()
            optimizer.zero_grad()
            
            # 1. Physics Loss (PDE residual: u_t - alpha * u_xx = 0)
            x_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            t_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            
            u_coll = model(x_coll, t_coll)
            
            # Compute gradients
            u_x = torch.autograd.grad(u_coll, x_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_t = torch.autograd.grad(u_coll, t_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            
            loss_pde = torch.mean((u_t - ALPHA * u_xx) ** 2)
            
            # 2. Initial Condition Loss (t=0)
            x_ic = torch.rand(N_IC, 1, device=device)
            t_ic = torch.zeros(N_IC, 1, device=device)
            u_ic_pred = model(x_ic, t_ic)
            u_ic_target = u_ic_fn(x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
            
            # 3. Boundary Condition Loss (x=0, x=1)
            t_bc = torch.rand(N_BC, 1, device=device)
            u_bc_0 = model(torch.zeros(N_BC, 1, device=device), t_bc)
            u_bc_1 = model(torch.ones(N_BC, 1, device=device), t_bc)
            loss_bc = torch.mean(u_bc_0 ** 2) + torch.mean(u_bc_1 ** 2)
            
            loss_data = loss_ic + loss_bc
            loss = loss_data + loss_pde
            
            loss.backward()
            optimizer.step()
        else:
            # Step 0: Calculate initial losses
            model.eval()
            
            # Compute PDE loss at step 0 with gradients enabled locally
            x_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            t_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            u_coll = model(x_coll, t_coll)
            u_x = torch.autograd.grad(u_coll, x_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_t = torch.autograd.grad(u_coll, t_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            loss_pde = torch.mean((u_t - ALPHA * u_xx) ** 2)
            
            with torch.no_grad():
                x_ic = torch.rand(N_IC, 1, device=device)
                t_ic = torch.zeros(N_IC, 1, device=device)
                u_ic_pred = model(x_ic, t_ic)
                u_ic_target = u_ic_fn(x_ic)
                loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
                
                t_bc = torch.rand(N_BC, 1, device=device)
                u_bc_0 = model(torch.zeros(N_BC, 1, device=device), t_bc)
                u_bc_1 = model(torch.ones(N_BC, 1, device=device), t_bc)
                loss_bc = torch.mean(u_bc_0 ** 2) + torch.mean(u_bc_1 ** 2)
                
                loss_data = loss_ic + loss_bc
                loss = loss_data + loss_pde

        # Evaluate and log every 20 steps
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                u_eval = model(x_eval_t, t_eval_t).cpu().numpy().reshape(100, 100)
            
            # Save surface evaluation
            np.save(f"frames/frame_{step:03d}.npy", u_eval)
            
            loss_val = loss.item()
            data_loss_val = loss_data.item()
            pde_loss_val = loss_pde.item()
            
            loss_log[str(step)] = {
                "loss": loss_val,
                "data_loss": data_loss_val,
                "pde_loss": pde_loss_val
            }
            
            # Update progress bar description with latest loss metrics
            pbar.set_postfix(loss=f"{loss_val:.6f}")
            
    # Save loss log
    with open("frames/loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=4)
        
    end_time = time.time()
    print(f"\nTraining completed successfully in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
