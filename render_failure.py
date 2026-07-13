import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
BETA = 10.0    # Advection wave velocity (convection dominant)
N_COLL = 2000  # Number of collocation points
N_IC = 200      # Number of initial condition points
N_BC = 200      # Number of boundary condition points
STEPS = 600     # Number of training steps
LR = 1e-3       # Learning rate

GRID_X = 15
GRID_Y = 20
DURATION_SEC = 4.0

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

# Diverging color scheme for wave visualization (Blue-Cyan -> Slate-Dark -> Crimson-Red)
def get_color(v):
    # Clamp value between -1.0 and 1.0
    v = max(-1.0, min(1.0, float(v)))
    if v < 0:
        # Interpolate between deep blue-cyan (-1.0) and dark background (0.0)
        t = v + 1.0
        c1 = (8, 145, 178)
        c2 = (7, 10, 19)
    else:
        # Interpolate between dark background (0.0) and hot crimson (1.0)
        t = v
        c1 = (7, 10, 19)
        c2 = (225, 29, 72)
        
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

def main():
    start_time = time.time()
    
    device = torch.device("cpu")
    model = PINN(hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 100x100 grid for full evaluation
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 1, 100)
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid, indexing='ij')
    
    x_eval_t = torch.tensor(x_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_eval_t = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    # Analytical Exact Solution: u(x, t) = sin(2*pi*(x - beta*t))
    u_exact = np.sin(2 * np.pi * (x_mesh - BETA * t_mesh))
    
    # Downsampling indices (15 in space, 20 in time)
    x_indices = np.linspace(0, 99, GRID_X, dtype=int)
    t_indices = np.linspace(0, 99, GRID_Y, dtype=int)
    
    u_exact_down = u_exact[x_indices][:, t_indices]
    
    frames_pinn = []
    frames_diagnostics = []
    
    print("Starting PINN training on Advection equation...")
    for step in range(STEPS + 1):
        if step > 0:
            model.train()
            
            # Compute data loss (IC + Periodic BCs)
            optimizer.zero_grad()
            
            # Initial Condition Loss (t=0)
            x_ic = torch.rand(N_IC, 1, device=device)
            t_ic = torch.zeros(N_IC, 1, device=device)
            u_ic_pred = model(x_ic, t_ic)
            u_ic_target = torch.sin(2 * np.pi * x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
            
            # Boundary Condition Loss (Periodic BCs)
            t_bc = torch.rand(N_BC, 1, device=device).requires_grad_(True)
            x_bc_0 = torch.zeros(N_BC, 1, device=device).requires_grad_(True)
            x_bc_1 = torch.ones(N_BC, 1, device=device).requires_grad_(True)
            
            u_bc_0 = model(x_bc_0, t_bc)
            u_bc_1 = model(x_bc_1, t_bc)
            
            u_bc_0_x = torch.autograd.grad(u_bc_0, x_bc_0, grad_outputs=torch.ones_like(u_bc_0), create_graph=True)[0]
            u_bc_1_x = torch.autograd.grad(u_bc_1, x_bc_1, grad_outputs=torch.ones_like(u_bc_1), create_graph=True)[0]
            
            loss_bc = torch.mean((u_bc_0 - u_bc_1) ** 2) + torch.mean((u_bc_0_x - u_bc_1_x) ** 2)
            
            loss_data = loss_ic + loss_bc
            loss_data.backward(retain_graph=True)
            
            # Data Gradient Norm
            grad_data_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_data_norm += p.grad.norm(2).item() ** 2
            grad_data_norm = grad_data_norm ** 0.5
            
            # Compute PDE Residual Loss
            optimizer.zero_grad()
            x_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            t_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            u_coll = model(x_coll, t_coll)
            u_x = torch.autograd.grad(u_coll, x_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_t = torch.autograd.grad(u_coll, t_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            
            loss_pde = torch.mean((u_t + BETA * u_x) ** 2)
            loss_pde.backward(retain_graph=True)
            
            # PDE Gradient Norm
            grad_pde_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_pde_norm += p.grad.norm(2).item() ** 2
            grad_pde_norm = grad_pde_norm ** 0.5
            
            # Optimization update step
            optimizer.zero_grad()
            loss_total = loss_data + loss_pde
            loss_total.backward()
            optimizer.step()
            
            grad_ratio = grad_pde_norm / (grad_data_norm + 1e-8)
            
        else:
            # Step 0 Initial diagnostics
            grad_ratio = 1.0
            with torch.no_grad():
                x_ic = torch.rand(N_IC, 1, device=device)
                t_ic = torch.zeros(N_IC, 1, device=device)
                u_ic_pred = model(x_ic, t_ic)
                u_ic_target = torch.sin(2 * np.pi * x_ic)
                loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
                
                t_bc = torch.rand(N_BC, 1, device=device)
                u_bc_0 = model(torch.zeros(N_BC, 1, device=device), t_bc)
                u_bc_1 = model(torch.ones(N_BC, 1, device=device), t_bc)
                loss_bc = torch.mean((u_bc_0 - u_bc_1) ** 2)
                
                loss_data = loss_ic + loss_bc
                
            x_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            t_coll = torch.rand(N_COLL, 1, device=device).requires_grad_(True)
            u_coll = model(x_coll, t_coll)
            u_x = torch.autograd.grad(u_coll, x_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            u_t = torch.autograd.grad(u_coll, t_coll, grad_outputs=torch.ones_like(u_coll), create_graph=True)[0]
            loss_pde = torch.mean((u_t + BETA * u_x) ** 2)
            loss_total = loss_data + loss_pde

        # Evaluate model diagnostics every 20 steps
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                u_pred = model(x_eval_t, t_eval_t).cpu().numpy().reshape(100, 100)
                
            # Compute Relative L2 Error
            l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            
            # Downsample PINN prediction
            u_pred_down = u_pred[x_indices][:, t_indices]
            
            frames_pinn.append(u_pred_down)
            frames_diagnostics.append((step, l2_error, grad_ratio))
            
            print(f"Step {step:03d}/600 | Total Loss: {loss_total.item():.5f} | Relative L2 Error: {l2_error:.4f} | Grad Ratio: {grad_ratio:.3f}")

    # 6. Generate Side-by-Side SMIL SVG
    svg_width = 440
    svg_height = 540
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>',
        '    .main-text { font-family: system-ui, -apple-system, sans-serif; }',
        '    .mono-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '    .header-text { font-family: system-ui, sans-serif; font-size: 10px; font-weight: 700; fill: #94a3b8; letter-spacing: 0.5px; text-anchor: middle; }',
        '  </style>',
        '  <!-- Background -->',
        '  <rect width="100%" height="100%" fill="#070a13" />'
    ]
    
    # Heatmap dimensions and positions
    x_start_pinn = 30
    x_start_exact = 245
    y_start = 50
    cell_w = 11
    cell_h = 18
    overlap_w = 11.2
    overlap_h = 18.2
    
    # Panel Headers
    svg_parts.append('  <!-- Panel Headers -->')
    svg_parts.append(f'  <text x="{x_start_pinn + (GRID_X*cell_w)//2}" y="42" class="header-text">VANILLA PINN</text>')
    svg_parts.append(f'  <text x="{x_start_exact + (GRID_X*cell_w)//2}" y="42" class="header-text">EXACT SOLUTION</text>')
    
    # 7. Render Heatmap Cells
    svg_parts.append('  <!-- PINN Prediction Heatmap -->')
    svg_parts.append('  <g>')
    svg_parts.append(f'    <rect x="{x_start_pinn}" y="{y_start}" width="{GRID_X*cell_w}" height="{GRID_Y*cell_h}" fill="none" stroke="#1e293b" stroke-width="1" />')
    for i in range(GRID_X):
        for j in range(GRID_Y):
            cell_colors = []
            for u_down in frames_pinn:
                val = u_down[i, j]
                cell_colors.append(get_color(val))
            initial_color = cell_colors[0]
            values_str = ";".join(cell_colors)
            
            x_pos = x_start_pinn + i * cell_w
            y_pos = y_start + j * cell_h
            svg_parts.append(
                f'    <rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{overlap_w:.1f}" height="{overlap_h:.1f}" fill="{initial_color}">'
                f'      <animate attributeName="fill" calcMode="discrete" values="{values_str}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'    </rect>'
            )
    svg_parts.append('  </g>')
    
    svg_parts.append('  <!-- Exact Solution Heatmap (Static Grid) -->')
    svg_parts.append('  <g>')
    svg_parts.append(f'    <rect x="{x_start_exact}" y="{y_start}" width="{GRID_X*cell_w}" height="{GRID_Y*cell_h}" fill="none" stroke="#1e293b" stroke-width="1" />')
    for i in range(GRID_X):
        for j in range(GRID_Y):
            val = u_exact_down[i, j]
            color = get_color(val)
            
            x_pos = x_start_exact + i * cell_w
            y_pos = y_start + j * cell_h
            svg_parts.append(
                f'    <rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{overlap_w:.1f}" height="{overlap_h:.1f}" fill="{color}" />'
            )
    svg_parts.append('  </g>')
    
    # 8. Render Axes and Labels
    svg_parts.append('  <!-- Axes and Labels -->')
    # T Axis Ticks
    svg_parts.append(f'  <text x="22" y="{y_start + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="22" y="{y_start + GRID_Y*cell_h//2 + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="22" y="{y_start + GRID_Y*cell_h + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="10" y="230" fill="#94a3b8" font-size="10" font-weight="600" text-anchor="middle" transform="rotate(-90 10 230)" class="main-text">Time (t)</text>')
    
    # Left Heatmap X Axis Ticks
    svg_parts.append(f'  <text x="{x_start_pinn}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="{x_start_pinn + GRID_X*cell_w//2}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="{x_start_pinn + GRID_X*cell_w}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="{x_start_pinn + GRID_X*cell_w//2}" y="438" fill="#64748b" font-size="9.5" text-anchor="middle" font-weight="600" class="main-text">Space (x)</text>')
    
    # Right Heatmap X Axis Ticks
    svg_parts.append(f'  <text x="{x_start_exact}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="{x_start_exact + GRID_X*cell_w//2}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="{x_start_exact + GRID_X*cell_w}" y="425" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="{x_start_exact + GRID_X*cell_w//2}" y="438" fill="#64748b" font-size="9.5" text-anchor="middle" font-weight="600" class="main-text">Space (x)</text>')

    # 9. Render HUD Statistics (Updates dynamically)
    svg_parts.append('  <!-- HUD Diagnostics Panel -->')
    num_frames = len(frames_diagnostics)
    for k, (step, l2_err, grad_ratio) in enumerate(frames_diagnostics):
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        
        initial_display = "inline" if k == 0 else "none"
        
        svg_parts.append(
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <text x="30" y="22" fill="#cbd5e1" font-size="11" class="main-text">Step: <tspan fill="#06b6d4" font-weight="700" class="mono-text">{step}</tspan></text>'
            f'    <text x="180" y="22" fill="#cbd5e1" font-size="11" text-anchor="middle" class="main-text">L2 Error: <tspan fill="#ef4444" font-weight="700" class="mono-text">{l2_err:.3f}</tspan></text>'
            f'    <text x="{svg_width - 30}" y="22" fill="#cbd5e1" font-size="11" text-anchor="end" class="main-text">Grad Ratio: <tspan fill="#eab308" font-weight="700" class="mono-text">{grad_ratio:.3f}</tspan></text>'
            f'  </g>'
        )

    # 10. Render Causal Citation & Link
    svg_parts.append('  <!-- Citation and Metadata -->')
    svg_parts.append(f'  <line x1="30" y1="455" x2="410" y2="455" stroke="#1e293b" stroke-width="1" />')
    
    # FM1 Badge
    svg_parts.append(f'  <rect x="30" y="468" width="36" height="15" rx="3" fill="#ef4444" opacity="0.9" />')
    svg_parts.append(f'  <text x="48" y="479" fill="#070a13" font-size="9" font-weight="800" text-anchor="middle" class="main-text">FM1</text>')
    svg_parts.append(f'  <text x="74" y="479" fill="#94a3b8" font-size="10" font-weight="600" class="main-text">Advection Propagation Failure</text>')
    
    # Body text
    svg_parts.append(
        f'  <text x="30" y="500" fill="#64748b" font-size="9.5" class="main-text">'
        f'    <tspan x="30" dy="0">One of six failure modes catalogued across 24+ controlled</tspan>'
        f'    <tspan x="30" dy="13">experiments in "An Atlas of PINN Failures".</tspan>'
        f'  </text>'
    )
    
    # Interactive Link Placeholder
    repo_url = "https://github.com/GURU1001S/An_Atlas_of_Physics_Informed_Neural_Network_Failures_The_Zugzwang_Thesis"
    svg_parts.append(
        f'  <a href="{repo_url}" target="_blank">'
        f'    <text x="30" y="530" fill="#475569" font-size="8.5" class="main-text">'
        f'      Live minimal reproduction. Read full manuscript: <tspan fill="#38bdf8" font-weight="600" text-decoration="underline">github.com/GURU1001S/An_Atlas_of_PINN_Failures</tspan>'
        f'    </text>'
        f'  </a>'
    )

    # Close Tag
    svg_parts.append('</svg>')
    
    # Ensure assets directory exists
    os.makedirs("assets", exist_ok=True)
    
    # Save SVG file
    output_path = "assets/zugzwang-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    end_time = time.time()
    print(f"Successfully generated PINN failure reproduction SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"Entire script finished in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
