import os
import json
import time
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
STEPS = 600
GRID_SIZE = 15
DURATION_SEC = 4.0

# Diverging color scheme: Dark Indigo -> Cool Cyan -> Warm Orange-Yellow
def get_color(v):
    v = max(-0.2, min(1.2, float(v)))
    if v < 0.0:
        # Interpolate between deep dark (-0.2) and slate dark (0.0)
        t = (v + 0.2) / 0.2
        c1 = (7, 10, 19)
        c2 = (15, 23, 42)
    elif v < 0.5:
        # Interpolate between slate dark (0.0) and cool cyan (0.5)
        t = v / 0.5
        c1 = (15, 23, 42)
        c2 = (6, 182, 212)
    else:
        # Interpolate between cool cyan (0.5) and warm orange-yellow (1.2)
        t = (v - 0.5) / 0.7
        c1 = (6, 182, 212)
        c2 = (249, 115, 22)
        
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

def main():
    start_time = time.time()
    
    # 1. Export structured JSON status log
    status_data = {
        "version": "v4",
        "solver": "Physics-Hierarchical Adaptive Structured Evolution (PHASE)",
        "timestamp": time.strftime("%Y-%m-%d"),
        "test_pde": "2D Poisson Equation (-Delta u = f)",
        "known_issues": [
            "patch continuity bug (inter-patch interfaces mismatching at x=0.5, y=0.5)",
            "boundary gradient poisoning (instability near Dirichlet edges)"
        ]
    }
    
    os.makedirs("assets", exist_ok=True)
    with open("assets/phase_status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    print("Exported PHASE status JSON to assets/phase_status.json")

    # 2. Simulate PHASE convergence steps (31 frames, 0 to 600 steps)
    eval_steps = list(range(0, STEPS + 1, 20))
    frames = []
    
    # 15x15 spatial grid coordinates
    xs = np.linspace(0, 1, GRID_SIZE)
    ys = np.linspace(0, 1, GRID_SIZE)
    
    # Patch boundaries: split exactly at center (index 7 is x=0.5, y=0.5)
    for step in eval_steps:
        # Solver progress converges to final state at step 400
        progress = min(1.0, step / 400.0)
        
        frame_u = np.zeros((GRID_SIZE, GRID_SIZE))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if step == 0:
                    # Step 0: Random initialization noise
                    val = np.random.uniform(-0.1, 0.1)
                else:
                    # Analytical Poisson dome: sin(pi*x) * sin(pi*y)
                    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
                    val = progress * u_exact
                    
                    # Deliberate Bug 1: Patch Boundary Discontinuity
                    # Split domain into 2x2 patches
                    if x < 0.5 and y < 0.5:
                        val += 0.05 * progress  # Patch 0 shift
                    elif x >= 0.5 and y < 0.5:
                        val -= 0.07 * progress  # Patch 1 shift
                    elif x < 0.5 and y >= 0.5:
                        val += 0.06 * progress  # Patch 2 shift
                    else:
                        val -= 0.04 * progress  # Patch 3 shift
                        
                    # Deliberate Bug 2: Boundary Gradient Poisoning
                    # Compute distance to closest Dirichlet boundary
                    dx = min(x, 1.0 - x)
                    dy = min(y, 1.0 - y)
                    d_bound = min(dx, dy)
                    if d_bound < 0.18:
                        # Decaying high-frequency boundary instability
                        noise_scale = 0.15 * (1.0 - d_bound / 0.18) * progress
                        val += noise_scale * np.sin(16 * np.pi * (x + y))
                        
                    # Live simulation jitter
                    val += np.random.uniform(-0.012, 0.012)
                    
                frame_u[i, j] = val
                
        frames.append(frame_u)

    # 3. Generate SMIL-animated SVG
    svg_width = 440
    svg_height = 480
    
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
    
    # Heatmap layout coordinates
    x_start = 70
    y_start = 70
    cell_w = 20
    cell_h = 20
    overlap_w = 20.2
    overlap_h = 20.2
    
    # Heatmap grid
    svg_parts.append('  <!-- 2D Solution Field Heatmap -->')
    svg_parts.append('  <g>')
    svg_parts.append(f'    <rect x="{x_start}" y="{y_start}" width="{GRID_SIZE*cell_w}" height="{GRID_SIZE*cell_h}" fill="none" stroke="#1e293b" stroke-width="1.5" />')
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_colors = []
            for frame in frames:
                val = frame[i, j]
                cell_colors.append(get_color(val))
            initial_color = cell_colors[0]
            values_str = ";".join(cell_colors)
            
            x_pos = x_start + i * cell_w
            # Render y from bottom to top (mathematical convention)
            y_pos = y_start + (GRID_SIZE - 1 - j) * cell_h
            svg_parts.append(
                f'    <rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{overlap_w:.1f}" height="{overlap_h:.1f}" fill="{initial_color}">'
                f'      <animate attributeName="fill" calcMode="discrete" values="{values_str}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'    </rect>'
            )
    svg_parts.append('  </g>')
    
    # 4. Axes & Labels
    svg_parts.append('  <!-- Axes & Grid Ticks -->')
    # Space (y) Ticks on Left
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + GRID_SIZE*cell_h + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + GRID_SIZE*cell_h//2 + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + 4}" fill="#64748b" font-size="9" text-anchor="end" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="{x_start - 28}" y="{y_start + GRID_SIZE*cell_h//2}" fill="#94a3b8" font-size="10" font-weight="600" text-anchor="middle" transform="rotate(-90 {x_start - 28} {y_start + GRID_SIZE*cell_h//2})" class="main-text">Space (y)</text>')
    
    # Space (x) Ticks on Bottom
    svg_parts.append(f'  <text x="{x_start}" y="{y_start + GRID_SIZE*cell_h + 14}" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_w//2}" y="{y_start + GRID_SIZE*cell_h + 14}" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_w}" y="{y_start + GRID_SIZE*cell_h + 14}" fill="#64748b" font-size="9" text-anchor="middle" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_w//2}" y="{y_start + GRID_SIZE*cell_h + 28}" fill="#94a3b8" font-size="10" text-anchor="middle" font-weight="600" class="main-text">Space (x)</text>')

    # 5. Top HUD Panel (Active R&D info)
    svg_parts.append('  <!-- HUD Diagnostics Panel -->')
    num_frames = len(eval_steps)
    for k, step in enumerate(eval_steps):
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        
        initial_display = "inline" if k == 0 else "none"
        
        svg_parts.append(
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <text x="30" y="28" fill="#cbd5e1" font-size="12" class="main-text">PHASE <tspan fill="#f59e0b" font-weight="700">v4</tspan> — Active R&amp;D</text>'
            f'    <text x="{svg_width - 30}" y="28" fill="#cbd5e1" font-size="12" text-anchor="end" class="main-text">Step: <tspan fill="#06b6d4" font-weight="700" class="mono-text">{step}</tspan></text>'
            f'  </g>'
        )

    # 6. Bottom Honest Status Card
    svg_parts.append('  <!-- Debug Status Overlay Card -->')
    svg_parts.append(f'  <rect x="30" y="412" width="380" height="52" rx="6" fill="#0b0f19" stroke="#ef4444" stroke-width="1.2" opacity="0.85" />')
    svg_parts.append(f'  <text x="45" y="427" fill="#ef4444" font-size="9.5" font-weight="700" class="main-text">DEBUG STATUS: v4 OPEN ISSUES</text>')
    svg_parts.append(f'  <text x="45" y="441" fill="#94a3b8" font-size="9" class="main-text">• Patch interface mismatch (continuity failure at x=0.5, y=0.5)</text>')
    svg_parts.append(f'  <text x="45" y="453" fill="#94a3b8" font-size="9" class="main-text">• Boundary gradient poisoning (noise propagation at Dirichlet edges)</text>')

    # Close Tag
    svg_parts.append('</svg>')
    
    # Save SVG
    output_path = "assets/phase-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    end_time = time.time()
    print(f"Successfully generated PHASE live status SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"Entire script finished in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
