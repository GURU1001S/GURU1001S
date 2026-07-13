import os
import re
import json
import glob
import numpy as np

# Define parameters
GRID_SIZE = 20
DURATION_SEC = 4.0

# Colormap control points
COLOR_POINTS = [
    (15, 23, 42),    # v = 0.0 (Slate 900)
    (88, 28, 135),   # v = 0.25 (Purple 800)
    (190, 24, 74),   # v = 0.5 (Pink 700)
    (234, 88, 12),   # v = 0.75 (Orange 600)
    (250, 204, 21)   # v = 1.0 (Yellow 400)
]

def get_color(v):
    # Clamp value between 0.0 and 1.0
    v = max(0.0, min(1.0, float(v)))
    idx = int(v * 4)
    if idx >= 4:
        return f"rgb{COLOR_POINTS[-1]}"
    t = (v - idx * 0.25) / 0.25
    c1 = COLOR_POINTS[idx]
    c2 = COLOR_POINTS[idx+1]
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

def main():
    # 1. Load training logs
    log_path = "frames/loss_log.json"
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}. Did you run render_pde.py first?")
        return
        
    with open(log_path, "r") as f:
        loss_log = json.load(f)
        
    # 2. Find and load all frame files
    frame_files = glob.glob("frames/frame_*.npy")
    if not frame_files:
        print("Error: No frame files found in frames/ directory.")
        return
        
    # Extract steps and sort them
    steps_data = []
    for fp in frame_files:
        match = re.search(r'frame_(\d+)\.npy', fp)
        if match:
            step = int(match.group(1))
            steps_data.append((step, fp))
            
    steps_data.sort(key=lambda x: x[0])
    num_frames = len(steps_data)
    print(f"Processing {num_frames} frames...")
    
    # 3. Read and downsample frames
    frames_downsampled = []
    for step, fp in steps_data:
        u = np.load(fp)  # 100x100 array
        # Downsample to GRID_SIZE x GRID_SIZE by slicing
        # u is [x, t]. Slicing at index spacing 100 // GRID_SIZE
        step_sz = u.shape[0] // GRID_SIZE
        u_down = u[::step_sz, ::step_sz]
        frames_downsampled.append((step, u_down))
        
    # 4. Generate SVG content
    svg_width = 440
    svg_height = 450
    
    # Build Defs and Styles
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>',
        '    .main-text { font-family: system-ui, -apple-system, sans-serif; }',
        '    .mono-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '  </style>',
        '  <!-- Background -->',
        '  <rect width="100%" height="100%" fill="#0b0f17" />'
    ]
    
    # 5. Render Heatmap Grid
    x_start = 50
    y_start = 50
    cell_size = 18
    overlap_size = 18.2 # overlap slightly to avoid rendering gaps
    
    svg_parts.append('  <!-- Heatmap Panel -->')
    svg_parts.append(f'  <g>')
    # Draw border around heatmap
    svg_parts.append(f'    <rect x="{x_start}" y="{y_start}" width="{GRID_SIZE*cell_size}" height="{GRID_SIZE*cell_size}" fill="none" stroke="#223047" stroke-width="1.5" />')
    
    # Add cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Gather values for this cell across all steps
            cell_colors = []
            for step, u_down in frames_downsampled:
                val = u_down[i, j]
                cell_colors.append(get_color(val))
                
            initial_color = cell_colors[0]
            values_str = ";".join(cell_colors)
            
            x_pos = x_start + i * cell_size
            y_pos = y_start + j * cell_size
            
            svg_parts.append(
                f'    <rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{overlap_size:.1f}" height="{overlap_size:.1f}" fill="{initial_color}">'
                f'      <animate attributeName="fill" calcMode="discrete" values="{values_str}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'    </rect>'
            )
            
    svg_parts.append('  </g>')
    
    # 6. Render Heatmap Axes and Labels
    svg_parts.append('  <!-- Heatmap Axes Labels -->')
    # X Axis
    svg_parts.append(f'  <text x="{x_start}" y="425" fill="#64748b" font-size="10" text-anchor="middle" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_size//2}" y="425" fill="#64748b" font-size="10" text-anchor="middle" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_size}" y="425" fill="#64748b" font-size="10" text-anchor="middle" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="{x_start + GRID_SIZE*cell_size//2}" y="438" fill="#94a3b8" font-size="11" text-anchor="middle" font-weight="600" class="main-text">Space (x)</text>')
    
    # T Axis
    svg_parts.append(f'  <text x="42" y="{y_start + 4}" fill="#64748b" font-size="10" text-anchor="end" class="main-text">0.0</text>')
    svg_parts.append(f'  <text x="42" y="{y_start + GRID_SIZE*cell_size//2 + 4}" fill="#64748b" font-size="10" text-anchor="end" class="main-text">0.5</text>')
    svg_parts.append(f'  <text x="42" y="{y_start + GRID_SIZE*cell_size + 4}" fill="#64748b" font-size="10" text-anchor="end" class="main-text">1.0</text>')
    svg_parts.append(f'  <text x="18" y="230" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90 18 230)" class="main-text">Time (t)</text>')
    
    # 7. Render SMIL Animated Stats
    svg_parts.append('  <!-- Animated Training Stats -->')
    for k, (step, _) in enumerate(frames_downsampled):
        step_str = str(step)
        # Fetch losses from log (default to 0.0 if not found)
        metrics = loss_log.get(step_str, {"loss": 0.0, "data_loss": 0.0, "pde_loss": 0.0})
        loss = metrics["loss"]
        
        # Display attribute timeline values for this frame's stats block
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        
        initial_display = "inline" if k == 0 else "none"
        
        # Generate stats group
        svg_parts.append(
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <text x="{x_start}" y="35" fill="#e2e8f0" font-size="12" class="main-text">Step: <tspan fill="#38bdf8" font-weight="700" class="mono-text">{step}</tspan></text>'
            f'    <text x="{x_start + GRID_SIZE*cell_size}" y="35" fill="#e2e8f0" font-size="12" text-anchor="end" class="main-text">Loss: <tspan fill="#ef4444" font-weight="700" class="mono-text">{loss:.6f}</tspan></text>'
            f'  </g>'
        )
        
    # Close Tag
    svg_parts.append('</svg>')
    
    # Ensure assets directory exists
    os.makedirs("assets", exist_ok=True)
    
    # Write SVG file
    output_path = "assets/pinn-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    print(f"Successfully generated minimal SMIL animated SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
