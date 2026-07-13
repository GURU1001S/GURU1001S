import os
import datetime
import hashlib
import json
import numpy as np

# 1. Setup deterministic seed based on today's calendar date
today_str = datetime.date.today().strftime("%Y-%m-%d")
seed = int(hashlib.md5(today_str.encode()).hexdigest(), 16) % 1000000
np.random.seed(seed)

# Parameters
STEPS = 600
DURATION_SEC = 4.0
NODE_LABELS = ["A", "B", "C", "D", "E"]
NUM_NODES = len(NODE_LABELS)

# Fixed signature colors for the nodes and stream lines
NODE_COLORS = ["#06b6d4", "#f97316", "#ec4899", "#a855f7", "#10b981"]

def get_bezier_path(x1, y1, x2, y2, curvature=12):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if L == 0:
        return f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
    nx = -dy / L
    ny = dx / L
    cx = mx + curvature * nx
    cy = my + curvature * ny
    return f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"

def main():
    # 2. Generate a random true DAG structure
    true_weights = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if np.random.rand() < 0.5:
                w = np.random.uniform(0.25, 0.55) * np.random.choice([-1, 1])
                true_weights[i, j] = w
                
    # Ensure we always have at least 3 true edges for visual richness
    while np.count_nonzero(true_weights) < 3:
        i = np.random.randint(0, NUM_NODES - 1)
        j = np.random.randint(i + 1, NUM_NODES)
        if true_weights[i, j] == 0:
            true_weights[i, j] = np.random.uniform(0.25, 0.55) * np.random.choice([-1, 1])

    # 3. Choose a random intervention step and target node
    int_step = np.random.randint(260, 340)
    int_node = np.random.randint(0, NUM_NODES - 1)

    # 4. Simulate streaming time series
    X = np.zeros((STEPS, NUM_NODES))
    X[0] = np.random.normal(0, 0.2, NUM_NODES)
    
    for s in range(1, STEPS):
        for j in range(NUM_NODES):
            if s >= int_step and j == int_node:
                X[s, j] = 4.0 + np.random.normal(0, 0.1)
            else:
                val = 0.3 * X[s-1, j]
                for i in range(NUM_NODES):
                    if true_weights[i, j] != 0:
                        val += true_weights[i, j] * X[s-1, i]
                X[s, j] = val + np.random.normal(0, 0.15)

    # 5. Perform streaming causal inference (VAR(1) OLS) every 20 steps
    eval_steps = list(range(0, STEPS + 1, 20))
    frames_data = []

    for t in eval_steps:
        est_weights = np.zeros((NUM_NODES, NUM_NODES))
        if t >= 20:
            Z = X[0:t-1, :]
            for j in range(NUM_NODES):
                Y = X[1:t, j]
                try:
                    w_est, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
                    for i in range(NUM_NODES):
                        est_weights[i, j] = w_est[i]
                except np.linalg.LinAlgError:
                    pass
                    
        node_vals = X[min(t, STEPS - 1)]
        frames_data.append((t, est_weights, node_vals))

    # 6. Generate SVG Content
    svg_width = 440
    svg_height = 450
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>',
        '    .main-text { font-family: system-ui, -apple-system, sans-serif; }',
        '    .mono-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '    .node-label { font-family: system-ui, sans-serif; font-weight: 700; fill: #f8fafc; font-size: 11px; text-anchor: middle; dominant-baseline: middle; }',
        '  </style>',
        '  <defs>',
        '    <!-- Arrowhead marker definition -->',
        '    <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">',
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />',
        '    </marker>',
        '  </defs>',
        '  <!-- Background -->',
        '  <rect width="100%" height="100%" fill="#070a13" />'
    ]
    
    # Left Panel: Causal Graph
    # Center: 220, cy: 250, Radius: 120
    cx = 220
    cy = 250
    R = 120
    angles = [2 * np.pi * i / NUM_NODES - np.pi / 2 for i in range(NUM_NODES)]
    node_coords = []
    for a in angles:
        nx = cx + R * np.cos(a)
        ny = cy + R * np.sin(a)
        node_coords.append((nx, ny))

    # Concentric background circles on Left Panel
    svg_parts.append('  <!-- Radar Grid -->')
    svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="60" fill="none" stroke="#1e293b" stroke-dasharray="3 6" opacity="0.25" stroke-width="1" />')
    svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="120" fill="none" stroke="#1e293b" stroke-dasharray="4 8" opacity="0.2" stroke-width="1" />')
    svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="180" fill="none" stroke="#1e293b" stroke-dasharray="5 10" opacity="0.15" stroke-width="1" />')

    # 7. Render Edges (Paths & Particles)
    svg_parts.append('  <!-- Curved Causal Edges -->')
    svg_parts.append('  <g>')
    
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i == j:
                continue
                
            x1, y1 = node_coords[i]
            x2, y2 = node_coords[j]
            bezier_path = get_bezier_path(x1, y1, x2, y2, curvature=12)
            
            edge_strokes = []
            edge_widths = []
            edge_opacities = []
            particle_opacities = []
            
            for t, est_weights, _ in frames_data:
                w = est_weights[i, j]
                if abs(w) > 0.15:
                    color = "#06b6d4" if w > 0 else "#f43f5e"
                    width = 1.0 + 4.0 * min(1.0, abs(w))
                    opacity = min(0.9, 0.35 + abs(w))
                    part_opacity = min(0.95, 0.5 + abs(w))
                else:
                    color = "#475569"
                    width = 0.0
                    opacity = 0.0
                    part_opacity = 0.0
                    
                edge_strokes.append(color)
                edge_widths.append(f"{width:.2f}")
                edge_opacities.append(f"{opacity:.2f}")
                particle_opacities.append(f"{part_opacity:.2f}")
                
            initial_stroke = edge_strokes[0]
            initial_width = edge_widths[0]
            initial_opacity = edge_opacities[0]
            initial_part_opacity = particle_opacities[0]
            flow_dur = f"{0.9 + 0.6 * np.random.rand():.2f}"
            
            svg_parts.append(
                f'    <path d="{bezier_path}" stroke="{initial_stroke}" stroke-width="{initial_width}" opacity="{initial_opacity}" marker-end="url(#arrow)" fill="none">'
                f'      <animate attributeName="stroke" calcMode="discrete" values="{";".join(edge_strokes)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'      <animate attributeName="stroke-width" calcMode="discrete" values="{";".join(edge_widths)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'      <animate attributeName="opacity" calcMode="discrete" values="{";".join(edge_opacities)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'    </path>'
                f'    <circle r="2.2" fill="{initial_stroke}" opacity="{initial_part_opacity}">'
                f'      <animate attributeName="fill" calcMode="discrete" values="{";".join(edge_strokes)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'      <animate attributeName="opacity" calcMode="discrete" values="{";".join(particle_opacities)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
                f'      <animateMotion path="{bezier_path}" dur="{flow_dur}s" repeatCount="indefinite" />'
                f'    </circle>'
            )
            
    svg_parts.append('  </g>')

    # 8. Render Nodes (Circles & Glowing Auras with Signature Colors)
    svg_parts.append('  <!-- Graph Nodes -->')
    svg_parts.append('  <g>')
    
    for i in range(NUM_NODES):
        nx, ny = node_coords[i]
        node_color = NODE_COLORS[i]
        
        node_fills = []
        node_strokes = []
        node_radii = []
        aura_radii = []
        aura_opacities = []
        
        for t, _, node_vals in frames_data:
            val = node_vals[i]
            
            if t >= int_step and i == int_node:
                # Intervened: Solid color fill, expanded glow
                fill_col = node_color
                radius = 15.5
                aura_rad = 23.5
                aura_op = 0.35
            elif val > 1.2:
                # Propagated: Solid color fill, medium glow
                fill_col = node_color
                radius = 14.0
                aura_rad = 20.0
                aura_op = 0.25
            else:
                # Normal: Dark transparent fill, thin outline, minimal glow
                fill_col = "#111524"
                radius = 12.0
                aura_rad = 16.0
                aura_op = 0.08
                
            node_fills.append(fill_col)
            node_radii.append(f"{radius:.1f}")
            aura_radii.append(f"{aura_rad:.1f}")
            aura_opacities.append(f"{aura_op:.2f}")
            
        initial_fill = node_fills[0]
        initial_radius = node_radii[0]
        initial_aura_radius = aura_radii[0]
        initial_aura_op = aura_opacities[0]
        
        svg_parts.append(
            f'    <!-- Aura glow circle -->'
            f'    <circle cx="{nx:.1f}" cy="{ny:.1f}" r="{initial_aura_radius}" fill="{node_color}" opacity="{initial_aura_op}" stroke="none">'
            f'      <animate attributeName="r" calcMode="discrete" values="{";".join(aura_radii)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'      <animate attributeName="opacity" calcMode="discrete" values="{";".join(aura_opacities)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    </circle>'
            f'    <!-- Core circle -->'
            f'    <circle cx="{nx:.1f}" cy="{ny:.1f}" r="{initial_radius}" fill="{initial_fill}" stroke="{node_color}" stroke-width="1.8">'
            f'      <animate attributeName="fill" calcMode="discrete" values="{";".join(node_fills)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'      <animate attributeName="r" calcMode="discrete" values="{";".join(node_radii)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    </circle>'
            f'    <text x="{nx:.1f}" y="{ny:.1f}" class="node-label" dy="0.5">{NODE_LABELS[i]}</text>'
        )
        
    svg_parts.append('  </g>')

    # 9. Render HUD Statistics (Updates dynamically)
    svg_parts.append('  <!-- HUD Panel -->')
    num_frames = len(eval_steps)
    for k, t in enumerate(eval_steps):
        if t < int_step:
            int_status = "None"
        else:
            int_status = f"Node {NODE_LABELS[int_node]} Active"
            
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        
        initial_display = "inline" if k == 0 else "none"
        
        svg_parts.append(
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <text x="30" y="35" fill="#cbd5e1" font-size="12" class="main-text">Step: <tspan fill="#06b6d4" font-weight="700" class="mono-text">{t}</tspan></text>'
            f'    <text x="{svg_width - 30}" y="35" fill="#cbd5e1" font-size="12" text-anchor="end" class="main-text">Intervention: <tspan fill="#ef4444" font-weight="700">{int_status}</tspan></text>'
            f'  </g>'
        )

    # Close Tag
    svg_parts.append('</svg>')

    # Ensure assets directory exists
    os.makedirs("assets", exist_ok=True)
    
    # Save SVG file
    output_path = "assets/causal-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    print(f"Successfully generated causal live SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
