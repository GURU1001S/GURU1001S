import os
import time
import numpy as np

# Set random seed
np.random.seed(42)

# Parameters
STEPS = 600
DURATION_SEC = 4.0
NODE_LABELS = ["sensor_A", "pressure", "temperature", "RUL"]
NUM_NODES = len(NODE_LABELS)
NODE_COLORS = ["#38bdf8", "#fb7185", "#fb923c", "#34d399"]

def main():
    start_time = time.time()
    
    eval_steps = list(range(0, STEPS + 1, 20))
    num_frames = len(eval_steps)
    
    # Coordinates of 4 nodes in a horizontal chain
    # Width of SVG: 500
    cx_coords = [80, 200, 320, 440]
    cy_coord = 140
    
    svg_width = 500
    svg_height = 290
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>',
        '    .main-text { font-family: system-ui, -apple-system, sans-serif; }',
        '    .mono-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '    .node-label { font-family: system-ui, sans-serif; font-weight: 700; fill: #f8fafc; font-size: 10px; text-anchor: middle; dominant-baseline: middle; }',
        '  </style>',
        '  <defs>',
        '    <!-- Arrowhead -->',
        '    <marker id="arrow" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />',
        '    </marker>',
        '  </defs>',
        '  <!-- Background -->',
        '  <rect width="100%" height="100%" fill="#070a13" />',
        '  <text x="30" y="30" fill="#f8fafc" font-size="13" font-weight="700" class="main-text">CausalNerve — Live Streaming Observability</text>'
    ]
    
    # 1. Render Edges (Straight arrows in chain)
    svg_parts.append('  <!-- Directed Edges -->')
    for i in range(NUM_NODES - 1):
        x1, y1 = cx_coords[i], cy_coord
        x2, y2 = cx_coords[i+1], cy_coord
        
        edge_strokes = []
        edge_widths = []
        for t in eval_steps:
            if t >= 220 and i == 1:
                # Active causal flow from pressure to temperature
                color = "#fb923c"
                width = 2.5
            elif t >= 420 and i == 2:
                # Active causal flow from temperature to RUL
                color = "#34d399"
                width = 2.5
            else:
                color = "#334155"
                width = 1.2
            edge_strokes.append(color)
            edge_widths.append(f"{width:.2f}")
            
        initial_stroke = edge_strokes[0]
        initial_width = edge_widths[0]
        
        svg_parts.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{initial_stroke}" stroke-width="{initial_width}" marker-end="url(#arrow)">'
            f'    <animate attributeName="stroke" calcMode="discrete" values="{";".join(edge_strokes)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <animate attributeName="stroke-width" calcMode="discrete" values="{";".join(edge_widths)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'  </line>'
        )

    # 2. Render Nodes (Circles & Flashing Auras)
    svg_parts.append('  <!-- Telemetry Nodes -->')
    for i in range(NUM_NODES):
        nx = cx_coords[i]
        ny = cy_coord
        node_color = NODE_COLORS[i]
        
        node_fills = []
        aura_radii = []
        aura_opacities = []
        for t in eval_steps:
            if t >= 220 and i == 1:
                # Pressure intervened/anomalous (flashing)
                fill_col = node_color
                aura_rad = 18.0 + 3.0 * np.sin(t / 2.0)
                aura_op = 0.35
            elif t >= 420 and (i == 2 or i == 3):
                # Anomaly propagated downstream
                fill_col = node_color
                aura_rad = 17.0
                aura_op = 0.25
            else:
                # Normal state
                fill_col = "#111524"
                aura_rad = 13.0
                aura_op = 0.08
            node_fills.append(fill_col)
            aura_radii.append(f"{aura_rad:.1f}")
            aura_opacities.append(f"{aura_op:.2f}")
            
        initial_fill = node_fills[0]
        initial_aura_rad = aura_radii[0]
        initial_aura_op = aura_opacities[0]
        
        svg_parts.append(
            f'  <!-- Node {NODE_LABELS[i]} -->'
            f'  <circle cx="{nx}" cy="{ny}" r="{initial_aura_rad}" fill="{node_color}" opacity="{initial_aura_op}">'
            f'    <animate attributeName="r" calcMode="discrete" values="{";".join(aura_radii)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <animate attributeName="opacity" calcMode="discrete" values="{";".join(aura_opacities)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'  </circle>'
            f'  <circle cx="{nx}" cy="{ny}" r="12" fill="{initial_fill}" stroke="{node_color}" stroke-width="1.8">'
            f'    <animate attributeName="fill" calcMode="discrete" values="{";".join(node_fills)}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'  </circle>'
            f'  <text x="{nx}" y="{ny}" class="node-label">{NODE_LABELS[i][0].upper()}</text>'
            f'  <text x="{nx}" y="{ny + 22}" fill="#94a3b8" font-size="8.5" text-anchor="middle" class="main-text">{NODE_LABELS[i]}</text>'
        )

    # 3. HUD Alert Banner Card (Toggles based on simulation step)
    svg_parts.append('  <!-- Observability Alert Banner -->')
    for k, t in enumerate(eval_steps):
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        initial_display = "inline" if k == 0 else "none"
        
        if t < 220:
            status_text = "STATUS: SYSTEM OBSERVABLE (ALL NOMINAL)"
            alert_color = "#34d399" # Green
            desc_text = "Observability matrix rank = 4. No anomalies detected."
        elif t < 420:
            status_text = "ALERT: INTERVENTION DETECTED ON NODE [pressure]"
            alert_color = "#fb923c" # Orange
            desc_text = "Signal mismatch detected. Causal flow propagation active."
        else:
            status_text = "CRITICAL ALERT: PROPAGATING ANOMALY DETECTED"
            alert_color = "#f43f5e" # Red
            desc_text = "Downstream cascade active on [temperature] and [RUL] nodes."
            
        svg_parts.append(
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{DURATION_SEC}s" repeatCount="indefinite" />'
            f'    <!-- Banner Box -->'
            f'    <rect x="30" y="200" width="440" height="60" rx="5" fill="#0b0f19" stroke="{alert_color}" stroke-width="1.2" opacity="0.9" />'
            f'    <!-- Text overlay -->'
            f'    <text x="45" y="218" fill="{alert_color}" font-size="9.5" font-weight="800" class="main-text">{status_text}</text>'
            f'    <text x="45" y="233" fill="#64748b" font-size="9" class="main-text">{desc_text}</text>'
            f'    <text x="45" y="248" fill="#475569" font-size="8.5" class="main-text">Observability Epoch Cycle: {t} ms</text>'
            f'  </g>'
        )
        
    svg_parts.append('</svg>')
    
    # Save SVG file
    output_path = "assets/causalnerve-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    end_time = time.time()
    print(f"Successfully generated CausalNerve live SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"Entire script finished in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
