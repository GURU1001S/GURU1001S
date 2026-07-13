import os
import json
import time
import tracemalloc
import torch
import torch.nn as nn
import numpy as np

# Set random seed
torch.manual_seed(42)

# Models definition
class VanillaSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, N, D = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class TICSCAttention(nn.Module):
    def __init__(self, d_model, K=8):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, N, D = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        if N <= self.K:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, V)
            
        # O(N*K) sliding causal window attention using PyTorch unfold
        pad_size = self.K - 1
        zeros_pad = torch.zeros(B, pad_size, D, device=x.device)
        K_padded = torch.cat([zeros_pad, K], dim=1)
        V_padded = torch.cat([zeros_pad, V], dim=1)
        
        K_unfolded = K_padded.unfold(1, self.K, 1).transpose(-2, -1)
        V_unfolded = V_padded.unfold(1, self.K, 1).transpose(-2, -1)
        
        Q_unsqueezed = Q.unsqueeze(-2)
        scores = torch.matmul(Q_unsqueezed, K_unfolded.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V_unfolded).squeeze(-2)
        return out

def main():
    device = torch.device("cpu")
    d_model = 64
    K_val = 8
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    vanilla_model = VanillaSelfAttention(d_model).to(device)
    ticsc_model = TICSCAttention(d_model, K=K_val).to(device)
    
    vanilla_model.eval()
    ticsc_model.eval()
    
    results = {
        "seq_lengths": seq_lengths,
        "vanilla": {"time_ms": [], "memory_kb": []},
        "ticsc": {"time_ms": [], "memory_kb": []}
    }
    
    print("Running empirical scaling benchmarks (TICSC vs Vanilla Attention)...")
    
    # Warmup
    x_warm = torch.randn(1, 128, d_model, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = vanilla_model(x_warm)
            _ = ticsc_model(x_warm)
            
    for N in seq_lengths:
        x = torch.randn(1, N, d_model, device=device)
        
        # 1. Benchmark Vanilla Attention
        tracemalloc.start()
        with torch.no_grad():
            _ = vanilla_model(x)
        _, peak_vanilla = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Time measurement (average over 15 iterations)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(15):
                _ = vanilla_model(x)
            t1 = time.perf_counter()
        time_vanilla = ((t1 - t0) / 15.0) * 1000.0  # in ms
        mem_vanilla = peak_vanilla / 1024.0         # in KB
        
        # 2. Benchmark TICSC
        tracemalloc.start()
        with torch.no_grad():
            _ = ticsc_model(x)
        _, peak_ticsc = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Time measurement (average over 15 iterations)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(15):
                _ = ticsc_model(x)
            t1 = time.perf_counter()
        time_ticsc = ((t1 - t0) / 15.0) * 1000.0  # in ms
        mem_ticsc = peak_ticsc / 1024.0           # in KB
        
        results["vanilla"]["time_ms"].append(time_vanilla)
        results["vanilla"]["memory_kb"].append(mem_vanilla)
        results["ticsc"]["time_ms"].append(time_ticsc)
        results["ticsc"]["memory_kb"].append(mem_ticsc)
        
        print(f"N={N:04d} | Vanilla: {time_vanilla:.2f}ms, {mem_vanilla:.1f}KB | TICSC: {time_ticsc:.2f}ms, {mem_ticsc:.1f}KB")
        
    # Save JSON log
    os.makedirs("assets", exist_ok=True)
    with open("assets/ticsc_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved benchmark log to assets/ticsc_benchmark.json")
    
    # 3. Generate SMIL-animated SVG Chart
    svg_width = 500
    svg_height = 380
    
    # Map coordinates
    x_start = 60
    y_start = 60
    chart_w = 380
    chart_h = 240
    
    # Find scale limits
    max_time = max(max(results["vanilla"]["time_ms"]), max(results["ticsc"]["time_ms"]))
    # round max_time up to a nice integer boundary
    max_y_limit = float(np.ceil(max_time * 1.1))
    if max_y_limit == 0:
        max_y_limit = 10.0
        
    # X mapping helper (equally spaced log2 ticks)
    # i: index 0 to 5
    def get_x_coord(i):
        return x_start + (i / 5.0) * chart_w
        
    # Y mapping helper (linear time)
    def get_y_coord(t):
        return (y_start + chart_h) - (t / max_y_limit) * chart_h
        
    vanilla_points = []
    ticsc_points = []
    for idx, N in enumerate(seq_lengths):
        cx = get_x_coord(idx)
        cy_v = get_y_coord(results["vanilla"]["time_ms"][idx])
        cy_t = get_y_coord(results["ticsc"]["time_ms"][idx])
        
        vanilla_points.append(f"{cx:.1f},{cy_v:.1f}")
        ticsc_points.append(f"{cx:.1f},{cy_t:.1f}")
        
    vanilla_d = "M " + " L ".join(vanilla_points)
    ticsc_d = "M " + " L ".join(ticsc_points)
    
    duration = 4.0
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>',
        '    .main-text { font-family: system-ui, -apple-system, sans-serif; }',
        '    .mono-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '    .chart-label { font-size: 9px; fill: #64748b; }',
        '    .legend-text { font-size: 10px; font-weight: 600; }',
        '  </style>',
        '  <!-- Background -->',
        '  <rect width="100%" height="100%" fill="#070a13" />',
        '  <!-- Grid Lines -->',
        f'  <line x1="{x_start}" y1="{y_start}" x2="{x_start + chart_w}" y2="{y_start}" stroke="#1e293b" stroke-width="1" />',
        f'  <line x1="{x_start}" y1="{y_start + chart_h//2}" x2="{x_start + chart_w}" y2="{y_start + chart_h//2}" stroke="#1e293b" stroke-dasharray="2 4" stroke-width="1" />',
        f'  <line x1="{x_start}" y1="{y_start + chart_h}" x2="{x_start + chart_w}" y2="{y_start + chart_h}" stroke="#1e293b" stroke-width="1" />',
        f'  <line x1="{x_start}" y1="{y_start}" x2="{x_start}" y2="{y_start + chart_h}" stroke="#1e293b" stroke-width="1" />',
        f'  <line x1="{x_start + chart_w}" y1="{y_start}" x2="{x_start + chart_w}" y2="{y_start + chart_h}" stroke="#1e293b" stroke-width="1" />',
        '  <!-- Legend -->'
    ]
    
    # Legend
    svg_parts.append(f'  <g transform="translate(60, 48)">')
    svg_parts.append(f'    <rect width="8" height="8" rx="2" fill="#ef4444" />')
    svg_parts.append(f'    <text x="14" y="8" fill="#ef4444" class="legend-text main-text">Vanilla Attention O(N²)</text>')
    svg_parts.append(f'    <rect x="180" width="8" height="8" rx="2" fill="#06b6d4" />')
    svg_parts.append(f'    <text x="194" y="8" fill="#06b6d4" class="legend-text main-text">TICSC Attention O(N·K)</text>')
    svg_parts.append(f'  </g>')
    
    # Y-Axis Labels
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + 4}" class="chart-label main-text" text-anchor="end">{max_y_limit:.1f} ms</text>')
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + chart_h//2 + 4}" class="chart-label main-text" text-anchor="end">{max_y_limit/2:.1f} ms</text>')
    svg_parts.append(f'  <text x="{x_start - 8}" y="{y_start + chart_h + 4}" class="chart-label main-text" text-anchor="end">0.0 ms</text>')
    
    # Rotated Y-Axis Label
    svg_parts.append(f'  <text x="{x_start - 35}" y="{y_start + chart_h//2}" fill="#94a3b8" font-size="10" font-weight="600" text-anchor="middle" transform="rotate(-90 {x_start - 35} {y_start + chart_h//2})" class="main-text">Execution Time (ms)</text>')
    
    # X-Axis Labels
    for idx, N in enumerate(seq_lengths):
        cx = get_x_coord(idx)
        svg_parts.append(f'  <text x="{cx:.1f}" y="{y_start + chart_h + 15}" class="chart-label main-text" text-anchor="middle">{N}</text>')
        if idx > 0 and idx < 5:
            # dashed vertical grid lines
            svg_parts.append(f'  <line x1="{cx:.1f}" y1="{y_start}" x2="{cx:.1f}" y2="{y_start + chart_h}" stroke="#1e293b" stroke-dasharray="2 4" stroke-width="1" />')
            
    svg_parts.append(f'  <text x="{x_start + chart_w//2}" y="{y_start + chart_h + 28}" fill="#94a3b8" font-size="10" text-anchor="middle" font-weight="600" class="main-text">Sequence Length (N)</text>')

    # Curves with self-drawing SMIL animation
    svg_parts.append('  <!-- Curves -->')
    svg_parts.append(
        f'  <path d="{vanilla_d}" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="1000" stroke-dashoffset="1000" stroke-linecap="round">'
        f'    <animate attributeName="stroke-dashoffset" values="1000;0" dur="{duration}s" repeatCount="indefinite" />'
        f'  </path>'
        f'  <path d="{ticsc_d}" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-dasharray="1000" stroke-dashoffset="1000" stroke-linecap="round">'
        f'    <animate attributeName="stroke-dashoffset" values="1000;0" dur="{duration}s" repeatCount="indefinite" />'
        f'  </path>'
    )
    
    # Dots representing the data points (drawn dynamically)
    svg_parts.append('  <!-- Data Point Markers -->')
    for idx, N in enumerate(seq_lengths):
        cx = get_x_coord(idx)
        cy_v = get_y_coord(results["vanilla"]["time_ms"][idx])
        cy_t = get_y_coord(results["ticsc"]["time_ms"][idx])
        
        # Calculate dynamic reveal timing for markers
        # Sweep passes x at ratio (idx / 5.0) of duration
        start_ratio = idx / 5.0
        # Marker starts invisible, fades in at the exact moment the sweep line passes
        op_values = ["0.0"] * 10
        for m in range(10):
            ratio_m = m / 9.0
            if ratio_m >= start_ratio:
                op_values[m] = "1.0"
        op_str = ";".join(op_values)
        
        svg_parts.append(
            f'  <circle cx="{cx:.1f}" cy="{cy_v:.1f}" r="4" fill="#ef4444" stroke="#070a13" stroke-width="1" opacity="0.0">'
            f'    <animate attributeName="opacity" calcMode="discrete" values="{op_str}" dur="{duration}s" repeatCount="indefinite" />'
            f'  </circle>'
            f'  <circle cx="{cx:.1f}" cy="{cy_t:.1f}" r="4" fill="#06b6d4" stroke="#070a13" stroke-width="1" opacity="0.0">'
            f'    <animate attributeName="opacity" calcMode="discrete" values="{op_str}" dur="{duration}s" repeatCount="indefinite" />'
            f'  </circle>'
        )

    # Sweeping scanline
    svg_parts.append('  <!-- Sweep Scanline -->')
    svg_parts.append(
        f'  <line x1="{x_start}" y1="{y_start - 10}" x2="{x_start}" y2="{y_start + chart_h + 5}" stroke="#f8fafc" stroke-width="1.2" opacity="0.75">'
        f'    <animateTransform attributeName="transform" type="translate" values="0,0; {chart_w},0" dur="{duration}s" repeatCount="indefinite" />'
        f'  </line>'
    )

    # Dynamic HUD text box (displays stats depending on scanline position)
    svg_parts.append('  <!-- Dynamic Stat HUD -->')
    num_frames = len(seq_lengths)
    for k, N in enumerate(seq_lengths):
        disp_vals = ["none"] * num_frames
        disp_vals[k] = "inline"
        values_str_for_display = ";".join(disp_vals)
        
        initial_display = "inline" if k == 0 else "none"
        
        time_v = results["vanilla"]["time_ms"][k]
        mem_v = results["vanilla"]["memory_kb"][k]
        time_t = results["ticsc"]["time_ms"][k]
        mem_t = results["ticsc"]["memory_kb"][k]
        
        # Format strings
        hud_text = (
            f'  <g display="{initial_display}">'
            f'    <animate attributeName="display" calcMode="discrete" values="{values_str_for_display}" dur="{duration}s" repeatCount="indefinite" />'
            # Background panel for HUD
            f'    <rect x="250" y="80" width="180" height="75" rx="5" fill="#0b0f19" stroke="#334155" stroke-width="1" opacity="0.9" />'
            # HUD stats text
            f'    <text x="262" y="98" fill="#94a3b8" font-size="10.5" font-weight="700" class="main-text">SEQUENCE N = {N}</text>'
            f'    <text x="262" y="118" fill="#ef4444" font-size="9.5" class="main-text">O(N²): <tspan class="mono-text">{time_v:.2f}ms</tspan> | <tspan class="mono-text">{mem_v:.1f}KB</tspan></text>'
            f'    <text x="262" y="136" fill="#06b6d4" font-size="9.5" class="main-text">O(N·K): <tspan class="mono-text">{time_t:.2f}ms</tspan> | <tspan class="mono-text">{mem_t:.1f}KB</tspan></text>'
            f'  </g>'
        )
        svg_parts.append(hud_text)

    # Document title
    svg_parts.append(f'  <text x="30" y="30" fill="#f8fafc" font-size="13" font-weight="700" class="main-text">Attention vs TICSC Complexity Scaling</text>')

    # Close SVG tag
    svg_parts.append('</svg>')
    
    # Save SVG file
    output_path = "assets/ticsc-live.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
        
    print(f"Successfully generated TICSC benchmark animated SVG at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
