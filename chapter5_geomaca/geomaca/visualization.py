"""
GeoMACA Visualization

Functions to visualize credit assignment for interpretability and paper figures.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import os


def visualize_credit_assignment(
    geomaca,
    trajectory: Dict[str, torch.Tensor],
    final_reward: torch.Tensor,
    save_path: str = 'credit_viz.pdf',
    agent_names: Optional[List[str]] = None,
    show_plot: bool = True
) -> Optional[object]:
    """
    Generate credit visualization figure for paper.
    
    Creates a 2x2 figure showing:
    (a) Geographic attention heatmap (3x3 grid for 9 zones)
    (b) Temporal attention bar chart
    (c) Agent attention pie chart or bar
    (d) Top 10 credit cells
    
    Args:
        geomaca: GeoMACA model instance
        trajectory: Trajectory data dict
        final_reward: Terminal reward value
        save_path: Path to save figure
        agent_names: Optional custom agent names
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib figure object (if matplotlib available)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping visualization.")
        return None
    
    # Get credit assignment
    with torch.no_grad():
        credit, (alpha_temp, alpha_geo, alpha_agent) = geomaca(trajectory, final_reward)
    
    # Convert to numpy
    alpha_temp_np = alpha_temp[0].cpu().numpy()
    alpha_geo_np = alpha_geo[0].cpu().numpy()
    alpha_agent_np = alpha_agent[0].cpu().numpy()
    credit_np = credit[0].cpu().numpy()
    
    if agent_names is None:
        agent_names = ['Economy', 'Environment', 'Society']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Geographic attention heatmap
    ax = axes[0, 0]
    num_zones = len(alpha_geo_np)
    grid_size = int(np.sqrt(num_zones))
    
    if grid_size * grid_size == num_zones:
        geo_heatmap = alpha_geo_np.reshape(grid_size, grid_size)
    else:
        # Non-square grid - reshape as best as possible
        geo_heatmap = alpha_geo_np.reshape(-1, 1) if num_zones < 4 else alpha_geo_np.reshape(3, 3)
    
    im = ax.imshow(geo_heatmap, cmap='Reds', vmin=0, vmax=alpha_geo_np.max())
    ax.set_title('(a) Geographic Credit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Zone Column')
    ax.set_ylabel('Zone Row')
    
    # Add text annotations
    for i in range(geo_heatmap.shape[0]):
        for j in range(geo_heatmap.shape[1]):
            val = geo_heatmap[i, j]
            color = 'white' if val > geo_heatmap.max() * 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Credit')
    
    # (b) Temporal attention
    ax = axes[0, 1]
    num_turns = len(alpha_temp_np)
    bars = ax.bar(range(num_turns), alpha_temp_np, color='steelblue', edgecolor='navy')
    
    # Highlight max
    max_idx = np.argmax(alpha_temp_np)
    bars[max_idx].set_color('#e74c3c')
    bars[max_idx].set_edgecolor('darkred')
    
    ax.set_title('(b) Temporal Credit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Turn')
    ax.set_ylabel('Credit Weight')
    ax.set_xticks(range(num_turns))
    ax.set_xticklabels([f'T{i+1}' for i in range(num_turns)])
    ax.set_ylim(0, alpha_temp_np.max() * 1.2)
    
    # Add value labels
    for i, v in enumerate(alpha_temp_np):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    # (c) Agent attention
    ax = axes[1, 0]
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    bars = ax.bar(agent_names[:len(alpha_agent_np)], alpha_agent_np, color=colors[:len(alpha_agent_np)], 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_title('(c) Agent Credit', fontsize=14, fontweight='bold')
    ax.set_ylabel('Credit Weight')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(alpha_agent_np):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    # (d) Top credit cells
    ax = axes[1, 1]
    credit_flat = credit_np.flatten()
    top_k = min(10, len(credit_flat))
    top_indices = np.argsort(credit_flat)[-top_k:][::-1]
    
    num_zones = credit_np.shape[1]
    num_agents = credit_np.shape[2]
    
    top_labels = []
    top_values = []
    top_colors = []
    
    for idx in top_indices:
        t = idx // (num_zones * num_agents)
        z = (idx % (num_zones * num_agents)) // num_agents
        a = idx % num_agents
        
        label = f'T{t+1}-Z{z+1}-{agent_names[a][:3]}'
        top_labels.append(label)
        top_values.append(credit_flat[idx])
        top_colors.append(colors[a] if a < len(colors) else 'gray')
    
    y_pos = range(len(top_labels))
    ax.barh(y_pos, top_values, color=top_colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels)
    ax.set_title('(d) Top 10 Credit Cells', fontsize=14, fontweight='bold')
    ax.set_xlabel('Credit Weight')
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_values):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Credit visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    caption: str = "GeoMACA Ablation Study",
    label: str = "tab:geomaca_ablation"
) -> str:
    """
    Generate LaTeX table from ablation results.
    
    Args:
        results: Dict of {method_name: {metric: value}}
        caption: Table caption
        label: Table label for referencing
    
    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Ep. to Conv.} & \\textbf{Profit} & \\textbf{Seg.} \\\\")
    lines.append("\\midrule")
    
    for name, metrics in results.items():
        profit = metrics.get('profit_mean', 0)
        seg = metrics.get('segregation_mean', 0)
        conv = metrics.get('convergence_mean', 0)
        
        profit_str = f"\\${profit/1e6:.0f}M" if profit > 0 else f"\\${profit/1e6:.1f}M"
        seg_str = f"{seg:.3f}"
        conv_str = f"{conv:.0f}"
        
        # Bold best results
        if name == 'Full GeoMACA':
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{conv_str}}} & \\textbf{{{profit_str}}} & \\textbf{{{seg_str}}} \\\\")
        else:
            lines.append(f"{name} & {conv_str} & {profit_str} & {seg_str} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def plot_learning_curves(
    curves: Dict[str, List[np.ndarray]],
    metric_name: str = 'Profit',
    save_path: str = 'learning_curves.pdf',
    show_plot: bool = True
) -> Optional[object]:
    """
    Plot learning curves for different methods.
    
    Args:
        curves: Dict of {method_name: [array of curves per seed]}
        metric_name: Name of the metric being plotted
        save_path: Path to save figure
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping plot.")
        return None
    
    colors = {
        'Full GeoMACA': '#2ecc71',
        'STAS-style (No Geographic)': '#e74c3c',
        'No Geographic': '#e74c3c',
        'No Temporal': '#f39c12',
        'No Agent': '#9b59b6',
        'Uniform Credit': '#95a5a6',
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, seed_curves in curves.items():
        if not seed_curves:
            continue
        
        curves_array = np.array(seed_curves)
        mean_curve = curves_array.mean(axis=0)
        std_curve = curves_array.std(axis=0)
        
        color = colors.get(name, 'blue')
        x = range(len(mean_curve))
        
        ax.plot(x, mean_curve, label=name, color=color, linewidth=2)
        ax.fill_between(
            x, mean_curve - std_curve, mean_curve + std_curve,
            alpha=0.2, color=color
        )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Learning Curves: Credit Assignment Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Learning curves saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_credit_heatmap_over_episode(
    geomaca,
    trajectories: List[Dict[str, torch.Tensor]],
    final_rewards: List[torch.Tensor],
    save_path: str = 'credit_heatmap_episode.pdf',
    show_plot: bool = True
) -> Optional[object]:
    """
    Create a heatmap showing how credit assignment evolves during training.
    
    Args:
        geomaca: GeoMACA model
        trajectories: List of trajectories at different training episodes
        final_rewards: Corresponding final rewards
        save_path: Path to save figure
        show_plot: Whether to display
    
    Returns:
        matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed.")
        return None
    
    num_episodes = len(trajectories)
    
    # Collect agent credit over episodes
    agent_credits = []
    
    with torch.no_grad():
        for traj, reward in zip(trajectories, final_rewards):
            credit, (_, _, alpha_agent) = geomaca(traj, reward)
            agent_credits.append(alpha_agent[0].cpu().numpy())
    
    agent_credits = np.array(agent_credits)  # [num_episodes, num_agents]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agent_names = ['Economy', 'Environment', 'Society']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (name, color) in enumerate(zip(agent_names, colors)):
        if i < agent_credits.shape[1]:
            ax.plot(range(num_episodes), agent_credits[:, i], label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Agent Credit', fontsize=12)
    ax.set_title('Agent Credit Evolution During Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

