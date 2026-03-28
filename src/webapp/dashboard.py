import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx

from src.webapp.misc import set_matplotlib_font

def draw_dashboard_to_st(sim_data, time_labels, time_unit):
    set_matplotlib_font()
    fig = plt.figure(figsize=(16, 5.5), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#F8F9FA')

    x_vals = [0] + time_labels
    y_exp = [0] + sim_data['exposure']
    y_int = [0] + sim_data['interaction']

    ax1.plot(x_vals, y_exp, 'o-', color='#00B4D8', linewidth=3, markersize=8, label='触达曝光总人数')
    ax1.plot(x_vals, y_int, 's-', color='#E63946', linewidth=3.5, markersize=8, label='实质互动(含评赞)')

    theoretical = [0] + [sim_data['num_agents'] * sim_data['prob'] * (1 - math.exp(-0.4 * i)) for i in range(1, 6)]
    ax1.plot(x_vals, theoretical, '--', color='#FFB703', alpha=0.9, linewidth=2.5, label='回归预测累积曲线')

    ax1.set_title(f"舆论场时间动力学 (T={sim_data['time_span']} {time_unit})", fontsize=16, fontweight='bold',
                  color='#333333', pad=15)
    ax1.set_xlabel(f"演化时间 ({time_unit})", fontsize=13, color='#333333')
    ax1.set_ylabel("受众数量 (人)", fontsize=13, color='#333333')
    ax1.set_xticks(x_vals)
    ax1.tick_params(labelsize=11, colors='#333333')
    ax1.legend(fontsize=12, facecolor='white', edgecolor='#CCCCCC', labelcolor='#333333')
    ax1.grid(color='#CCCCCC', linestyle=':', alpha=0.8)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#F8F9FA')
    G = nx.DiGraph()
    G.add_edges_from(sim_data['edges'])
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=0.8, seed=42)
        nx.draw(G, pos, ax=ax2, node_size=60, node_color='#FFB703', edge_color='#999999', arrowsize=10, alpha=0.8)
    ax2.set_title("互动社交裂变图谱", fontsize=16, fontweight='bold', color='#333333', pad=15)

    plt.tight_layout()
    return fig
