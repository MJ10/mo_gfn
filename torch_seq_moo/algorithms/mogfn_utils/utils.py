import torch
import numpy as np
import itertools
from polyleven import levenshtein
import matplotlib.pyplot as plt
from botorch.utils.multi_objective import pareto
import wandb

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(levenshtein(*pair))
    return np.mean(dists)

def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in itertools.product(*spaces) 
                     if np.allclose(sum(comb), 1.0)])

def thermometer(v, n_bins=50, vmin=0, vmax=32):
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

def plot_pareto(pareto_rewards, all_rewards, pareto_only=False, objective_names=None):
    if pareto_rewards.shape[-1] < 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not pareto_only:
            ax.scatter(*np.hsplit(all_rewards, all_rewards.shape[-1]), color="grey", label="All Samples")
        ax.scatter(*np.hsplit(pareto_rewards, pareto_rewards.shape[-1]), color="red", label="Pareto Front")
        if objective_names is not None:
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
        else:
            ax.set_xlabel("Reward 1")
            ax.set_ylabel("Reward 2")
        ax.legend()
        return wandb.Image(fig)
    if pareto_rewards.shape[-1] == 3:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(
            x=all_rewards[:, 0],
            y=all_rewards[:, 1],
            z=all_rewards[:, 2],
            mode='markers',
            marker_color="grey",
            name="All Samples"
        ),
        go.Scatter3d(
            x=pareto_rewards[:, 0],
            y=pareto_rewards[:, 1],
            z=pareto_rewards[:, 2],
            mode='markers',
            marker_color="red",
            name="Pareto Front"
        )])
        if objective_names is not None:
            fig.update_layout(
                scene = dict(
                    xaxis_title=objective_names[0],
                    yaxis_title=objective_names[1],
                    zaxis_title=objective_names[2],
                )
            )
        fig.update_traces(marker=dict(size=8),
                  selector=dict(mode='markers'))
        return fig
    # if pareto_rewards.shape[-1] > 3:
    #     fig, axes = plt.subplots(pareto_rewards.shape[-1], pareto_rewards.shape[-1])
    #     for i in range(pareto_rewards.shape[-1]):
    #         for j in range(pareto_candidates.shape[-1]):
    #             if not pareto_only:
    #                 ax.scatter(all_rewards[:, i], all_rewards[j], lcolor="grey", label="All Samples")
    #             ax.scatter(*np.hsplit(pareto_rewards, pareto_rewards.shape[-1]), color="red", label="Pareto Front")
    #             ax.set_xlabel("Reward 1")
    #             ax.set_ylabel("Reward 2")
    #             ax.legend()
    #         fig.suptitle('Sharing x per column, y per row')
    #         ax1.plot(x, y)
    #         ax2.plot(x, y**2, 'tab:orange')
    #         ax3.plot(x, -y, 'tab:green')
    #         ax4.plot(x, -y**2, 'tab:red')

    #     for ax in fig.get_axes():
    #         ax.label_outer()

def pareto_frontier(solutions, rewards, maximize=True):
    pareto_mask = pareto.is_non_dominated(torch.tensor(rewards) if maximize else -torch.tensor(rewards))
    pareto_front = solutions[pareto_mask]
    pareto_rewards = rewards[pareto_mask]
    return pareto_front, pareto_rewards
