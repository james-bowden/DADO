import os
from tqdm import tqdm
import sys
import argparse
import pickle
import math
from time import time
from copy import deepcopy
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
from glob import glob

# Silence specific Orbax sharding warning
warnings.filterwarnings("ignore", message=".*Sharding info not provided when restoring.*")

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Reduce XLA compilation time/warnings
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"

# Check if working directory ends with 'DADO', then append to sys.path
cwd = os.getcwd()
if cwd.endswith('DADO'):
    sys.path.append(cwd)
else:
    raise ValueError(f"Working directory {cwd} does not end with 'DADO'")

# Third-party imports
import numpy as np
from scipy.stats import spearmanr
import networkx as nx
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate as tb
import mlxu

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Project imports
from src.problems.real.oracle import OracleObjective
from src.problems.synthetic.graph.base import SyntheticGraphObjective
from src.decomposition.graphs import GeneralizedGraph, JunctionTree
from src.decomposition.fgm.mlp.base import MLPGeneralizedFunctionalGraph, MLPFunctionalGraph
from src.problems.real.protein_structure import residue_contact_map
from src.opt.model_based.EDA.DADO.generalized import GeneralizedDADO
from src.opt.model_based.EDA.joint_EDA import JointEDA
from src.utils import count_params, jaxrng_to_int

STRUCTURE_PATHS_DICT = {
    "amyloid": "fold_amyloid_model_0.cif",
    "aav": "fold_aav_model_0.cif",
    "gb1_55": "fold_gb1_model_0.cif", 
    "tdp43": "fold_tdp43_model_0.cif",
    "gfp": "fold_avgfp_model_0.cif",
    "ynzc": "fold_ynzc_bacsu_tsuboyama_2023_2jvd_2025_09_18_01_16_model_0.cif",
    "gcn4": "fold_gcn4_model_0.cif",
    "his7": "fold_his7_model_0.cif",
    "phot": "fold_phot_chlre_model_0.cif",
}


def fit_oracle(obj_name, num_epochs, num_inner_iter, batch_size, learning_rate, seed, output_dir, vis=True, fit=True):
    """Fit oracle objectives and save plots."""
    if vis and fit:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes = axes.flatten()

    obj = OracleObjective(obj_name=obj_name)
    T = obj.values.size
    while batch_size > T:
        batch_size //= 2
    if fit:
        losses = obj.train(
            num_epochs=num_epochs,
            num_inner_iter=num_inner_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=True,
        )
        print(f"Oracle-{obj_name} #params: {count_params(obj.model)}")
        if vis:
            axes[0].plot(losses)
            axes[0].set_yscale('log')
            axes[0].set_ylabel('MSE')
            axes[0].set_xlabel('Iter')
            axes[0].set_title(f'Loss History - Oracle-{obj.obj_name} (L={len(obj.D)}, {T:.1e} data)')
            axes[1].scatter(obj.values, obj.query_all(), alpha=0.05)
            axes[1].set_xlabel('True Value')
            axes[1].set_ylabel('Predicted Value')
            preds = obj.query_all()
            pr = jnp.corrcoef(obj.values, preds)[0, 1]
            sr = spearmanr(obj.values, preds, axis=None).statistic
            axes[1].set_title(f'Real Data Corr: {pr:.2f}P / {sr:.2f}S')
    
    if vis and fit:
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4)
        plt.savefig(output_dir / 'oracle_training.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return obj, obj.query_all() if fit else None


def fit_reward_AF_FGM(
        obj, offline_inds, structure_path, 
        hidden_dims, batch_size, gr_batch_size, embedding_dim,
        num_epochs, num_inner_iter, learning_rate, seed, output_dir,
        vis=True, fit=True, save_load=True):
    """Fit AF-based FGM reward models and save plots."""
    losses = None
    if vis and save_load:
        N_plots = 5
        fig, axes = plt.subplots(2, 3, figsize=(4*3, 3*2))
        axes = axes.flatten()

    checkpointer_model = ocp.StandardCheckpointer()
    checkpointer_misc = ocp.PyTreeCheckpointer()

    # NOTE: assuming graph is always the same...
    fname = f"{os.getcwd()}/checkpoints/fgm/{obj.obj_name}/"
    fname = os.path.join(fname, f"arc_{'-'.join(map(str, hidden_dims))}_epochs_{num_epochs}_inner_{num_inner_iter}_lr_{learning_rate}_batch_{batch_size}_grbatch_{gr_batch_size}_embdim_{embedding_dim}_noffline_{'all' if offline_inds is None else len(offline_inds)}_seed_{seed}.nnx")
    edges_AF, _ = residue_contact_map(
        structure_path, 
        obj.WT_seq, # needs to be contiguous substring of the seq in the CIF file
        edgelist=True,
        verbose=False,
        active_inds=obj.active_inds,
        threshold=9.,
        )
    gg = GeneralizedGraph(len(obj.D), edges_AF, verbose=False)
    fg = MLPGeneralizedFunctionalGraph(
        gg, n_states_list=obj.D, 
        hidden_dims=hidden_dims, 
        batch_size=batch_size,
        gr_batch_size=gr_batch_size,
        embedding_dim=embedding_dim,
        rngs=jax.random.PRNGKey(seed),
    )
    if fit:
        if os.path.exists(fname) and save_load:
            graphdef, state = nnx.split(fg.model)
            state = checkpointer_model.restore(fname, state)
            fg.model = nnx.merge(graphdef, state)
            loaded_data = checkpointer_misc.restore(fname.replace(".nnx", ".dict"))
            losses = loaded_data["losses"]
            # print(f"Successfully loaded model from {fname}")
        else:
            fg.model.train()
            losses = MLPGeneralizedFunctionalGraph.fit(
                fg,
                obj.genotypes if offline_inds is None else obj.genotypes[offline_inds, :],
                obj.values if offline_inds is None else obj.values[offline_inds], 
                n_epochs=num_epochs,
                n_iter_inner=num_inner_iter,
                learning_rate=learning_rate,
                seed=seed,
            )
            if save_load:
                graphdef, state = nnx.split(fg.model)
                try: 
                    checkpointer_model.save(fname, state)
                    checkpointer_misc.save(fname.replace(".nnx", ".dict"), {"losses": losses})
                    print(f"Successfully saved model (and losses) to {fname}")
                except Exception as e:
                    # NOTE: oracles are common, so another process may have saved in meantime. 
                    state = checkpointer_model.restore(fname, state)
                    fg.model = nnx.merge(graphdef, state)
                    loaded_data = checkpointer_misc.restore(fname.replace(".nnx", ".dict"))
                    losses = loaded_data["losses"]
                    # print(f"Training intercepted. Threw out and successfully loaded model from {fname}.")
    # print(f"AF-FGM-{obj.obj_name} #params: {count_params(fg.model)}")
    fg.eval()
    fgm = SyntheticGraphObjective(
        D=obj.D,
        fg=MLPFunctionalGraph._split_merge(fg), # NOTE: important, to sever tracing
        weight_fn_name='id', # NOTE: already applied fg_pos_scale in fg.fit
    )
    fgm.obj_name = obj.obj_name
    if vis and save_load: # don't do if oracle.
        nx.draw_networkx(
            gg.graph, 
            ax=axes[0], 
            with_labels=True,
            node_size = 50,
            font_size=3,
        )
        og_nodes, og_edges = gg.n_nodes, gg.n_edges
        sg_nodes, sg_edges = sum(sg.n_nodes for sg in gg.subgraphs), sum(sg.n_edges for sg in gg.subgraphs)
        axes[0].set_title(f'AF Graph - {obj.obj_name} [N={og_nodes}, E={og_edges}]')
        axes[0].set_axis_off()
        node_sizes, clique_sizes = [], [] # cliques are basically edge sizes
        for sg in gg.subgraphs:
            colors = [
                "red" if node == sg.root else "blue" 
                for node in sg.graph.nodes()
            ]
            nx.draw_networkx(
                sg.graph, 
                pos=nx.nx_agraph.graphviz_layout(sg.graph, prog="twopi"), 
                ax=axes[1], 
                with_labels=True,
                node_size=100,
                font_size=5,
                node_color=colors
            )
            # assuming junction trees, which isn't actually correct.
            node_sizes += sg.clique_sizes
            clique_sizes += sg.clique_parent_sizes
        axes[1].set_title(f'AF Clique Graph - {obj.obj_name} [N={sg_nodes}, E={sg_edges}]')
        axes[1].set_axis_off()
        indices = np.arange(len(node_sizes))
        # node sizes in blue
        axes[2].bar(indices, node_sizes, color='blue', label='Node size')
        # clique "extra" in yellow stacked on top
        axes[2].bar(indices, np.array(clique_sizes) - np.array(node_sizes),
                    bottom=node_sizes, color='green', label='Clique size')
        axes[2].set_title(f'Sizes - {obj.obj_name}')
        axes[2].set_xlabel('Node Index')
        axes[2].set_ylabel('Size')
        axes[2].legend()
        if fit:
            axes[3].plot(losses)
            axes[3].set_yscale('log')
            axes[3].set_ylabel('MSE')
            axes[3].set_xlabel('Iter')
            axes[3].set_title(f'Loss History - Oracle-{obj.obj_name} (L={len(obj.D)})')
            # on the full real dataset
            y_pred = fg.evaluate_batched(obj.genotypes)
            y_true = obj.values
            # print(f"Test MSE: {jnp.mean((y_pred - y_true)**2):.2e}")
            axes[4].scatter(y_true, y_pred, alpha=0.05)
            axes[4].set_xlabel('True Value')
            axes[4].set_ylabel('Predicted Value')
            pr = jnp.corrcoef(y_true, y_pred)[0, 1]
            sr = spearmanr(y_true, y_pred, axis=None).statistic
            axes[4].set_title(f'Real Data Corr: {pr:.2f}P / {sr:.2f}S')
    
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4)
        plt.savefig(output_dir / 'af_fgm_training.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return fgm, losses

def init_worker(gpu_queue, allowed_gpus):
    """Initialize worker process for multiprocessing pool."""
    global _worker_gpu
    _worker_gpu = gpu_queue.get()  # GPU ID from allowed list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu)

def run_experiment_wrapper(params):
    """Wrapper function for multiprocessing."""
    (temp_Q, lr, fgm_params, prior_data, seeds, n_epochs, n_inner_iter, n_mc_samples, 
     n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir) = params
    
    # Reconstruct the objective and FGM model in the worker process
    from pathlib import Path
    import os
    
    obj = OracleObjective(obj_name=fgm_params['obj_name']) # NOTE: doesn't have model loaded in!
    
    # Load the FGM model from checkpoints (it should already be trained)
    reward_model_fgm, _ = fit_reward_AF_FGM(
        obj, fgm_params['offline_inds'], 
        fgm_params['structure_path'],
        fgm_params['hidden_dims_fgm'], fgm_params['batch_size_fgm'], 
        fgm_params['gr_batch_size_fgm'], fgm_params['embedding_dim_fgm'],
        fgm_params['n_epochs_fgm'], fgm_params['n_inner_iter_fgm'], 
        fgm_params['lr_fgm'], fgm_params['seed'], Path(outdir),
        vis=False, fit=True  # Don't fit, just load from checkpoint
    )
    
    # Use the FGM model as the objective for EDA training, keep oracle for evaluation
    return run_experiment(temp_Q, lr, reward_model_fgm, prior_data, seeds, n_epochs, n_inner_iter, 
                         n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir)

def run_joint_experiment(temp_Q, lr, obj, prior_data, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir):
    """Run only joint EDA experiment for given parameters and seed."""
    intermediate_file = f"{outdir}/results/joint_Q{temp_Q}-lr{lr}_s{seed}.dict"
    if os.path.exists(intermediate_file):
        return mlxu.load_pickle(intermediate_file)
    
    joint_eda = JointEDA(
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=n_epochs,
        num_inner_iter=n_inner_iter,
        num_MC_samples=n_mc_samples,
        learning_rate=lr,
        num_layers=n_layers,
        dim_ff=dim_ff,
        num_heads=n_heads,
        dim_attn=16*n_heads,
        use_transformer=use_transformer,
        use_prior=use_prior,
        prior_data=prior_data,
        temp_Q=temp_Q,
        replay_buffer=False,
    )
    d_joint = joint_eda.precompute()
    
    stats = {
        'max': d_joint['values'].max(axis=-1),
        'min': d_joint['values'].min(axis=-1),
        'ci_lower': jnp.quantile(d_joint['values'], 0.025, axis=-1),
        'ci_upper': jnp.quantile(d_joint['values'], 0.975, axis=-1),
        'values': d_joint['values'],
        'weights': d_joint['weights'],
        'mean': calculate_epoch_stats(d_joint['values'])[0],
        'std': calculate_epoch_stats(d_joint['values'])[1],
        'ess': d_joint['ess'],
        'mode': joint_eda.get_mode(),
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed,
        'method': 'joint',
        'samples': d_joint['samples']
    }
    mlxu.save_pickle(stats, intermediate_file)
    return stats

def run_fd_experiment(temp_Q, lr, obj, prior_data, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir):
    """Run only FD EDA experiment for given parameters and seed."""
    intermediate_file = f"{outdir}/results/fd_Q{temp_Q}-lr{lr}_s{seed}.dict"
    if os.path.exists(intermediate_file):
        return mlxu.load_pickle(intermediate_file)
        
    fgo = GeneralizedDADO(
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=n_epochs,
        num_inner_iter=n_inner_iter,
        num_MC_samples=n_mc_samples,
        learning_rate=lr,
        num_layers=n_layers,
        dim_ff=dim_ff,
        num_heads=n_heads,
        dim_attn=16*n_heads,
        use_transformer=use_transformer,
        verbose=False,
        use_prior=use_prior,
        prior_data=prior_data,
        temp_Q=temp_Q,
        replay_buffer=False,
    )
    d_fd = fgo.precompute()
    
    stats = {
        'max': d_fd['values'].max(axis=-1),
        'min': d_fd['values'].min(axis=-1),
        'ci_lower': jnp.quantile(d_fd['values'], 0.025, axis=-1),
        'ci_upper': jnp.quantile(d_fd['values'], 0.975, axis=-1),
        'values': d_fd['values'],
        'weights': d_fd['weights'],
        'mean': calculate_epoch_stats(d_fd['values'])[0],
        'std': calculate_epoch_stats(d_fd['values'])[1],
        'ess': d_fd['ess'],
        'mode': fgo.get_mode(),
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed,
        'method': 'fd', 
        'samples': d_fd['samples']
    }
    mlxu.save_pickle(stats, intermediate_file)
    return stats

def method_specific_wrapper(params):
    """Wrapper for method-specific experiment functions."""
    (temp_Q, lr, method, fgm_params, prior_data, seed, n_epochs, n_inner_iter, n_mc_samples, 
     n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir) = params
    
    # Reconstruct the objective and FGM model in the worker process
    from pathlib import Path
    import os

    obj = OracleObjective(obj_name=fgm_params['obj_name'])
    
    # Load the FGM model from checkpoints (it should already be trained)
    reward_model_fgm, _ = fit_reward_AF_FGM(
        obj, fgm_params['offline_inds'], 
        fgm_params['structure_path'],
        fgm_params['hidden_dims_fgm'], fgm_params['batch_size_fgm'], 
        fgm_params['gr_batch_size_fgm'], fgm_params['embedding_dim_fgm'],
        fgm_params['n_epochs_fgm'], fgm_params['n_inner_iter_fgm'], 
        fgm_params['lr_fgm'], fgm_params['seed'], Path(outdir),
        vis=False, fit=True  # Don't fit, just load from checkpoint. But need fit to be True.
    )
    
    if method == 'joint':
        return run_joint_experiment(temp_Q, lr, reward_model_fgm, prior_data, seed, n_epochs, n_inner_iter, 
                                  n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, 
                                  use_prior, outdir)
    elif method == 'fd':
        return run_fd_experiment(temp_Q, lr, reward_model_fgm, prior_data, seed, n_epochs, n_inner_iter, 
                               n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, 
                               use_prior, outdir)
    else:
        raise ValueError(f"Unknown method: {method}")

def aggregate_method_results(method, temp_Q, lr, sweep_seeds, additional_seeds, outdir):
    """Aggregate results from sweep + additional replicates for a specific method and hyperparameters."""
    # all_seeds = sweep_seeds + additional_seeds
    # only use additioanl seeds
    
    maxes, mins, weights_list = [], [], []
    means, stds, ess_list = [], [], []
    values, ci_lower, ci_upper = [], [], []
    modes = []
    print(f"Globbing {outdir}/results/{method}_Q{'*'}-lr{'*'}_s*.dict")
    fnames = glob(f"{outdir}/results/{method}_Q{'*'}-lr{'*'}_s*.dict")
    print(f"Found {len(fnames)} {method} files")
    n_reps = len(additional_seeds)
    n_params = len(fnames) // n_reps
    assert len(fnames) % n_reps == 0, f"{len(fnames)} files for {n_reps} doesnt' divide evenly"
    Qtemp, ind = np.inf, 0
    for i in range(n_params):
        t = fnames[i*n_reps].split('_Q')[-1].split('-lr')[0]
        if float(t) < float(Qtemp):
            Qtemp = t
            ind = i*n_reps
    assert float(Qtemp) < np.inf, f"Didn't find any temp less than inf"
    print(f"Selected {Qtemp}")
    fnames = fnames[ind:ind+n_reps]
    assert len(fnames) == n_reps
    
    for fname in fnames:
        if os.path.exists(fname):
            stats = mlxu.load_pickle(fname)
            maxes.append(stats['max'])
            mins.append(stats['min'])
            weights_list.append(stats['weights'])
            means.append(stats['mean'])
            stds.append(stats['std'])
            ess_list.append(stats['ess'])
            modes.append(stats['mode'])
            values.append(stats['values'])
            ci_lower.append(stats['ci_lower'])
            ci_upper.append(stats['ci_upper'])
    
    # Stack results
    maxes_stacked = np.stack(maxes, axis=0)
    mins_stacked = np.stack(mins, axis=0)
    weights_stacked = np.stack(weights_list, axis=0)
    means_stacked = np.stack(means, axis=0)
    stds_stacked = np.stack(stds, axis=0)
    values_stacked = np.stack(values, axis=0)
    ci_lower_stacked = np.stack(ci_lower, axis=0)
    ci_upper_stacked = np.stack(ci_upper, axis=0)

    return {
        'maxes': maxes_stacked,
        'mins': mins_stacked,
        'weights': weights_stacked,
        'means': means_stacked,
        'means_mean': means_stacked.mean(axis=0),
        'means_std': means_stacked.std(axis=0),
        'stds_mean': stds_stacked.mean(axis=0),
        'ess': ess_list,
        'mode': jnp.mean(jnp.array([m[-1] for m in modes])) if modes else None,
        'values': values_stacked,
        'ci_lower': ci_lower_stacked,
        'ci_upper': ci_upper_stacked,
        'selection': means_stacked.mean(axis=0)[-1]  # Final epoch mean
    }

def train_experiments(obj, num_epochs, num_inner_iter, mc_samples, num_layers, dim_ff, lr, use_prior, prior_data, n_heads, use_transformer, temp_Q, seed):
    """Initialize and train EDA models for real-world objective."""
    results = []
    L = len(obj.D)

    joint_eda = JointEDA(
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=num_epochs,
        num_inner_iter=num_inner_iter,
        num_MC_samples=mc_samples,
        learning_rate=lr,
        num_layers=num_layers,
        dim_ff=dim_ff,
        num_heads=n_heads,
        dim_attn=16*n_heads,
        use_transformer=use_transformer,
        use_prior=use_prior,
        prior_data=prior_data,
        temp_Q=temp_Q,
        replay_buffer=False,
    )
    start_time = time()
    d_joint = joint_eda.precompute()
    
    # Add all data needed for plots and tables, but remove the model
    joint_data = {
        'name': 'Joint EDA',
        'runtime': int(time() - start_time),
        # Core data for aggregate_method_results
        'max': d_joint['values'].max(axis=-1),
        'min': d_joint['values'].min(axis=-1), 
        'weights': d_joint['weights'],
        'mean': calculate_epoch_stats(d_joint['values'])[0],
        'std': calculate_epoch_stats(d_joint['values'])[1],
        'ess': d_joint['ess'],
        'mode': joint_eda.get_mode(),
        # Additional plotting data
        'values': d_joint['values'],  # Full values array for plotting
        'samples': d_joint.get('samples', None),  # Sample history if available
        'losses': d_joint.get('losses', None),   # Loss history if available
        'kld': d_joint.get('kld', None),         # KLD history if available
        # For plotting functions that need get_solutions() data
        'plot_samples': None,  # Will be populated by get_solutions if needed
        'plot_preds': None,    # Will be populated by get_solutions if needed 
        'plot_values': None,   # Will be populated by get_solutions if needed
        # Hyperparameters
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed,
        'method': 'joint'
    }
    # print(f"Trained Joint EDA on objective (L={L}): {joint_data['runtime']} s")
    results.append(joint_data)

    fgo = GeneralizedDADO(
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=num_epochs,
        num_inner_iter=num_inner_iter,
        num_MC_samples=mc_samples,
        learning_rate=lr,
        num_layers=num_layers,
        dim_ff=dim_ff,
        num_heads=n_heads,
        dim_attn=16*n_heads,
        use_transformer=use_transformer,
        verbose=False,
        use_prior=use_prior,
        prior_data=prior_data,
        temp_Q=temp_Q,
        replay_buffer=False,
    )
    start_time = time()
    d_fd = fgo.precompute()
    
    # Add all data needed for plots and tables, but remove the model
    fd_data = {
        'name': 'FGO',
        'runtime': int(time() - start_time),
        # Core data for aggregate_method_results
        'max': d_fd['values'].max(axis=-1),
        'min': d_fd['values'].min(axis=-1),
        'weights': d_fd['weights'], 
        'mean': calculate_epoch_stats(d_fd['values'])[0],
        'std': calculate_epoch_stats(d_fd['values'])[1],
        'ess': d_fd['ess'],
        'mode': fgo.get_mode(),
        # Additional plotting data
        'values': d_fd['values'],  # Full values array for plotting
        'samples': d_fd.get('samples', None),  # Sample history if available
        'losses': d_fd.get('losses', None),    # Loss history if available
        'kld': d_fd.get('kld', None),          # KLD history if available
        # For plotting functions that need get_solutions() data
        'plot_samples': None,  # Will be populated by get_solutions if needed
        'plot_preds': None,    # Will be populated by get_solutions if needed
        'plot_values': None,   # Will be populated by get_solutions if needed
        # Hyperparameters
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed, 
        'method': 'fd'
    }
    # print(f"Trained FGO on objective (L={L}): {fd_data['runtime']} s")
    results.append(fd_data)
    
    return results


def run_experiment(temp_Q, lr, obj, prior_data, seeds, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, n_heads, use_transformer, use_prior, outdir):
    """Run an experiment with a given Q, lr."""
    if os.path.exists(f"{outdir}/results/Q{temp_Q}-lr{lr}.dict"):
        stats = mlxu.load_pickle(f"{outdir}/results/Q{temp_Q}-lr{lr}.dict")
        return (stats['selection_joint'], stats['selection_fd'])
    
    # Run experiments and save intermediate results for each replicate
    for i, seed in enumerate(seeds):
        # Check if intermediate result already exists
        intermediate_file = f"{outdir}/results/Q{temp_Q}-lr{lr}_s{seed}.dict"
        if os.path.exists(intermediate_file):
            continue
            
        # Train both models for this seed
        results = train_experiments(
            obj, n_epochs, n_inner_iter, n_mc_samples, 
            n_layers, dim_ff, lr, use_prior, prior_data, 
            n_heads, use_transformer, temp_Q, seed
        )
        
        joint_result, fd_result = results  # Joint EDA result and FGO result
        
        # Save intermediate result for this replicate
        # TODO: modify this to save all data, using train experiments results...
        intermediate_stats = {
            'joint_max': joint_result['max'],
            'joint_min': joint_result['min'],
            'joint_weights': joint_result['weights'],
            'joint_mean': joint_result['mean'],
            'joint_std': joint_result['std'],
            'joint_ess': joint_result['ess'],
            'joint_mode': joint_result['mode'],
            'joint_values': joint_result['values'],
            'joint_samples': joint_result['samples'],
            'joint_losses': joint_result['losses'],
            'joint_kld': joint_result['kld'],
            'joint_runtime': joint_result['runtime'],
            'fd_max': fd_result['max'],
            'fd_min': fd_result['min'],
            'fd_weights': fd_result['weights'],
            'fd_mean': fd_result['mean'],
            'fd_std': fd_result['std'],
            'fd_ess': fd_result['ess'],
            'fd_mode': fd_result['mode'],
            'fd_values': fd_result['values'],
            'fd_samples': fd_result['samples'],
            'fd_losses': fd_result['losses'],
            'fd_kld': fd_result['kld'],
            'fd_runtime': fd_result['runtime'],
            'temp_Q': temp_Q,
            'lr': lr,
            'seed': seed,
            'rep_idx': i,
        }
        mlxu.save_pickle(intermediate_stats, intermediate_file)
    # Load and aggregate intermediate results
    joint_means, joint_stds, joint_ess = [], [], []
    joint_maxes, joint_mins, joint_Q = [], [], []
    joint_modes, joint_values, joint_samples, joint_losses, joint_kld = [], [], [], [], []
    fd_means, fd_stds, fd_ess = [], [], []
    fd_maxes, fd_mins, fd_Q = [], [], []
    fd_modes, fd_values, fd_samples, fd_losses, fd_kld = [], [], [], [], []

    for i in range(len(seeds)):
        intermediate_file = f"{outdir}/results/Q{temp_Q}-lr{lr}_s{seeds[i]}.dict"
        rep_stats = mlxu.load_pickle(intermediate_file)
        
        # Aggregate joint results
        joint_maxes.append(rep_stats['joint_max'])
        joint_mins.append(rep_stats['joint_min'])
        joint_Q.append(rep_stats['joint_weights'])
        joint_means.append(rep_stats['joint_mean'])
        joint_stds.append(rep_stats['joint_std'])
        joint_ess.append(rep_stats['joint_ess'])
        joint_modes.append(rep_stats['joint_mode'][0]) # only take the sequence, not the values
        joint_values.append(rep_stats['joint_values'])
        joint_samples.append(rep_stats['joint_samples'])
        joint_losses.append(rep_stats['joint_losses'])
        joint_kld.append(rep_stats['joint_kld'])
        
        # Aggregate fd results
        fd_maxes.append(rep_stats['fd_max'])
        fd_mins.append(rep_stats['fd_min'])
        fd_Q.append(rep_stats['fd_weights'])
        fd_means.append(rep_stats['fd_mean'])
        fd_stds.append(rep_stats['fd_std'])
        fd_ess.append(rep_stats['fd_ess'])
        fd_modes.append(rep_stats['fd_mode'][0]) # only take the sequence, not the values
        fd_values.append(rep_stats['fd_values'])
        fd_samples.append(rep_stats['fd_samples'])
        fd_losses.append(rep_stats['fd_losses'])
        fd_kld.append(rep_stats['fd_kld'])
        
        # Clean up intermediate file
        os.remove(intermediate_file)
    
    # Stack and compute final statistics
    joint_means_stacked = np.stack(joint_means, axis=0) # (n_reps, n_epochs)
    joint_stds_stacked = np.stack(joint_stds, axis=0) # (n_reps, n_epochs)
    joint_means_mean = joint_means_stacked.mean(axis=0) # (n_epochs,)
    joint_stds_mean = joint_stds_stacked.mean(axis=0) # (n_epochs,)
    selection_joint = joint_means_mean[-1] # scalar
    joint_maxes_stacked = np.stack(joint_maxes, axis=0) # (n_reps, n_epochs)
    joint_mins_stacked = np.stack(joint_mins, axis=0) # (n_reps, n_epochs)
    joint_Q_stacked = np.stack(joint_Q, axis=0).mean(axis=-2) # (n_reps, n_epochs, 1)
    joint_modes_stacked = np.stack(joint_modes, axis=0) # (n_reps, n_epochs)
    joint_values_stacked = np.stack(joint_values, axis=0) # (n_reps, n_epochs, L)
    joint_samples_stacked = np.stack(joint_samples, axis=0) # (n_reps, n_epochs, L)
    joint_losses_stacked = np.stack(joint_losses, axis=0) # (n_reps, n_epochs)
    joint_kld_stacked = np.stack(joint_kld, axis=0) # (n_reps, n_epochs)

    fd_means_stacked = np.stack(fd_means, axis=0) # (n_reps, n_epochs)
    fd_stds_stacked = np.stack(fd_stds, axis=0) # (n_reps, n_epochs)
    fd_means_mean = fd_means_stacked.mean(axis=0) # (n_epochs,)
    fd_stds_mean = fd_stds_stacked.mean(axis=0) # (n_epochs,)
    selection_fd = fd_means_mean[-1] # scalar
    fd_maxes_stacked = np.stack(fd_maxes, axis=0) # (n_reps, n_epochs)
    fd_mins_stacked = np.stack(fd_mins, axis=0) # (n_reps, n_epochs)
    fd_Q_stacked = np.stack(fd_Q, axis=0).mean(axis=-2) # (n_reps, n_epochs, L)
    fd_modes_stacked = np.stack(fd_modes, axis=0) # (n_reps, n_epochs)
    fd_values_stacked = np.stack(fd_values, axis=0) # (n_reps, n_epochs, L)
    fd_samples_stacked = np.stack(fd_samples, axis=0) # (n_reps, n_epochs, L)
    fd_losses_stacked = np.stack(fd_losses, axis=0) # (n_reps, n_epochs)
    fd_kld_stacked = np.stack(fd_kld, axis=0) # (n_reps, n_epochs)

    # Save detailed statistics
    stats_dict = {
        'joint_means_stacked': joint_means_stacked,
        'joint_stds_stacked': joint_stds_stacked,
        'joint_means_mean': joint_means_mean,
        'joint_stds_mean': joint_stds_mean,
        'joint_ess': joint_ess,
        'joint_maxes': joint_maxes_stacked,
        'joint_mins': joint_mins_stacked,
        'joint_weights': joint_Q_stacked,
        'joint_mode': joint_modes_stacked,
        'joint_values': joint_values_stacked,
        'joint_samples': joint_samples_stacked,
        'joint_losses': joint_losses_stacked,
        'joint_kld': joint_kld_stacked,
        'fd_maxes': fd_maxes_stacked,
        'fd_mins': fd_mins_stacked,
        'fd_means_stacked': fd_means_stacked,
        'fd_stds_stacked': fd_stds_stacked,
        'fd_means_mean': fd_means_mean,
        'fd_stds_mean': fd_stds_mean,
        'fd_ess': fd_ess,
        'fd_weights': fd_Q_stacked,
        'fd_mode': fd_modes_stacked,
        'fd_values': fd_values_stacked,
        'fd_samples': fd_samples_stacked,
        'fd_losses': fd_losses_stacked,
        'fd_kld': fd_kld_stacked,
        'temp_Q': temp_Q,
        'lr': lr,
        'selection_joint': selection_joint,
        'selection_fd': selection_fd
    }
    filename = f"Q{temp_Q}-lr{lr}.dict"
    mlxu.save_pickle(stats_dict, f"{outdir}/results/{filename}")

    return (selection_joint, selection_fd)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AF-FGM hyperparameter sweep experiments')
    
    # Basic arguments
    available_objs = list(STRUCTURE_PATHS_DICT.keys())
    parser.add_argument(
        '--obj', 
        type=str, 
        default='gfp',
        help=f'Objective to run. Available: {", ".join(available_objs)}. '
             f'Default: gfp'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory name. Results will be saved to experiments/AF_FGM/outdir/<obj_name>/. '
             'Default: uses timestamp'
    )
    parser.add_argument(
        '--plot_only',
        action='store_true',
        help='Only plot the results, do not run the experiments'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Just try printing all results to see updated progress'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.dict file to load as template'
    )
    
    # Global parameters
    parser.add_argument('--seed', type=int, default=91, help='Random seed')
    parser.add_argument('--n_reps', type=int, default=1, help='Number of replicates for each Q, lr')
    parser.add_argument('--n_final_reps', type=int, default=20, help='Number of final replicates for each Q, lr (addtl)')
    parser.add_argument('--n_pool', type=int, default=8, help='Number of processes to use in parallel. Usually RAM-constrained.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use (e.g., "1,2,3,4,5,6,7")')
    
    # Oracle parameters
    parser.add_argument('--n_epochs_oracle', type=int, default=1000, help='Number of epochs for oracle training')
    parser.add_argument('--n_inner_iter_oracle', type=int, default=1, help='Number of inner iterations for oracle training')
    parser.add_argument('--batch_size_oracle', type=int, default=2**14, help='Batch size for oracle training')
    parser.add_argument('--lr_oracle', type=float, default=1e-3, help='Learning rate for oracle training')
    parser.add_argument('--fgm_oracle', action='store_true', help='Use FGM oracle instead of NN w/o any structural bias')
    
    # Offline data
    # just choose worst 1000 sequences to fit prior to as a starting point.
    parser.add_argument('--n_offline', type=int, default=1000, help='Number of offline data points for initializing search distribution.')
    
    # FGM parameters
    parser.add_argument('--batch_size_fgm', type=int, default=2**10, help='Batch size for FGM training (will be set to n_offline)')
    parser.add_argument('--gr_batch_size_fgm', type=int, default=-1, help='Graph batch size for FGM training')
    parser.add_argument('--n_epochs_fgm', type=int, default=5000, help='Number of epochs for FGM training')
    parser.add_argument('--n_inner_iter_fgm', type=int, default=1, help='Number of inner iterations (each batch) for FGM training')
    parser.add_argument('--lr_fgm', type=float, default=1e-3, help='Learning rate for FGM training')
    parser.add_argument('--hidden_dims_fgm', type=str, default='128,128,16', help='Hidden dimensions for FGM (comma-separated)')
    parser.add_argument('--embedding_dim_fgm', type=int, default=4, help='Embedding dimension for FGM')
    
    # Optimization parameters (hyperparameter sweep)
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for optimization')
    parser.add_argument('--n_inner_iter', type=int, default=1, help='Number of inner iterations for optimization')
    parser.add_argument('--n_mc_samples', type=int, default=100, help='Number of MC samples for optimization')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for optimization models')
    parser.add_argument('--dim_ff', type=int, default=64, help='Feed-forward dimension for optimization models')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for transformer policy')
    parser.add_argument('--transformer', action='store_true', help='Use transformer policy instead of MLPs')
    parser.add_argument('--use_prior', type=str, default='init_only', help='Prior to use for optimization')

    args = parser.parse_args()
    
    # Parse comma-separated arguments
    args.hidden_dims_fgm = [int(x.strip()) for x in args.hidden_dims_fgm.split(',')]
    
    # Set objective-specific FGM parameters
    if args.obj == "amyloid": # L=42
        args.hidden_dims_fgm = [128, 16]
        args.lr_fgm = 0.0001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "aav": # L=28
        args.hidden_dims_fgm = [128, 128, 16]
        args.lr_fgm = 0.0001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "gb1_55": # L=55
        args.hidden_dims_fgm = [128, 128, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "gfp": # L=233
        args.hidden_dims_fgm = [128, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "tdp43": # L=84
        args.hidden_dims_fgm = [16, 16]
        args.lr_fgm = 0.0001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == 'ynzc': # L=39
        args.hidden_dims_fgm = [128, 128, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "phot": # L=118
        args.hidden_dims_fgm = [16, 16]
        args.lr_fgm = 0.0001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "his7": # L=171
        args.hidden_dims_fgm = [128, 128, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    elif args.obj == "gcn4": # L=44, Val R^2 is -.15
        args.hidden_dims_fgm = [16, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 5000
    
    # Validate objectives
    if args.obj not in available_objs:
        raise ValueError(f"Invalid objective: {args.obj}. Available objectives: {available_objs}")
    
    return args


def main():
    """Main execution function."""
    st = time()
    # Set multiprocessing start method to avoid JAX/CUDA issues
    mp.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    args = parse_args()
    args.fgm_oracle = True # always use FGM oracle in this case.

    # Parse allowed GPUs
    allowed_gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    print(f"Using GPUs: {allowed_gpus}")

    obj_name = args.obj
    print(f"Running experiments for objective: {obj_name}")
    
    # Create output directory
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"experiments/search_AF_FGM/{timestamp}/{obj_name}"
    else:
        outdir = f"experiments/search_AF_FGM/{args.outdir}/{obj_name}"
    
    os.makedirs(outdir + '/results', exist_ok=True)
    print(f"Results will be saved to: {outdir}")
    
    # Save configuration
    config_path = f"{outdir}/config.dict"
    with open(config_path, 'wb') as f:
        pickle.dump(vars(args), f)
    print(f"Configuration saved to: {config_path}")
    
    rkey = jax.random.PRNGKey(args.seed)
    
    # Sequential pre-training phase
    print("=== SEQUENTIAL PRE-TRAINING PHASE ===")
    
    # 1. Fit oracle
    print("1. Fitting oracle...")
    rkey, ckey = jax.random.split(rkey)
    obj, preds = fit_oracle(
        obj_name, 
        args.n_epochs_oracle, args.n_inner_iter_oracle, 
        args.batch_size_oracle, args.lr_oracle, jaxrng_to_int(ckey), Path(outdir),
        vis=True, fit=(not args.fgm_oracle or args.plot_only)
    )

    args.batch_size_fgm = min(
        args.batch_size_fgm,
        min(args.n_offline if args.n_offline > 0 else obj.genotypes.shape[0], 2**10)
    )

    STRUCTURE_PATH = os.path.join(
            os.environ["DIR_DATA"], 
            "af3_structures", 
            STRUCTURE_PATHS_DICT[obj_name]
    )
    
    # Handle FGM oracle case
    if args.fgm_oracle and not args.plot_only:
        print("Using FGM oracle - loading/training FGM oracle model...")
        rkey, ckey = jax.random.split(rkey)
        fgm, _ = fit_reward_AF_FGM(
            obj, None, 
            STRUCTURE_PATH,
            args.hidden_dims_fgm, args.batch_size_fgm, args.gr_batch_size_fgm, args.embedding_dim_fgm,
            args.n_epochs_fgm, args.n_inner_iter_fgm, args.lr_fgm, jaxrng_to_int(ckey), Path(outdir),
            vis=True, save_load=True, fit=True,
        )
        graphdef, state = nnx.split(fgm.fg.model)
        fgm.fg.model = nnx.merge(graphdef, state)
        obj.model = fgm.fg
        obj.model.forward = obj.model.__call__
        obj.batch_size = args.batch_size_fgm
        max_value = fgm.fg.model.constant.value + 1.
        print(f"FGM max value: {max_value}")

        # 3. Fit AF-FGM models
        print("3. Fitting AF-FGM model...just using objective!")
        reward_model_fgm = fgm

    # 2. Prepare offline data from oracle (WT and sequences with values ≤ WT)
    print("2. Preparing offline data...")
    # NOTE: take worst 1000 sequences to fit prior.
    if not args.plot_only:
        # preds = obj.query_all()
        preds = obj.values
        offline_inds = jnp.argsort(preds)[:1000]
        
        obj_max = preds.max()
        print(f"Max dataset: {obj_max}")
        offline_max = preds[offline_inds].max()
        print(f"Max offline: {offline_max}")

        # 4. Prepare prior data
        print("4. Preparing prior data...")
        prior_data = obj.genotypes[offline_inds]

    # misc: Create parameter tuples for multiprocessing
    np_rng = np.random.default_rng(args.seed)
    seeds = np_rng.integers(
        low=0, high=np.iinfo(np.int32).max, size=args.n_reps
        ).tolist()
    
    # 5. Fit priors for both Joint EDA and FD EDA (using 0 epochs to only fit prior)
    if not args.plot_only and args.use_prior != 'none':
        print("5. Fitting priors for Joint and FD EDAs...")
        
        # Fit priors for both EDA types (0 epochs, just to train the priors, params don't really matter)
        train_experiments(
            reward_model_fgm, num_epochs=0, num_inner_iter=args.n_inner_iter, mc_samples=args.n_mc_samples,
            num_layers=args.n_layers, dim_ff=args.dim_ff, lr=1e-3, use_prior=args.use_prior, 
            prior_data=prior_data, n_heads=args.n_heads, 
            use_transformer=args.transformer, temp_Q=1.0, seed=seeds[0]
        )
    
    print("=== HYPERPARAMETER SWEEP PHASE ===")
    
    # Hyperparameter sweep setup
    Q_sweep = [0.01, 0.05, 0.1, 0.5, 1, 5] # weight temperature
    lr_sweep = [5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2][::-1] # flip so less stable (higher) lr come first, less likely selected.
    print(f"Sweeping over {len(Q_sweep)} Q values, {len(lr_sweep)} learning rate values= {len(Q_sweep) * len(lr_sweep)} total.")
    Q_lr_sweep = np.array(np.meshgrid(Q_sweep, lr_sweep)).T.reshape(-1, 2)
    print(f"Cartesian product of Q, lr: {Q_lr_sweep.shape}")

    
    # Create FGM model parameters for reconstruction in workers
    fgm_params = {
        'obj_name': obj_name,
        'offline_inds': None, # use all
        'structure_path': STRUCTURE_PATH,
        'hidden_dims_fgm': args.hidden_dims_fgm,
        'batch_size_fgm': args.batch_size_fgm,
        'gr_batch_size_fgm': args.gr_batch_size_fgm,
        'embedding_dim_fgm': args.embedding_dim_fgm,
        'n_epochs_fgm': args.n_epochs_fgm,
        'n_inner_iter_fgm': args.n_inner_iter_fgm,
        'lr_fgm': args.lr_fgm,
        'seed': jaxrng_to_int(rkey)
    }
    
    if not args.plot_only:
        params_list = []
        for i, (temp_Q, lr) in enumerate(Q_lr_sweep):
            params_list.append((temp_Q, lr, fgm_params, prior_data, seeds, args.n_epochs, args.n_inner_iter, 
                            args.n_mc_samples, args.n_layers, args.dim_ff, args.n_heads, args.transformer, 
                            args.use_prior, outdir))

        with mp.Manager() as manager:
            # Create GPU assignment queue
            gpu_queue = manager.Queue()
            # Cycle through allowed GPUs for pool processes
            for i in range(args.n_pool):
                gpu_queue.put(allowed_gpus[i % len(allowed_gpus)])

            with mp.Pool(processes=args.n_pool, initializer=init_worker, initargs=(gpu_queue, allowed_gpus)) as pool:
                async_result = pool.map_async(run_experiment_wrapper, params_list)
                results = async_result.get()

    if not args.plot_only:
        mlxu.save_pickle(results, f"{outdir}/results.pkl")
    else:
        if os.path.exists(f"{outdir}/results.pkl"):
            results = mlxu.load_pickle(f"{outdir}/results.pkl")
        else:
            raise ValueError(f"No results found in {outdir}/results.pkl")

    def argmax_last(x, axis=None):
        flipped = jnp.flip(x, axis=axis)
        idx_flipped = jnp.argmax(flipped, axis=axis)
        size = x.shape[axis] if axis is not None else x.size
        return size - 1 - idx_flipped

    # find best Q, lr for each of joint, FGO by max mean
    joint_selection = jnp.array([r[0] for r in results])
    print(f"Joint selection: min {jnp.min(joint_selection)}, max {jnp.max(joint_selection)}, mean {jnp.mean(joint_selection)}, std {jnp.std(joint_selection)}")
    fd_selection = jnp.array([r[1] for r in results])
    print(f"FGO selection: min {jnp.min(fd_selection)}, max {jnp.max(fd_selection)}, mean {jnp.mean(fd_selection)}, std {jnp.std(fd_selection)}")
    joint_ind = jnp.argmax(joint_selection)
    joint_ind_l = argmax_last(joint_selection)
    if joint_ind != joint_ind_l and joint_selection[joint_ind] == joint_selection[joint_ind_l]:
        print(f"Joint: {Q_lr_sweep[joint_ind]} and {Q_lr_sweep[joint_ind_l]} both optimal.")
        joint_ind = joint_ind_l # use later one w/ higher Q -- probably more stable.
    fd_ind = jnp.argmax(fd_selection)
    fd_ind_l = argmax_last(fd_selection)
    if fd_ind != fd_ind_l and fd_selection[fd_ind] == fd_selection[fd_ind_l]:
        print(f"DADO: {Q_lr_sweep[fd_ind]} and {Q_lr_sweep[fd_ind_l]} both optimal.")
        fd_ind = fd_ind_l # use later one w/ higher Q -- probably more stable.
    joint_Q_star, joint_lr_star = Q_lr_sweep[joint_ind]
    fd_Q_star, fd_lr_star = Q_lr_sweep[fd_ind]
    print(f"Best joint Q, lr: {joint_Q_star}, {joint_lr_star}, {joint_selection[jnp.argmax(joint_selection)]}")
    print(f"Best FGO Q, lr: {fd_Q_star}, {fd_lr_star}, {fd_selection[jnp.argmax(fd_selection)]}")

    # Load best statistics for plotting and create EDA result objects for visualization
    joint_filename = f"Q{joint_Q_star}-lr{joint_lr_star}.dict"
    fd_filename = f"Q{fd_Q_star}-lr{fd_lr_star}.dict"
    joint_stats = mlxu.load_pickle(f"{outdir}/results/{joint_filename}")
    fd_stats = mlxu.load_pickle(f"{outdir}/results/{fd_filename}")

    joint_means = joint_stats['joint_means_mean']
    joint_stds = joint_stats['joint_stds_mean']
    joint_maxes = joint_stats['joint_maxes']
    joint_mins = joint_stats['joint_mins']
    fd_means = fd_stats['fd_means_mean']
    fd_stds = fd_stats['fd_stds_mean']
    fd_maxes = fd_stats['fd_maxes']
    fd_mins = fd_stats['fd_mins']
    

    # Create comprehensive plot showing all Q/lr combinations
    plt.figure(figsize=(12, 8))
    
    # if hasattr(reward_model_fgm, 'best_val') and reward_model_fgm.best_val is not None:
    #     plt.axhline(y=reward_model_fgm.best_val, color='black', linestyle='-', label='Max')
    
    for i, (temp_Q, lr) in enumerate(Q_lr_sweep):
        filename = f"Q{temp_Q}-lr{lr}.dict"
        stats = mlxu.load_pickle(f"{outdir}/results/{filename}")

        joint_means_all = stats['joint_means_mean']
        if i == 0:  # Only add label for first plot
            plt.plot(joint_means_all, color='blue', alpha=0.2, label='Joint EDA (all)')
        else:
            plt.plot(joint_means_all, color='blue', alpha=0.2)
    
        stats = mlxu.load_pickle(f"{outdir}/results/{filename}")
        fd_means_all = stats['fd_means_mean']
        if i == 0:  # Only add label for first plot
            plt.plot(fd_means_all, color='red', alpha=0.2, label='FGO (all)')
        else:
            plt.plot(fd_means_all, color='red', alpha=0.2)
    
    # Highlight best results
    plt.plot(joint_means, color='blue', linewidth=2, label='Joint EDA (best)')
    plt.plot(fd_means, color='red', linewidth=2, label='FGO (best)')
    
    plt.title(f"{args.obj} Optimization - All Hyperparameters (Q, lr)")
    plt.xlabel('Training Epoch')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.savefig(f"{outdir}/sweep_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Total time: {(time() - st) / 60} minutes')
    
    # Additional replicates for best hyperparameters
    print(f'Will now replicate ({args.n_final_reps}x) the best results over seeds...')
    additional_seeds = np_rng.integers(
        low=0, high=np.iinfo(np.int32).max, size=args.n_final_reps
        ).tolist()
    
    if not args.plot_only:
        # Create parameter lists for additional replicates with best hyperparameters
        joint_additional_params = []
        fd_additional_params = []
        for i, seed in enumerate(additional_seeds):
            # Joint EDA additional replicates
            joint_additional_params.append((joint_Q_star, joint_lr_star, 'joint', fgm_params, prior_data, seed, args.n_epochs, 
                                          args.n_inner_iter, args.n_mc_samples, args.n_layers, args.dim_ff, args.n_heads, 
                                          args.transformer, args.use_prior, outdir))
            # FD EDA additional replicates
            fd_additional_params.append((fd_Q_star, fd_lr_star, 'fd', fgm_params, prior_data, seed, args.n_epochs, 
                                       args.n_inner_iter, args.n_mc_samples, args.n_layers, args.dim_ff, args.n_heads, 
                                       args.transformer, args.use_prior, outdir))
        
        all_additional_params = joint_additional_params + fd_additional_params
        
        with mp.Manager() as manager_additional:
            # Create GPU assignment queue for additional experiments
            gpu_queue_additional = manager_additional.Queue()
            # Cycle through allowed GPUs for pool processes
            for i in range(args.n_pool):
                gpu_queue_additional.put(allowed_gpus[i % len(allowed_gpus)])

            with mp.Pool(processes=args.n_pool, initializer=init_worker, initargs=(gpu_queue_additional, allowed_gpus)) as pool:
                additional_results = pool.map(method_specific_wrapper, all_additional_params)
    
    # Aggregate results combining initial sweep + additional replicates
    print("Aggregating final results...")
    joint_aggregated = aggregate_method_results('joint', joint_Q_star, joint_lr_star, seeds, additional_seeds, outdir)
    fd_aggregated = aggregate_method_results('fd', fd_Q_star, fd_lr_star, seeds, additional_seeds, outdir)
    
    print(f"Joint EDA with {len(seeds) + len(additional_seeds)} replicates: {joint_aggregated['selection']:.4f}")
    print(f"FD EDA with {len(seeds) + len(additional_seeds)} replicates: {fd_aggregated['selection']:.4f}")

    # Compute statistics:
    joint_final_mean_reps = np.array(joint_aggregated['means'][:, -1])
    fd_final_mean_reps = np.array(fd_aggregated['means'][:, -1])
    joint_auc_reps = np.array([np.trapezoid(joint_aggregated['means'][i]) for i in range(joint_aggregated['means'].shape[0])])
    fd_auc_reps = np.array([np.trapezoid(fd_aggregated['means'][i]) for i in range(fd_aggregated['means'].shape[0])])
    assert joint_auc_reps.size == fd_auc_reps.size == 20, f"Joint EDA AUC reps size: {joint_auc_reps.size}, FD EDA AUC reps size: {fd_auc_reps.size}"
    from scipy.stats import wilcoxon, ttest_rel
    # x = your algorithm, y = baseline
    stat, p_value = wilcoxon(fd_auc_reps, joint_auc_reps, alternative='greater')
    print(f"Wilcoxon statistic: {stat} w/ one-sided p-value: {p_value}")
    # x = your algorithm, y = baseline
    stat, p_two_sided = ttest_rel(fd_auc_reps, joint_auc_reps)
    print(f"Paired t-test statistic: {stat} w/ two-sided p-value: {p_two_sided}")
    
    # Create final aggregated plot
    joint_means = joint_aggregated['means_mean']
    joint_stds = joint_aggregated['stds_mean']
    joint_ci_lower = joint_aggregated['ci_lower'].mean(axis=0)
    joint_ci_upper = joint_aggregated['ci_upper'].mean(axis=0)
    joint_maxes = joint_aggregated['maxes'].mean(axis=0)
    joint_mins = joint_aggregated['mins'].mean(axis=0)
    
    fd_means = fd_aggregated['means_mean'] 
    fd_stds = fd_aggregated['stds_mean']
    fd_ci_lower = fd_aggregated['ci_lower'].mean(axis=0)
    fd_ci_upper = fd_aggregated['ci_upper'].mean(axis=0)
    fd_maxes = fd_aggregated['maxes'].mean(axis=0)
    fd_mins = fd_aggregated['mins'].mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # add p-value in bottom-right corner
    t = 'none' # used this in conjunction w/ test_graph_hp.py
    pstring = f"$\\mathbf{{P = {p_two_sided:.2e}}}$"
    if p_two_sided >= 0.01:
        pstring = f"$\\mathbf{{P = {p_two_sided:.2f}}}$"
    ax.text( # edit second coord to place it in non-overlapping position
        0.95, .8, pstring, 
        transform=ax.transAxes,  # use axes coords (0–1)
        ha="right", va="center", 
        fontsize=20
    )
    # if hasattr(obj, 'best_val') and obj.best_val is not None:
    # plt.axhline(y=max_value, color='black', linestyle='--', lw=2, label='Global Maximum')
    
    plt.plot(joint_means, label=f'EDA Sample Mean', color="#377EB8", lw=2)
    plt.fill_between(range(joint_means.shape[-1]), joint_ci_lower, joint_ci_upper, alpha=0.3, color="#377EB8", label='EDA 95% CI')
    plt.plot(fd_means, label=f'DADO Sample Mean', color="#E41A1C", lw=2)
    plt.fill_between(range(fd_means.shape[-1]), fd_ci_lower, fd_ci_upper, alpha=0.3, color="#E41A1C", label='DADO 95% CI')

    plt.title(f"{obj.obj_name_pretty} protein (L={obj.L}, D={obj.D[0]})", fontsize=30)
    # plt.title(f"{obj.obj_name_pretty} protein, {t} Å contact threshold", fontsize=30)
    plt.xlabel('Training Iteration', fontsize=26, labelpad=15); plt.xticks(fontsize=20)
    plt.ylabel('f(x)', fontsize=26, labelpad=15); plt.yticks(fontsize=20)
    plt.savefig(f"{outdir}/rep_results_{obj.obj_name}.png", dpi=600, bbox_inches='tight')
    if t != 'none': plt.savefig(f"{outdir}/rep_results_{obj.obj_name}_A{t}.png", dpi=600, bbox_inches='tight')
    plt.legend(fontsize=20, loc='lower right')
    plt.savefig(f"{outdir}/rep_results_{obj.obj_name}_legend.png", dpi=600, bbox_inches='tight')
    if t != 'none': plt.savefig(f"{outdir}/rep_results_{obj.obj_name}_A{t}_legend.png", dpi=600, bbox_inches='tight')
    plt.close()

    print(f"/nAll experiments completed! Results saved to: {outdir}")

if __name__ == "__main__":
    main()
