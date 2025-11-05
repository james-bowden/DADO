import os
from tqdm import tqdm
import sys
import argparse
import pickle
import math
from time import time
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
import mlxu

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Project imports
from src.problems.real.oracle import OracleObjective
from src.problems.synthetic.graph.base import SyntheticGraphObjective
from src.decomposition.graphs import GeneralizedGraph, JunctionTree, DisconnectedGraph
from src.decomposition.fgm.mlp.base import MLPGeneralizedFunctionalGraph, MLPFunctionalGraph
from src.decomposition.fgm.tabular.base import TabularGeneralizedFunctionalGraph, TabularFunctionalGraph
from src.problems.real.protein_structure import residue_contact_map
from src.utils import count_params, jaxrng_to_int, bottom_percentile_indices

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
        vis=True, fit=True, save_load=True, threshold=4.5, use_tabular_fgm=False):
    """Fit AF-based FGM reward models and save plots."""
    losses = None
    if vis and save_load:
        N_plots = 5
        fig, axes = plt.subplots(2, 3, figsize=(4*3, 3*2))
        axes = axes.flatten()

    checkpointer_model = ocp.StandardCheckpointer()
    checkpointer_misc = ocp.PyTreeCheckpointer()

    N = obj.values.size
    np_rng = np.random.default_rng(seed)
    holdout_inds = np_rng.choice(np.arange(N), size=N//10, replace=False)
    offline_inds = np.setdiff1d(np.arange(N), holdout_inds)
    assert np.intersect1d(offline_inds, holdout_inds).size == 0
    print(f"Offline indices: {offline_inds.size}, Holdout indices: {holdout_inds.size}")

    # NOTE: assuming graph is always the same...
    fdir = f"{os.getcwd()}/checkpoints/fgm/{obj.obj_name}/"
    fname = os.path.join(fdir, f"t{threshold}_arc_{'-'.join(map(str, hidden_dims))}_epochs_{num_epochs}_inner_{num_inner_iter}_lr_{learning_rate}_batch_{batch_size}_grbatch_{gr_batch_size}_embdim_{embedding_dim}_noffline_{'all' if offline_inds is None else len(offline_inds)}_seed_{seed}.nnx")
    if not os.path.exists(fname):
        alt_fnames = glob(os.path.join(fdir, f"t{threshold}_arc_{'-'.join(map(str, hidden_dims))}_epochs_{num_epochs}_inner_{num_inner_iter}_lr_{learning_rate}_batch_{batch_size}_grbatch_{gr_batch_size}_embdim_{embedding_dim}_noffline_{'all' if offline_inds is None else len(offline_inds)}_seed_{'*'}.nnx"))
        if len(alt_fnames) > 0:
            fname = alt_fnames[0]
            print(f"Found alternate seed. Using.")
    print(os.path.exists(fname), fname)
    if threshold > 0:
        edges_AF, _ = residue_contact_map(
            structure_path, 
            obj.WT_seq, # needs to be contiguous substring of the seq in the CIF file
            edgelist=True,
            threshold=threshold,
            verbose=False,
            )
    else: # threshold == 0
        edges_AF = [] # should yield linear model...
    gg = GeneralizedGraph(len(obj.D), edges_AF, verbose=False)
    if use_tabular_fgm: raise NotImplementedError("Tabular FGM not implemented yet.")
    for sg in gg.subgraphs:
        if isinstance(sg, JunctionTree):
            use_tabular_fgm = False
            break
    if use_tabular_fgm:
        fg = TabularGeneralizedFunctionalGraph.fixed_graph_random(
            gg, n_states_list=obj.D, seed=seed,
        )
        fit_fn = lambda fg, x, y, n_epochs, n_iter_inner, learning_rate, seed: TabularGeneralizedFunctionalGraph.fit(
            fg, x, y, n_iter_inner=n_iter_inner, n_epochs=n_epochs, learning_rate=learning_rate, seed=seed,
            weight_fn_name='id', # MLP doesn't have weight fn
        )
    else:
        fg = MLPGeneralizedFunctionalGraph(
            gg, n_states_list=obj.D, 
            hidden_dims=hidden_dims, 
            batch_size=batch_size,
            gr_batch_size=gr_batch_size,
            embedding_dim=embedding_dim,
            rngs=jax.random.PRNGKey(seed),
        )
        fit_fn = MLPGeneralizedFunctionalGraph.fit
    if fit:
        if not use_tabular_fgm and os.path.exists(fname) and save_load:
            graphdef, state = nnx.split(fg.model)
            state = checkpointer_model.restore(fname, state)
            if use_tabular_fgm:
                fg.subgraphs = nnx.merge(graphdef, state)
            else:
                fg.model = nnx.merge(graphdef, state)
            loaded_data = checkpointer_misc.restore(fname.replace(".nnx", ".dict"))
            losses = loaded_data["losses"]
            print(f"Successfully loaded model from {fname}")
        else:
            fg.train()
            losses = fit_fn(
                fg,
                obj.genotypes if offline_inds is None else obj.genotypes[offline_inds, :],
                obj.values if offline_inds is None else obj.values[offline_inds], 
                n_epochs=num_epochs,
                n_iter_inner=num_inner_iter,
                learning_rate=learning_rate,
                seed=seed,
            )
            if save_load:
                if use_tabular_fgm:
                    pass
                else:
                    graphdef, state = nnx.split(fg.model)
                    try: 
                        checkpointer_model.save(fname, state)
                        checkpointer_misc.save(fname.replace(".nnx", ".dict"), {"losses": losses})
                        print(f"Successfully saved model (and losses) to {fname}")
                    except Exception as e:
                        # NOTE: oracles are common, so another process may have saved in meantime. 
                        state = checkpointer_model.restore(fname, state)
                        if use_tabular_fgm:
                            fg.subgraphs = nnx.merge(graphdef, state)
                        else:
                            fg.model = nnx.merge(graphdef, state)
                        loaded_data = checkpointer_misc.restore(fname.replace(".nnx", ".dict"))
                        losses = loaded_data["losses"]
                        print(f"Training intercepted. Threw out and successfully loaded model from {fname}.")
    print(f"AF-FGM-{obj.obj_name} #params: {count_params(fg)}")
    fg.eval()
    fgm = SyntheticGraphObjective(
        D=obj.D,
        fg=MLPFunctionalGraph._split_merge(fg) if use_tabular_fgm else TabularFunctionalGraph._split_merge(fg), # NOTE: important, to sever tracing
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
            if sg.n_nodes == 1 or not hasattr(sg, 'root'):
                colors = ['red']
            else:
                colors = [
                    "red" if (sg.n_nodes > 1 and node == sg.root) else "blue" 
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
            if isinstance(sg, JunctionTree):
                node_sizes += sg.clique_sizes
                clique_sizes += sg.clique_parent_sizes
            else: # exact tree
                node_sizes += sg.n_nodes * [1]
                if isinstance(sg, DisconnectedGraph):
                    clique_sizes += [1] * sg.n_nodes
                elif sg.n_nodes > 1:
                    parents = sg.n_nodes * [2]
                    parents[sg.root] = 1
                    clique_sizes += parents
                else:
                    clique_sizes += [1]
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
            if use_tabular_fgm:
                y_pred = fg(obj.genotypes[holdout_inds])
            else:
                y_pred = fg.evaluate_batched(obj.genotypes[holdout_inds])
            y_true = obj.values[holdout_inds]
            test_mse = jnp.mean((y_pred - y_true)**2)
            print(f"Test MSE: {test_mse:.2e}")
            axes[4].scatter(y_true, y_pred, alpha=0.05)
            axes[4].set_xlabel('True Value')
            axes[4].set_ylabel('Predicted Value')
            pr = jnp.corrcoef(y_true, y_pred)[0, 1]
            sr = spearmanr(y_true, y_pred, axis=None).statistic
            axes[4].set_title(f'Real Data Corr: {pr:.2f}P / {sr:.2f}S')
    
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4)
        plt.savefig(output_dir / f'af_fgm_training_t{threshold}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return fgm, test_mse, pr, clique_sizes, node_sizes

def init_worker(gpu_queue, allowed_gpus):
    """Initialize worker process for multiprocessing pool."""
    global _worker_gpu
    _worker_gpu = gpu_queue.get()  # GPU ID from allowed list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu)

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
    parser.add_argument('--n_final_reps', type=int, default=5, help='Number of final replicates for each Q, lr (addtl)')
    parser.add_argument('--n_pool', type=int, default=8, help='Number of processes to use in parallel. Usually RAM-constrained.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use (e.g., "1,2,3,4,5,6,7")')
    
    # Oracle parameters
    parser.add_argument('--n_epochs_oracle', type=int, default=1000, help='Number of epochs for oracle training')
    parser.add_argument('--n_inner_iter_oracle', type=int, default=1, help='Number of inner iterations for oracle training')
    parser.add_argument('--batch_size_oracle', type=int, default=2**14, help='Batch size for oracle training')
    parser.add_argument('--lr_oracle', type=float, default=1e-3, help='Learning rate for oracle training')
    parser.add_argument('--fgm_oracle', action='store_true', help='Use FGM oracle instead of NN w/o any structural bias')
    
    # Offline data
    parser.add_argument('--n_offline', type=int, default=1000, help='Number of offline data points')
    
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
    parser.add_argument('--n_mc_samples', type=int, default=1000, help='Number of MC samples for optimization')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for optimization models')
    parser.add_argument('--dim_ff', type=int, default=64, help='Feed-forward dimension for optimization models')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for transformer policy')
    parser.add_argument('--transformer', action='store_true', help='Use transformer policy instead of MLPs')
    parser.add_argument('--use_prior', type=str, default='init_only', help='Prior to use for optimization')

    args = parser.parse_args()
    
    # Parse comma-separated arguments
    args.hidden_dims_fgm = [int(x.strip()) for x in args.hidden_dims_fgm.split(',')]
    
    # Set objective-specific FGM parameters based on CV
    if args.obj == "gb1_55":
        args.hidden_dims_fgm = [128, 128, 16]
        args.lr_fgm = 0.001
        args.n_inner_iter_fgm = 1
        args.n_epochs_fgm = 50000
    else:
        raise NotImplementedError("Default parameters not specified for this objective.")
    
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
        vis=True, fit=(not args.fgm_oracle)
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
    thresh_a = [0., 2, 2.5, 2.75, 3, 4, 4.5, 5, 6, 7, 8, 9, 10, 15]
    max_clique_sizes, pearsons, mses, max_node_sizes = [], [], [], []
    if args.fgm_oracle:
        print("Using FGM oracle - loading/training FGM oracle model...")
        for t in tqdm(thresh_a):
            rkey, ckey = jax.random.split(rkey)
            fgm, test_mse, pearson, clique_sizes, node_sizes = fit_reward_AF_FGM(
                obj, None, 
                STRUCTURE_PATH,
                args.hidden_dims_fgm, args.batch_size_fgm, args.gr_batch_size_fgm, args.embedding_dim_fgm,
                args.n_epochs_fgm, args.n_inner_iter_fgm, args.lr_fgm, jaxrng_to_int(ckey), Path(outdir),
                vis=True, save_load=True, fit=True, threshold=t,
            )
            max_clique_sizes.append(max(clique_sizes))
            pearsons.append(pearson)
            mses.append(test_mse)
            max_node_sizes.append(max(node_sizes))

    inds_plotting = [2, 5, 9]
    other_inds = [i for i in range(len(thresh_a)) if i not in inds_plotting]
    thresh_a, pearsons, mses, max_clique_sizes, max_node_sizes = np.array(thresh_a), np.array(pearsons), np.array(mses), np.array(max_clique_sizes), np.array(max_node_sizes)
    inds_plotting, other_inds = np.array(inds_plotting), np.array(other_inds)
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # Left y-axis: MSE (line + dots)
    ax1.plot(thresh_a, pearsons, color='#377EB8', lw=2, alpha=0.7)
    ax1.scatter(thresh_a[other_inds], pearsons[other_inds], marker='o', color='#377EB8', s=10, label='Holdout Pearson') # blue
    ax1.scatter(thresh_a[inds_plotting], pearsons[inds_plotting], marker='*', color='#377EB8', s=100) # red
    ax1.set_xlabel('Residue contact threshold (Å)', fontsize=26, labelpad=15)
    ax1.set_ylabel('Holdout Pearson correlation', color='#377EB8', fontsize=26, labelpad=15)
    ax1.set_ylim([.94, .98])
    ax1.set_yticks([.94,.95, .96, .97, .98])
    ax1.tick_params(axis='y', labelcolor='#377EB8', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(False)
    # Right y-axis: max clique size (bars)
    ax2 = ax1.twinx()
    ax2.bar(thresh_a, max_node_sizes, color='#E41A1C', alpha=0.8, width=0.07)
    # draw "caps" atop each bar
    ax2.errorbar(
        thresh_a, max_node_sizes,
        yerr=[np.zeros_like(max_node_sizes), np.full_like(max_node_sizes, 1e-8)],  
        fmt='none',  # don't draw a marker
        ecolor='#E41A1C',
        elinewidth=1.5,
        capsize=5,  # length of the cap
        label='Largest node size'
    )
    ax2.set_ylabel('Largest meta-variable cardinality', color='#E41A1C', fontsize=26, labelpad=15)
    ax2.tick_params(axis='y', labelcolor='#E41A1C', labelsize=20)
    # Optional: combine legends
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    # ax1.legend(lines + bars, labels + bar_labels, loc='best', fontsize=20)
    plt.title(f"GB1 predictive model decompositions", fontsize=30)
    plt.tight_layout()
    plt.savefig(f"{outdir}/pearson_clique_angstroms.png", dpi=600, bbox_inches='tight')
    plt.close()


    fig, ax1 = plt.subplots()
    # Left y-axis: MSE (line + dots)
    ax1.plot(thresh_a, mses, color='#377EB8', lw=2, alpha=0.7)
    ax1.scatter(thresh_a[other_inds], mses[other_inds], marker='o', color='#377EB8', s=10, label='Holdout MSE') # blue
    ax1.scatter(thresh_a[inds_plotting], mses[inds_plotting], marker='*', color='#377EB8', s=100) # red
    ax1.set_xlabel('Residue contact threshold (Å)', fontsize=26, labelpad=15)
    ax1.set_ylabel('Holdout MSE', color='#377EB8', fontsize=26, labelpad=15)
    ax1.tick_params(axis='y', labelcolor='#377EB8', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(False)
    # Right y-axis: max clique size (bars)
    ax2 = ax1.twinx()
    ax2.bar(thresh_a, max_node_sizes, color='#E41A1C', alpha=0.8, width=0.07)
    # draw "caps" atop each bar
    ax2.errorbar(
        thresh_a, max_node_sizes,
        yerr=[np.zeros_like(max_node_sizes), np.full_like(max_node_sizes, 1e-8)],  
        fmt='none',  # don't draw a marker
        ecolor='#E41A1C',
        elinewidth=1.5,
        capsize=5,  # length of the cap
        label='Largest node size'
    )
    ax2.set_ylabel('Largest meta-variable cardinality', color='#E41A1C', fontsize=26, labelpad=15)
    ax2.tick_params(axis='y', labelcolor='#E41A1C', labelsize=20)
    # Optional: combine legends
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    # ax1.legend(lines + bars, labels + bar_labels, loc='best', fontsize=20)
    plt.title(f"GB1 predictive model decompositions", fontsize=30)
    plt.tight_layout()
    plt.savefig(f"{outdir}/mse_clique_angstroms.png", dpi=600, bbox_inches='tight')
    plt.close()


    
    print(f"/nAll experiments completed! Results saved to: {outdir}")

if __name__ == "__main__":
    main()
