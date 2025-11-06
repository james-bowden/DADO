import os
import sys
import argparse
from datetime import datetime
from time import time
from copy import deepcopy
import multiprocessing as mp
import pickle

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Check if working directory ends with 'DADO', then append to sys.path
cwd = os.getcwd()
if cwd.endswith('DADO'):
    sys.path.append(cwd)
else:
    raise ValueError(f"Working directory {cwd} does not end with 'DADO'")

# Third-party imports
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlxu
from glob import glob

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Project imports
from src.problems.synthetic.graph.tree.TreeObjective1 import TreeObjective1 as Tree1
from src.problems.synthetic.graph.tree.TreeObjective2 import TreeObjective2 as Tree2
from src.problems.synthetic.graph.tree.TreeObjective3 import TreeObjective3 as Tree3
from src.problems.synthetic.graph.tree.TreeObjective4 import TreeObjective4 as Tree4
from src.problems.synthetic.graph.tree.TreeObjective5 import TreeObjective5 as Tree5
from src.problems.synthetic.graph.tree.ChainObjective1 import ChainObjective1 as Chain1
from src.problems.synthetic.graph.tree.ChainObjective2 import ChainObjective2 as Chain2
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective1 import SE_TreeObjective1 as SETree1
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective2 import SE_TreeObjective2 as SETree2
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective3 import SE_TreeObjective3 as SETree3
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective4 import SE_TreeObjective4 as SETree4
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective5 import SE_TreeObjective5 as SETree5
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective6 import SE_TreeObjective6 as SETree6
from src.problems.synthetic.graph.tree.sign_epistatic.TreeObjective7 import SE_TreeObjective7 as SETree7
from src.opt.model_based.EDA.DADO.tree import TreeDADO
from src.opt.model_based.EDA.joint_EDA import JointEDA


def init_worker(gpu_queue, allowed_gpus):
    """Initialize worker process for multiprocessing pool."""
    global _worker_gpu
    _worker_gpu = gpu_queue.get()  # GPU ID from allowed list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu)
    # print(f"Worker PID {os.getpid()} assigned GPU {_worker_gpu}")

def calculate_epoch_stats(values_history):
    """Calculate mean and standard deviation of objective values for each epoch."""
    means = []
    stds = []
    
    for values in values_history:
        means.append(float(values.mean()))
        stds.append(float(values.std()))
        
    return np.array(means), np.array(stds)

def run_experiment_wrapper(params):
    """Wrapper function for multiprocessing."""
    temp_Q, lr, obj_name, negate, seeds, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir = params
    # print(f"Worker {os.getpid()} starting: GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}, Q={temp_Q}, lr={lr}")
    obj = globals()[obj_name](negate=negate)
    return run_experiment(temp_Q, lr, obj, seeds, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir)

def run_joint_experiment(temp_Q, lr, obj, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir):
    """Run only joint EDA experiment for given parameters and seed."""
    intermediate_file = f"{outdir}/results/joint_Q{temp_Q}-lr{lr}_s{seed}.dict"
    if os.path.exists(intermediate_file):
        return mlxu.load_pickle(intermediate_file)
    
    joint_eda = JointEDA(
        use_autoregressive=True,
        n_variables=obj.L,
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=n_epochs,
        num_inner_iter=n_inner_iter,
        num_MC_samples=n_mc_samples,
        learning_rate=lr,
        temp_Q=temp_Q,
        n_layers=n_layers,
        dim_ff=dim_ff,
    )
    d_joint = joint_eda.precompute()
    
    stats = {
        'values': d_joint['values'],
        'max': d_joint['values'].max(axis=-1),
        'min': d_joint['values'].min(axis=-1),
        'ci_upper': jnp.quantile(d_joint['values'], 0.975, axis=-1),
        'ci_lower': jnp.quantile(d_joint['values'], 0.025, axis=-1),
        'weights': d_joint['weights'],
        'mean': calculate_epoch_stats(d_joint['values'])[0],
        'std': calculate_epoch_stats(d_joint['values'])[1],
        'ess': d_joint['ess'],
        'losses': d_joint['losses'],
        'mode': joint_eda.get_mode(),
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed,
        'method': 'joint',
    }
    mlxu.save_pickle(stats, intermediate_file)
    return stats

def run_fd_experiment(temp_Q, lr, obj, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir):
    """Run only FD EDA experiment for given parameters and seed."""
    intermediate_file = f"{outdir}/results/fd_Q{temp_Q}-lr{lr}_s{seed}.dict"
    if os.path.exists(intermediate_file):
        return mlxu.load_pickle(intermediate_file)
        
    tree_eda = TreeDADO(
        n_variables=obj.L,
        n_states=obj.D,
        objective_fn=obj,
        seed=seed,
        num_epochs=n_epochs,
        num_inner_iter=n_inner_iter,
        num_MC_samples=n_mc_samples,
        learning_rate=lr,
        temp_Q=temp_Q,
        n_layers=n_layers,
        dim_ff=dim_ff,
    )
    d_fd = tree_eda.precompute()
    
    stats = {
        'values': d_fd['values'],
        'max': d_fd['values'].max(axis=-1),
        'min': d_fd['values'].min(axis=-1),
        'ci_upper': jnp.quantile(d_fd['values'], 0.975, axis=-1),
        'ci_lower': jnp.quantile(d_fd['values'], 0.025, axis=-1),
        'weights': d_fd['weights'],
        'mean': calculate_epoch_stats(d_fd['values'])[0],
        'std': calculate_epoch_stats(d_fd['values'])[1],
        'ess': d_fd['ess'],
        'losses': d_fd['losses'],
        'mode': tree_eda.get_mode(),
        'temp_Q': temp_Q,
        'lr': lr,
        'seed': seed,
        'method': 'fd',
    }
    mlxu.save_pickle(stats, intermediate_file)
    return stats

def method_specific_wrapper(params):
    """Wrapper for method-specific experiment functions."""
    temp_Q, lr, method, obj_name, negate, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir = params
    # print(f"Method worker {os.getpid()} starting: GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}, method={method}")
    obj = globals()[obj_name](negate=negate)
    
    if method == 'joint':
        return run_joint_experiment(temp_Q, lr, obj, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir)
    elif method == 'fd':
        return run_fd_experiment(temp_Q, lr, obj, seed, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir)
    else:
        raise ValueError(f"Unknown method: {method}")

def aggregate_method_results(method, temp_Q, lr, sweep_seeds, additional_seeds, outdir):
    """Aggregate results from sweep + additional replicates for a specific method and hyperparameters."""
    # all_seeds = sweep_seeds + additional_seeds
    # NOTE: use only additional seeds.
    
    values, maxes, mins, weights_list = [], [], [], []
    means, stds, ess_list = [], [], []
    ci_lower, ci_upper = [], []
    modes = []
    print(f"Globbing {outdir}/results/{method}_Q{'*'}-lr{'*'}_s*.dict")
    fnames = glob(f"{outdir}/results/{method}_Q{'*'}-lr{'*'}_s*.dict")
    print(f"Found {len(fnames)} {method} files")
    
    for fname in fnames:
        if os.path.exists(fname):
            stats = mlxu.load_pickle(fname)
            values.append(stats['values'])
            ci_lower.append(stats['ci_lower'])
            ci_upper.append(stats['ci_upper'])
            maxes.append(stats['max'])
            mins.append(stats['min'])
            weights_list.append(stats['weights'])
            means.append(stats['mean'])
            stds.append(stats['std'])
            ess_list.append(stats['ess'])
            modes.append(stats['mode'])
    
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
        'values': values_stacked,
        'ci_lower': ci_lower_stacked,
        'ci_upper': ci_upper_stacked,
        'ess': ess_list,
        'mode': jnp.mean(jnp.array([m[-1] for m in modes])) if modes else None,
        'selection': means_stacked.mean(axis=0)[-1]  # Final epoch mean
    }

def run_experiment(temp_Q, lr, obj, seeds, n_epochs, n_inner_iter, n_mc_samples, n_layers, dim_ff, outdir):
    """Run an experiment with a given Q and lr."""
    if os.path.exists(f"{outdir}/results/Q{temp_Q}-lr{lr}.dict"):
        stats = mlxu.load_pickle(f"{outdir}/results/Q{temp_Q}-lr{lr}.dict")
        return (float(stats['selection_joint']), float(stats['selection_fd']))
    
    # Run experiments and save intermediate results for each replicate
    for i, seed in enumerate(seeds):
        # Check if intermediate result already exists
        intermediate_file = f"{outdir}/results/Q{temp_Q}-lr{lr}_s{seed}.dict"
        if os.path.exists(intermediate_file):
            continue
            
        joint_eda = JointEDA(
            use_autoregressive=True,
            n_variables=obj.L,
            n_states=obj.D,
            objective_fn=obj,
            seed=seed,
            num_epochs=n_epochs,
            num_inner_iter=n_inner_iter,
            num_MC_samples=n_mc_samples,
            learning_rate=lr,
            temp_Q=temp_Q,
            n_layers=n_layers,
            dim_ff=dim_ff,
        )
        d_joint = joint_eda.precompute()
        
        tree_eda = TreeDADO(
            n_variables=obj.L,
            n_states=obj.D,
            objective_fn=obj,
            seed=seed,
            num_epochs=n_epochs,
            num_inner_iter=n_inner_iter,
            num_MC_samples=n_mc_samples,
            learning_rate=lr,
            temp_Q=temp_Q,
            n_layers=n_layers,
            dim_ff=dim_ff,
        )
        d_fd = tree_eda.precompute()
        
        # Save intermediate result for this replicate
        intermediate_stats = {
            'joint_max': d_joint['values'].max(axis=-1),
            'joint_min': d_joint['values'].min(axis=-1),
            'joint_weights': d_joint['weights'],
            'joint_mean': calculate_epoch_stats(d_joint['values'])[0],
            'joint_std': calculate_epoch_stats(d_joint['values'])[1],
            'joint_values': d_joint['values'],
            'joint_ess': d_joint['ess'],
            'joint_losses': d_joint['losses'],
            'fd_max': d_fd['values'].max(axis=-1),
            'fd_min': d_fd['values'].min(axis=-1),
            'fd_weights': d_fd['weights'],
            'fd_mean': calculate_epoch_stats(d_fd['values'])[0],
            'fd_std': calculate_epoch_stats(d_fd['values'])[1],
            'fd_values': d_fd['values'],
            'fd_ess': d_fd['ess'],
            'fd_losses': d_fd['losses'],
            'temp_Q': temp_Q,
            'lr': lr,
            'seed': seed,
            'rep_idx': i,
        }
        mlxu.save_pickle(intermediate_stats, intermediate_file)
    # Load and aggregate intermediate results
    joint_means, joint_stds, joint_ess = [], [], []
    joint_maxes, joint_mins, joint_Q = [], [], []
    joint_values, joint_losses = [], []
    fd_means, fd_stds, fd_ess = [], [], []
    fd_maxes, fd_mins, fd_Q = [], [], []
    fd_values, fd_losses = [], []
    
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
        joint_losses.append(rep_stats['joint_losses'])
        joint_values.append(rep_stats['joint_values'])
        
        # Aggregate fd results
        fd_maxes.append(rep_stats['fd_max'])
        fd_mins.append(rep_stats['fd_min'])
        fd_Q.append(rep_stats['fd_weights'])
        fd_means.append(rep_stats['fd_mean'])
        fd_stds.append(rep_stats['fd_std'])
        fd_ess.append(rep_stats['fd_ess'])
        fd_losses.append(rep_stats['fd_losses'])
        fd_values.append(rep_stats['fd_values'])

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
    joint_losses_stacked = np.stack(joint_losses, axis=0) # (n_reps, n_epochs)
    joint_values_stacked = np.stack(joint_values, axis=0) # (n_reps, n_epochs, L)

    fd_means_stacked = np.stack(fd_means, axis=0) # (n_reps, n_epochs)
    fd_stds_stacked = np.stack(fd_stds, axis=0) # (n_reps, n_epochs)
    fd_means_mean = fd_means_stacked.mean(axis=0) # (n_epochs,)
    fd_stds_mean = fd_stds_stacked.mean(axis=0) # (n_epochs,)
    selection_fd = fd_means_mean[-1] # scalar
    fd_maxes_stacked = np.stack(fd_maxes, axis=0) # (n_reps, n_epochs)
    fd_mins_stacked = np.stack(fd_mins, axis=0) # (n_reps, n_epochs)
    fd_Q_stacked = np.stack(fd_Q, axis=0).mean(axis=-2) # (n_reps, n_epochs, L)
    fd_losses_stacked = np.stack(fd_losses, axis=0) # (n_reps, n_epochs)
    fd_values_stacked = np.stack(fd_values, axis=0) # (n_reps, n_epochs, L)

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
        'joint_losses': joint_losses_stacked,
        'joint_mode': joint_eda.get_mode(),
        'joint_values': joint_values_stacked,
        'fd_maxes': fd_maxes_stacked,
        'fd_mins': fd_mins_stacked,
        'fd_means_stacked': fd_means_stacked,
        'fd_stds_stacked': fd_stds_stacked,
        'fd_means_mean': fd_means_mean,
        'fd_stds_mean': fd_stds_mean,
        'fd_ess': fd_ess,
        'fd_weights': fd_Q_stacked,
        'fd_losses': fd_losses_stacked,
        'fd_values': fd_values_stacked,
        'fd_mode': tree_eda.get_mode(),
        'temp_Q': temp_Q,
        'lr': lr,
        'selection_joint': selection_joint,
        'selection_fd': selection_fd
    }
    filename = f"Q{temp_Q}-lr{lr}.dict"
    mlxu.save_pickle(stats_dict, f"{outdir}/results/{filename}")

    return (
        float(selection_joint),
        float(selection_fd)
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tree-structured synthetic experiments')
    
    # Basic arguments
    available_objs = [
        'Tree1', 'Tree2', 'Tree3', 'Tree4', 'Tree5', 
        'SETree1', 'SETree2', 'SETree3', 'SETree4', 'SETree5', 'SETree6', 'SETree7',
        'Chain1', 'Chain2'
    ]
    parser.add_argument(
        '--obj', 
        type=str, 
        default='SETree4',
        help=f'Objective to run. Available: {", ".join(available_objs)}. '
             f'Default: Tree1'
    )
    parser.add_argument(
        '--negate',
        action='store_true',
        help='Negate the objective'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory name. Results will be saved to experiments/search_synthetic/tree/outdir/<obj_name>/. '
             'Default: uses timestamp'
    )
    parser.add_argument(
        '--plot_only',
        action='store_true',
        help='Only plot the results, do not run the experiments'
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
    parser.add_argument('--n_pool', type=int, default=10, help='Number of processes to use in parallel. Usually RAM-constrained.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use (e.g., "1,2,3,4,5,6,7")')
    
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for optimization')
    parser.add_argument('--n_inner_iter', type=int, default=1, help='Number of inner iterations for optimization')
    parser.add_argument('--n_mc_samples', type=int, default=100, help='Number of MC samples for optimization')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for optimization models')
    parser.add_argument('--dim_ff', type=int, default=64, help='Dimension of feedforward layer for optimization models')

    return parser.parse_args()


def main():
    """Main execution function."""
    st = time()
    # Set multiprocessing start method to avoid JAX/CUDA issues
    mp.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    args = parse_args()

    # Parse allowed GPUs
    allowed_gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    print(f"Using GPUs: {allowed_gpus}")

    print(f"Running experiments for objective: {args.obj}")
    obj = globals()[args.obj](negate=args.negate)
    # if obj.L < 8:
    obj.exhaustive_search() # implemented tree MP
    print(f"\tBest solution: {obj.best_val:.2f} at {obj.best_x}")
    # print(f"Eval best: {obj.query(obj.best_x[None, :])}")

    Q_sweep = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 2.0, 4., 8.]
    lr_sweep = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    print(f"\tSweeping over {len(Q_sweep)} Q values and {len(lr_sweep)} learning rate values = {len(Q_sweep) * len(lr_sweep)} total.")
    Q_lr_sweep = np.array(np.meshgrid(Q_sweep, lr_sweep)).T.reshape(-1, 2)
    # print(f"Cartesian product of Q and lr: {Q_lr_sweep.shape}")

    # Create output directory
    if args.outdir is None:
        args.outdir = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"experiments/synthetic/{args.outdir}/{args.obj}"
    outdir = outdir + '-neg' if args.negate else outdir
    os.makedirs(outdir + '/results', exist_ok=True)
    print(f"Results will be saved to: {outdir}")

    # Save configuration
    config_path = f"{outdir}/config.dict"
    with open(config_path, 'wb') as f:
        pickle.dump(vars(args), f)
    print(f"Configuration saved to: {config_path}")

    # Create parameter tuples for multiprocessing
    np_rng = np.random.default_rng(args.seed)   # create generator with fixed seed
    seeds = np_rng.integers(
        low=0, high=np.iinfo(np.int32).max, size=args.n_reps
        ).tolist()
    params_list = []
    for temp_Q, lr in Q_lr_sweep:
        params_list.append((temp_Q, lr, args.obj, args.negate, seeds, args.n_epochs, args.n_inner_iter, args.n_mc_samples, args.n_layers, args.dim_ff, outdir))

    if not args.plot_only:
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
        results = mlxu.load_pickle(f"{outdir}/results.pkl")

    # find best Q, lr for each of joint, FGO by max mean
    joint_selection = jnp.array([r[0] for r in results])
    print(f"Joint selection: min {jnp.min(joint_selection)}, max {jnp.max(joint_selection)}, mean {jnp.mean(joint_selection)}, std {jnp.std(joint_selection)}")
    fd_selection = jnp.array([r[1] for r in results])
    print(f"FGO selection: min {jnp.min(fd_selection)}, max {jnp.max(fd_selection)}, mean {jnp.mean(fd_selection)}, std {jnp.std(fd_selection)}")
    joint_ind = jnp.argmax(joint_selection)
    fd_ind = jnp.argmax(fd_selection)
    joint_Q_star, joint_lr_star = Q_lr_sweep[joint_ind]
    fd_Q_star, fd_lr_star = Q_lr_sweep[fd_ind]
    print(f"Best joint Q, lr: {joint_Q_star}, {joint_lr_star}, {joint_selection[jnp.argmax(joint_selection)]}")
    print(f"Best FGO Q, lr: {fd_Q_star}, {fd_lr_star}, {fd_selection[jnp.argmax(fd_selection)]}")
    
    # # Create comprehensive plot showing all Q/lr combinations
    # plt.figure(figsize=(12, 8))
    
    # if hasattr(obj, 'best_val') and obj.best_val is not None:
    #     plt.axhline(y=obj.best_val, color='black', linestyle='-', label='Max')
    
    # # Plot all joint results
    # for i, (temp_Q, lr) in enumerate(Q_lr_sweep):
    #     filename = f"Q{temp_Q}-lr{lr}.dict"
    #     stats = mlxu.load_pickle(f"{outdir}/results/{filename}")
    #     joint_means_all = stats['joint_means_mean']
    #     joint_stds_all = stats['joint_stds_mean']
        
    #     if i == 0:  # Only add label for first plot
    #         plt.plot(joint_means_all, color='blue', alpha=0.2, label='Joint EDA (all)')
    #     else:
    #         plt.plot(joint_means_all, color='blue', alpha=0.2)
    
    # # Plot all FD results
    # for i, (temp_Q, lr) in enumerate(Q_lr_sweep):
    #     filename = f"Q{temp_Q}-lr{lr}.dict"
    #     stats = mlxu.load_pickle(f"{outdir}/results/{filename}")
    #     fd_means_all = stats['fd_means_mean']
    #     fd_stds_all = stats['fd_stds_mean']
        
    #     if i == 0:  # Only add label for first plot
    #         plt.plot(fd_means_all, color='red', alpha=0.2, label='FGO (Tree, all)')
    #     else:
    #         plt.plot(fd_means_all, color='red', alpha=0.2)
    
    # # Highlight best results
    # plt.plot(joint_means, color='blue', linewidth=2, label='Joint EDA (best)')
    # plt.plot(fd_means, color='red', linewidth=2, label='FGO (Tree, best)')
    
    # plt.title(f"{args.obj} Optimization - All Hyperparameters")
    # plt.xlabel('Training Epoch')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.savefig(f"{outdir}/all_results.png")
    # plt.close()
    
    print(f'Will now replicate ({args.n_final_reps}x) the best results over seeds...')
    additional_seeds = np_rng.integers(
        low=0, high=np.iinfo(np.int32).max, size=args.n_final_reps
        ).tolist()
    
    if not args.plot_only:
        # Create parameter lists for only the methods/hyperparams we need
        joint_params = [(joint_Q_star, joint_lr_star, 'joint', args.obj, args.negate, seed, args.n_epochs, args.n_inner_iter, args.n_mc_samples, args.n_layers, args.dim_ff, outdir) for seed in additional_seeds]
        fd_params = [(fd_Q_star, fd_lr_star, 'fd', args.obj, args.negate, seed, args.n_epochs, args.n_inner_iter, args.n_mc_samples, args.n_layers, args.dim_ff, outdir) for seed in additional_seeds]
        all_params = joint_params + fd_params
        
        with mp.Manager() as manager_additional:
            # Create GPU assignment queue for additional experiments
            gpu_queue_additional = manager_additional.Queue()
            # Cycle through allowed GPUs for pool processes
            for i in range(args.n_pool):
                gpu_queue_additional.put(allowed_gpus[i % len(allowed_gpus)])

            with mp.Pool(processes=args.n_pool, initializer=init_worker, initargs=(gpu_queue_additional, allowed_gpus)) as pool:
                additional_results = pool.map(method_specific_wrapper, all_params)
    
    # Aggregate results combining sweep + additional replicates
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

    # Update plotting to use aggregated results
    joint_means = joint_aggregated['means_mean']
    joint_stds = joint_aggregated['stds_mean']
    joint_maxes = joint_aggregated['maxes']
    joint_mins = joint_aggregated['mins']
    joint_ci_lower = joint_aggregated['ci_lower'].mean(axis=0)
    joint_ci_upper = joint_aggregated['ci_upper'].mean(axis=0)
    joint_mode = joint_aggregated['mode']
    
    fd_means = fd_aggregated['means_mean']
    fd_stds = fd_aggregated['stds_mean']
    fd_maxes = fd_aggregated['maxes']
    fd_mins = fd_aggregated['mins']
    fd_ci_lower = fd_aggregated['ci_lower'].mean(axis=0)
    fd_ci_upper = fd_aggregated['ci_upper'].mean(axis=0)
    fd_mode = fd_aggregated['mode']
    
    # Create final plot with aggregated results (sweep + additional replicates)
    fig, ax = plt.subplots(figsize=(10, 6))
    # if hasattr(obj, 'best_val') and obj.best_val is not None:
    #     plt.axhline(y=obj.best_val, color='black', linestyle='--', lw=2, label='Global Maximum')
    pstring = f"$\\mathbf{{P = {p_two_sided:.2e}}}$"
    if p_two_sided >= 0.01:
        pstring = f"$\\mathbf{{P = {p_two_sided:.2f}}}$"
    ax.text( # change 2nd coordinate to make higher/lower on plot
        0.95, 0.2, pstring, 
        transform=ax.transAxes,  # use axes coords (0–1)
        ha="right", va="center", 
        fontsize=20
    )
    
    plt.plot(joint_means, label=f'EDA Sample Mean', color="#377EB8", lw=2)
    plt.fill_between(range(len(joint_means)), joint_ci_lower, joint_ci_upper, alpha=0.3, color="#377EB8", label='EDA 95% CI')
    
    plt.plot(fd_means, label=f'DADO Sample Mean', color="#E41A1C", lw=2)
    plt.fill_between(range(len(fd_means)), fd_ci_lower, fd_ci_upper, alpha=0.3, color="#E41A1C", label='DADO 95% CI')
    
    plt.title(f"L = {obj.L}", fontsize=30)
    plt.xlabel('Training Iteration', fontsize=26, labelpad=15); plt.xticks(fontsize=20)
    plt.ylabel('f(x)', fontsize=26, labelpad=15); plt.yticks(fontsize=20)
    plt.savefig(f"{outdir}/tree_rep_results_{obj.L}L.png", dpi=600, bbox_inches='tight')
    plt.legend(fontsize=20, loc='upper left')
    plt.savefig(f"{outdir}/tree_rep_results_{obj.L}L_legend.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Final aggregated results saved to {outdir}/tree_rep_results_{obj.L}L.png")

if __name__ == "__main__":
    main()
