#!/usr/bin/env python3
"""
AF_FGM_fit_cv.py

Cross-validation grid search for AF-FGM models on real-world datasets.
"""

import argparse
import itertools
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import sys

# Environment setup--optional, but good if you intend to multiprocess.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.)

# Check if working directory ends with 'DADO', then append to sys.path
cwd = os.getcwd()
if cwd.endswith('DADO'):
    sys.path.append(cwd)
else:
    raise ValueError(f"Working directory {cwd} does not end with 'DADO'")

from src.problems.real.oracle import OracleObjective
from src.decomposition.graphs import GeneralizedGraph
from src.decomposition.fgm.mlp.base import MLPGeneralizedFunctionalGraph
from src.problems.real.protein_structure import residue_contact_map

# Structure paths mapping
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

def fit_af_fgm_single(obj_name, structure_path, train_indices, val_indices, 
                     hidden_dims, batch_size, gr_batch_size, embedding_dim,
                     num_epochs, num_inner_iter, learning_rate, seed, use_wt: bool = False):
    """Fit a single AF-FGM model and return train/val metrics"""
    obj = OracleObjective(obj_name=obj_name)
    wt_int = None
    if use_wt: wt_int = obj.obj.str_to_int(obj.obj.WT).squeeze()
    
    # Get AF-based graph structure
    edges_AF, _ = residue_contact_map(
        structure_path, 
        obj.WT_seq,
        edgelist=True,
        verbose=False,
    )
    gg = GeneralizedGraph(len(obj.D), edges_AF, verbose=False)
    
    # Create functional graph
    fg = MLPGeneralizedFunctionalGraph(
        gg, n_states_list=obj.D, 
        hidden_dims=hidden_dims, 
        batch_size=batch_size,
        gr_batch_size=gr_batch_size,
        embedding_dim=embedding_dim,
        rngs=jax.random.PRNGKey(seed),
        wt=wt_int,
    )
    
    # Train on training set
    losses = MLPGeneralizedFunctionalGraph.fit(
        fg,
        obj.genotypes[train_indices, :],
        obj.values[train_indices], 
        n_epochs=num_epochs,
        n_iter_inner=num_inner_iter,
        learning_rate=learning_rate,
        seed=seed,
    )
    
    # Evaluate on train and validation sets
    train_pred = fg.evaluate_batched(obj.genotypes[train_indices, :])
    train_true = obj.values[train_indices]
    train_mse = float(jnp.mean((train_pred - train_true)**2))
    train_r2 = float(r2_score(np.array(train_true), np.array(train_pred)))
    train_sr = float(spearmanr(np.array(train_true), np.array(train_pred), axis=None).statistic)
    
    val_pred = fg.evaluate_batched(obj.genotypes[val_indices, :])
    val_true = obj.values[val_indices]
    val_mse = float(jnp.mean((val_pred - val_true)**2))
    val_r2 = float(r2_score(np.array(val_true), np.array(val_pred)))
    val_sr = float(spearmanr(np.array(val_true), np.array(val_pred), axis=None).statistic)
    
    return train_mse, train_r2, train_sr, val_mse, val_r2, val_sr

def _fit_af_fgm_fold_worker(args):
    """Worker function for multiprocessing AF-FGM fold evaluation"""
    (fold_idx, train_idx, val_idx, obj_name, structure_path, hidden_dims, 
     batch_size, gr_batch_size, embedding_dim, n_epochs, n_inner, lr, seed, use_wt) = args
    train_mse, train_r2, train_sr, val_mse, val_r2, val_sr = fit_af_fgm_single(
        obj_name, structure_path, train_idx, val_idx,
        hidden_dims, batch_size, gr_batch_size, embedding_dim,
        n_epochs, n_inner, lr, seed + fold_idx, use_wt
    )
    return {
        'train_mse': train_mse, 'train_r2': train_r2,
        'train_sr': train_sr,
        'val_mse': val_mse, 'val_r2': val_r2,
        'val_sr': val_sr,
        'fold_idx': fold_idx
    }

def af_fgm_cv_grid_search(obj_name, structure_path, param_grid, k=5, seed=4002, use_wt: bool = False):
    """
    Perform k-fold cross validation grid search for AF-FGM models
    
    Args:
        obj_name: Objective name
        structure_path: Structure file path
        param_grid: Dict with keys 'learning_rates', 'n_inner_iter', 'n_epochs', 'hidden_dims', 
                   'batch_size', 'gr_batch_size', 'embedding_dim'
        k: Number of folds
        seed: Random seed
    
    Returns:
        List of result dictionaries with metrics and parameters
    """
    results = []
    
    # Create multiprocessing pool once, reuse for all evaluations
    with mp.Pool(processes=k) as pool:
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            param_grid['learning_rates'],
            param_grid['n_inner_iter'], 
            param_grid['n_epochs'],
            param_grid['hidden_dims']
        ))
        
        obj = OracleObjective(obj_name=obj_name)
        n_samples = obj.values.size
        
        # Create k-fold splits
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(kf.split(range(n_samples)))
        
        print(f"Running {len(param_combinations)} parameter combinations with {k}-fold CV...")
        
        for lr, n_inner, n_epochs, hidden_dims in tqdm(param_combinations, desc="Param combinations"):
            batch_size = param_grid['batch_size']
            gr_batch_size = param_grid['gr_batch_size']
            embedding_dim = param_grid['embedding_dim']
            
            # Prepare arguments for each fold
            fold_args = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                args = (fold_idx, train_idx, val_idx, obj_name, structure_path, 
                       hidden_dims, batch_size, gr_batch_size, embedding_dim,
                       n_epochs, n_inner, lr, seed, use_wt)
                fold_args.append(args)
            
            # Run k folds in parallel
            fold_results_raw = pool.map(_fit_af_fgm_fold_worker, fold_args)
            
            # Filter out any failed results and extract metrics
            fold_results = []
            for result in fold_results_raw:
                if result is not None:
                    fold_results.append({
                        'train_mse': result['train_mse'], 
                        'train_r2': result['train_r2'],
                        'train_sr': result['train_sr'],
                        'val_mse': result['val_mse'], 
                        'val_r2': result['val_r2'],
                        'val_sr': result['val_sr'],
                    })
            
            if len(fold_results) < k:
                print(f"Warning: Only {len(fold_results)}/{k} folds completed successfully")
            
            # Average across folds
            if fold_results:  # Only compute if we have at least some results
                avg_result = {
                    'objective': obj_name,
                    'model_type': 'af_fgm',
                    'learning_rate': lr,
                    'n_inner_iter': n_inner,
                    'n_epochs': n_epochs,
                    'hidden_dims': hidden_dims,
                    'batch_size': batch_size,
                    'gr_batch_size': gr_batch_size,
                    'embedding_dim': embedding_dim,
                    'train_mse_mean': np.mean([r['train_mse'] for r in fold_results]),
                    'train_mse_std': np.std([r['train_mse'] for r in fold_results]),
                    'train_r2_mean': np.mean([r['train_r2'] for r in fold_results]),
                    'train_r2_std': np.std([r['train_r2'] for r in fold_results]),
                    'train_sr_mean': np.mean([r['train_sr'] for r in fold_results]),
                    'train_sr_std': np.std([r['train_sr'] for r in fold_results]),
                    'val_mse_mean': np.mean([r['val_mse'] for r in fold_results]),
                    'val_mse_std': np.std([r['val_mse'] for r in fold_results]),
                    'val_r2_mean': np.mean([r['val_r2'] for r in fold_results]),
                    'val_r2_std': np.std([r['val_r2'] for r in fold_results]),
                    'val_sr_mean': np.mean([r['val_sr'] for r in fold_results]),
                    'val_sr_std': np.std([r['val_sr'] for r in fold_results]),
                }
                results.append(avg_result)
    
    return results

def get_default_param_grid():
    """Get default parameter grid for AF-FGM models"""
    
    # Common parameters
    hidden_dims_options = [
        [16, 16],
        [128, 16],
        [128, 128, 16],
    ]
    # NOTE: based on amyloid, 1e-2 seems too high. 
    learning_rates = [
        1e-3, 
        1e-4,
    ]
    n_inner_iter_options = [ # don't change this...
        1, 
    ]
    n_epochs = [
        5000, 
        50000, 
    ]
    
    af_fgm_param_grid = {
        'hidden_dims': hidden_dims_options,
        'learning_rates': learning_rates,
        'n_inner_iter': n_inner_iter_options,
        'n_epochs': n_epochs,
        'batch_size': 2**10,  # 1024 as specified
        'gr_batch_size': -1,  # Full graph as specified
        'embedding_dim': 4
    }
    
    return af_fgm_param_grid

def find_top_configs(results, top_k=5):
    """Find top-k hyperparameter configurations for MSE and R²"""
    if not results:
        return [], []

    # Sort results: lower MSE is better, higher R² is better
    sorted_by_mse = sorted(results, key=lambda x: x['val_mse_mean'])[:top_k]
    sorted_by_r2 = sorted(results, key=lambda x: x['val_r2_mean'], reverse=True)[:top_k]

    return sorted_by_mse, sorted_by_r2


def config_key(config):
    """Returns a tuple that uniquely identifies a config (must be hashable)"""
    return (tuple(config['hidden_dims']),  # Convert list to tuple
            config['learning_rate'], 
            config['n_inner_iter'], 
            config['n_epochs'])


def print_top_configs_interleaved(sorted_by_mse, sorted_by_r2, obj_name):
    """Prints the top-k configs interleaved from both MSE and R² metrics"""
    print(f"\n{'='*60}")
    print(f"TOP HYPERPARAMETER CONFIGURATIONS FOR {obj_name.upper()}")
    print(f"{'='*60}")

    seen = set()

    for i in range(max(len(sorted_by_mse), len(sorted_by_r2))):
        for config_list, metric_name in [(sorted_by_mse, "MSE"), (sorted_by_r2, "R²")]:
            if i < len(config_list):
                config = config_list[i]
                key = config_key(config)
                if key not in seen:
                    seen.add(key)
                    print(f"\n🔹 Rank {i+1} by {metric_name}:")
                    print(f"   Hidden Dims: {config['hidden_dims']}")
                    print(f"   Learning Rate: {config['learning_rate']}")
                    print(f"   N Inner Iter: {config['n_inner_iter']}")
                    print(f"   N Epochs: {config['n_epochs']}")
                    print(f"   Validation MSE: {config['val_mse_mean']:.6f} ±{config['val_mse_std']:.6f}")
                    print(f"   Validation R²: {config['val_r2_mean']:.6f} ±{config['val_r2_std']:.6f}")
                    print(f"   Validation Spearman: {config['val_sr_mean']:.6f} ±{config['val_sr_std']:.6f}")

def main():
    parser = argparse.ArgumentParser(description='AF-FGM Cross-Validation Grid Search')
    parser.add_argument('obj', choices=list(STRUCTURE_PATHS_DICT.keys()),
                        help='Objective function to evaluate')
    
    args = parser.parse_args()
    
    obj_name = args.obj
    # NOTE: this file is also available in src/problems/real/data/af3_structures/.
    structure_path = os.path.join(
        os.environ["DIR_DATA"], 
        "af3_structures", 
        STRUCTURE_PATHS_DICT[obj_name]
    )
    
    print(f"Running AF-FGM cross-validation for objective: {obj_name}")
    print(f"Structure path: {structure_path}")
    
    # Get parameter grid
    param_grid = get_default_param_grid()
    
    print(f"\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    total_combinations = (len(param_grid['hidden_dims']) * 
                         len(param_grid['learning_rates']) * 
                         len(param_grid['n_inner_iter']) * 
                         len(param_grid['n_epochs']))
    
    # Run cross-validation
    results = af_fgm_cv_grid_search(
        obj_name=obj_name,
        structure_path=structure_path,
        param_grid=param_grid,
        k=5,  # 5-fold CV
        seed=49502, #42
        use_wt=False, # NOTE: WTG; seems worse.
    )
    
    print(f"\nCompleted {len(results)} model evaluations")
    
    top_mse, top_r2 = find_top_configs(results, top_k=5)
    print_top_configs_interleaved(top_mse, top_r2, obj_name=obj_name)
    
    # TODO: change to normal mlp, nonneg, scaled.
    output_dir = str(Path(__file__).parent / 'cv' / f'GEmlp_{obj_name}_')
    results_file = output_dir + 'results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    
    print(f"\nCross-validation complete for {obj_name}!")

if __name__ == "__main__":
    main()