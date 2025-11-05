import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.problems.real.base import sequence_to_index, alphabet_to_integer
from src.problems.real.nonbinary.base import NonBinaryObjective

class GB1_55(NonBinaryObjective):
    """
    GB1 dataset of 55 positions with 20 states each. Covers WT+ all single, double mutants.
    Full dataset is https://ars.els-cdn.com/content/image/1-s2.0-S0960982214012688-mmc2.xlsx.
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/olson14/gb1_55.npz"
        # NOTE: omits the start codon.
        # NOTE: doesn't undergo much conformational change when bound, so probably ok to use unbound structure.
        self.WT = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
        self.active_inds = None
        # TODO: get structure distogram for WT.
        self.str_to_int = lambda x: alphabet_to_integer(x)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            xlsx_path = f"{os.environ['DIR_DATA']}/olson14/1-s2.0-S0960982214012688-mmc2.xlsx"
            if not os.path.exists(xlsx_path):
                os.system(f"wget https://ars.els-cdn.com/content/image/1-s2.0-S0960982214012688-mmc2.xlsx -O {xlsx_path}")
            df = process_gb1_data(xlsx_path)
            x = self.str_to_int(df["x"].values)
            values = np.array(df["y"].values)
            y = (values - values.min()) / (values.max() - values.min())
            np.savez(filepath, genotypes=x, values=y)

        self.genotypes = jnp.array(x)
        values = jnp.array(y)
        assert values.min() >= 0 and values.max() <= 1, "Values should be normalized to [0, 1]"
        best_x = self.genotypes[values.argmax()]

        def query_fn(x):
            ind = sequence_to_index(x, self.genotypes)
            return jax.lax.select(
                ind == -1,
                -jnp.inf,
                values[ind]
            )

        # Initialize the parent class -- max=True (defaults)
        super().__init__(
            D=[20]*55,
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "GB1_55"

        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")


def process_gb1_data(excel_file_path):
    """
    Process GB1 data including double mutants, single mutants, and wild type.
    Replicates mavenn functionality (https://mavenn.readthedocs.io/en/latest/datasets/dataset_gb1.html)
    but extends to include all variant types.
    """
    df = pd.read_excel(excel_file_path)
    
    # Extract data sections (skip header rows)
    double_mut_df = df.iloc[2:, 1:11].copy()
    double_mut_df.columns = ['Mut1_WT_aa', 'Mut1_Position', 'Mut1_Mutation', 
                            'Mut2_WT_aa', 'Mut2_Position', 'Mut2_Mutation',
                            'Input_Count', 'Selection_Count', 'Mut1_Fitness', 'Mut2_Fitness']
    double_mut_df = double_mut_df.dropna()
    
    single_mut_df = df.iloc[2:, 13:18].copy()
    single_mut_df.columns = ['WT_aa', 'Position', 'Mutation', 'Input_Count', 'Selection_Count']
    single_mut_df = single_mut_df.dropna()
    
    wt_df = df.iloc[2:, 20:22].copy()
    wt_df.columns = ['Input_Count', 'Selection_Count'] 
    wt_df = wt_df.dropna()
    
    # Reconstruct wild-type sequence from double mutants
    wt_1_df = double_mut_df[['Mut1_Position', 'Mut1_WT_aa']].copy()
    wt_1_df.columns = ['pos', 'aa']
    wt_2_df = double_mut_df[['Mut2_Position', 'Mut2_WT_aa']].copy()
    wt_2_df.columns = ['pos', 'aa']
    wt_seq_df = pd.concat([wt_1_df, wt_2_df], axis=0).drop_duplicates().sort_values(by='pos').reset_index(drop=True)
    
    # Construct wild-type sequence
    wt_seq = ''.join(wt_seq_df['aa'])
    
    # Get WT baseline for log2 enrichment calculation
    wt_in_ct = wt_df['Input_Count'].iloc[0]
    wt_out_ct = wt_df['Selection_Count'].iloc[0]
    wt_baseline = np.log2((wt_out_ct + 1) / (wt_in_ct + 1))
    
    # Process each variant type
    all_data = []
    
    # 1. Process double mutants (dist=2)
    pos1s = double_mut_df['Mut1_Position'].values - 2  # Convert to 0-indexed
    pos2s = double_mut_df['Mut2_Position'].values - 2
    aa1s = double_mut_df['Mut1_Mutation'].values
    aa2s = double_mut_df['Mut2_Mutation'].values
    
    for i, (pos1, aa1, pos2, aa2) in enumerate(zip(pos1s, aa1s, pos2s, aa2s)):
        mut_seq_list = list(wt_seq)
        mut_seq_list[pos1] = aa1
        mut_seq_list[pos2] = aa2
        mut_seq = ''.join(mut_seq_list)
        
        in_ct = double_mut_df['Input_Count'].iloc[i]
        out_ct = double_mut_df['Selection_Count'].iloc[i]
        y = np.log2((out_ct + 1) / (in_ct + 1)) - wt_baseline
        
        all_data.append({
            'dist': 2,
            'input_ct': in_ct,
            'selected_ct': out_ct, 
            'y': y,
            'x': mut_seq
        })
    
    # 2. Process single mutants (dist=1)
    positions = single_mut_df['Position'].values - 2  # Convert to 0-indexed
    mutations = single_mut_df['Mutation'].values
    
    for i, (pos, aa) in enumerate(zip(positions, mutations)):
        mut_seq_list = list(wt_seq)
        mut_seq_list[pos] = aa
        mut_seq = ''.join(mut_seq_list)
        
        in_ct = single_mut_df['Input_Count'].iloc[i]
        out_ct = single_mut_df['Selection_Count'].iloc[i]
        y = np.log2((out_ct + 1) / (in_ct + 1)) - wt_baseline
        
        all_data.append({
            'dist': 1,
            'input_ct': in_ct,
            'selected_ct': out_ct,
            'y': y, 
            'x': mut_seq
        })
    
    # 3. Add wild type (dist=0)
    all_data.append({
        'dist': 0,
        'input_ct': wt_in_ct,
        'selected_ct': wt_out_ct,
        'y': 0.0,  # WT is the baseline
        'x': wt_seq
    })
    
    final_df = pd.DataFrame(all_data)
    # Filter by input count threshold (like in mavenn)
    return final_df[final_df['input_ct'] >= 10].reset_index(drop=True)