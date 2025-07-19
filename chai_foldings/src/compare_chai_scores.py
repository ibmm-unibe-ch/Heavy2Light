"""
CHAI-1 Comparison 
Calculates mean scores across all folded sequences from NPZ files and pLDDT from CIF files,
then performs statistical comparison between two given groups.
"""

import numpy as np
import sys
import os
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from typing import Dict, Tuple, Optional, List
import warnings

#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

def find_files(base_directory, npz_pattern="scores.model_idx_0.npz", cif_pattern="pred.model_idx_0.cif"):
    """
    Find all NPZ and CIF files in subdirectories of the base directory.
    
    Args:
        base_directory (str): Base directory to search
        npz_pattern (str): Pattern to match NPZ files
        cif_pattern (str): Pattern to match CIF files
    
    Returns:
        tuple: (npz_files dict, cif_files dict) mapping sequence names to file paths
    """
    npz_files = {}
    cif_files = {}
    
    # Search for NPZ files in all subdirectories
    npz_search_pattern = os.path.join(base_directory, "**", npz_pattern)
    found_npz_files = glob.glob(npz_search_pattern, recursive=True)
    
    for file_path in found_npz_files:
        # Extract sequence name from directory structure
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Handle multiple model indices if they exist
        model_idx = os.path.basename(file_path).split('.')[-2]  # Extract model_idx_X
        key = f"{dir_name}_{model_idx}" if "model_idx" in model_idx else dir_name
        npz_files[key] = file_path
    
    # Search for CIF files in all subdirectories
    cif_search_pattern = os.path.join(base_directory, "**", cif_pattern)
    found_cif_files = glob.glob(cif_search_pattern, recursive=True)
    
    for file_path in found_cif_files:
        # Extract sequence name from directory structure
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Handle multiple model indices if they exist in filename
        filename = os.path.basename(file_path)
        if "model_idx" in filename:
            model_idx = filename.split('.')[-2] if '.' in filename else "model_idx_0"
            key = f"{dir_name}_{model_idx}"
        else:
            key = dir_name
        cif_files[key] = file_path
    
    return npz_files, cif_files

def calculate_mean_plddt_from_cif(cif_file_path):
    """
    Calculate the mean pLDDT score from a .cif file.
    
    Args:
        cif_file_path (str): Path to the .cif file
    
    Returns:
        float or None: Mean pLDDT score, or None if calculation fails
    """
    try:
        plddt_scores = []
        
        with open(cif_file_path, 'r') as file:
            for line in file:
                # Check if line starts with ATOM
                if line.startswith('ATOM'):
                    # Split the line by whitespace
                    columns = line.split()
                    
                    # The pLDDT score is the second-to-last column
                    if len(columns) >= 2:
                        try:
                            plddt_score = float(columns[-2])
                            plddt_scores.append(plddt_score)
                        except ValueError:
                            # Skip lines where conversion fails
                            continue
        
        if plddt_scores:
            return np.mean(plddt_scores)
        else:
            return None
            
    except Exception as e:
        print(f"Error calculating pLDDT from {cif_file_path}: {e}")
        return None

def load_scores_from_npz(filepath):
    """
    Load scores from a single NPZ file.
    
    Args:
        filepath (str): Path to NPZ file
    
    Returns:
        dict: Dictionary of scores
    """
    try:
        data = np.load(filepath)
        scores = {}
        
        for key in data.keys():
            arr = data[key]
            # Convert to scalar if it's a 0-dimensional array
            if arr.ndim == 0:
                scores[key] = float(arr.item())
            else:
                # For arrays, calculate mean
                scores[key] = float(np.mean(arr))
        
        data.close()
        return scores
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_scores_single_directory(base_directory, verbose=True, npz_pattern="scores.model_idx_0.npz", cif_pattern="pred.model_idx_0.cif"):
    """
    Analyze scores from all NPZ and CIF files in a single directory structure.
    
    Args:
        base_directory (str): Base directory containing folded sequences
        verbose (bool): Whether to print detailed output
        npz_pattern (str): Pattern to match NPZ files
        cif_pattern (str): Pattern to match CIF files
    
    Returns:
        pd.DataFrame: DataFrame containing all scores
    """
    
    if verbose:
        print(f"Analyzing scores in: {base_directory}")
    
    # Find all NPZ and CIF files
    npz_files, cif_files = find_files(base_directory, npz_pattern, cif_pattern)
    
    if not npz_files and not cif_files:
        if verbose:
            print("No NPZ or CIF files found!")
        return None
    
    if verbose:
        print(f"Found {len(npz_files)} NPZ files and {len(cif_files)} CIF files")
    
    # Get all unique sequence names
    all_sequences = set(npz_files.keys()) | set(cif_files.keys())
    
    # Load scores from all files
    all_scores = {}
    failed_files = []
    
    for seq_name in all_sequences:
        scores = {}
        
        # Load NPZ scores if available
        if seq_name in npz_files:
            npz_scores = load_scores_from_npz(npz_files[seq_name])
            if npz_scores is not None:
                scores.update(npz_scores)
            else:
                failed_files.append(f"{seq_name} (NPZ)")
        
        # Load CIF pLDDT scores if available
        if seq_name in cif_files:
            plddt_score = calculate_mean_plddt_from_cif(cif_files[seq_name])
            if plddt_score is not None:
                scores['mean_plddt'] = plddt_score
            else:
                failed_files.append(f"{seq_name} (CIF)")
        
        # Only add to results if we got at least some scores
        if scores:
            all_scores[seq_name] = scores
    
    if failed_files and verbose:
        print(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    if not all_scores:
        if verbose:
            print("No scores could be loaded!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_scores, orient='index')
    
    if verbose:
        print(f"Successfully loaded scores from {len(all_scores)} sequences")
        print(f"Score types found: {list(df.columns)}")
    
    return df

def check_and_apply_log_transform(data, column_name, group_name, verbose=True):
    """
    Check if data can be log-transformed and apply if appropriate.
    
    Args:
        data: Data array/series
        column_name: Name of the column
        group_name: Name of the group
        verbose: Whether to print information
    
    Returns:
        tuple: (transformed_data, was_transformed, transform_info)
    """
    # Check if all values are positive (required for log transform)
    if np.any(data <= 0):
        if verbose:
            print(f"  {group_name} - {column_name}: Cannot log-transform (contains non-positive values)")
        return data, False, "Cannot log-transform: non-positive values"
    
    # Apply log transformation
    log_data = np.log(data)
    
    # Test normality of log-transformed data
    if len(log_data) <= 50:
        _, p_log = stats.shapiro(log_data)
        test_name = "Shapiro-Wilk"
    else:
        _, _, p_log = stats.anderson(log_data, dist='norm')
        test_name = "Anderson-Darling"
        p_log = p_log if isinstance(p_log, float) else 0.05
    
    is_normal_after_log = p_log > 0.05
    
    if verbose:
        print(f"  {group_name} - {column_name}: Log-transform normality p = {p_log:.2e} ({'Normal' if is_normal_after_log else 'Not normal'})")
    
    transform_info = f"Log-transformed, {test_name} p = {p_log:.2e}"
    
    return log_data, True, transform_info

def perform_statistical_tests(df1: pd.DataFrame, df2: pd.DataFrame, group1_name: str, group2_name: str, 
                            try_log_transform: bool = True, verbose: bool = True) -> Dict:
    """
    Perform statistical tests to compare scores between two groups.
    
    Args:
        df1: DataFrame with scores from first group
        df2: DataFrame with scores from second group
        group1_name: Name of first group
        group2_name: Name of second group
        try_log_transform: Whether to try log transformation for non-normal data
        verbose: Whether to print detailed information
    
    Returns:
        Dict: Dictionary containing test results
    """
    results = {}
    
    # Get common columns
    common_columns = set(df1.columns) & set(df2.columns)
    
    if not common_columns:
        print("Warning: No common score types found between directories!")
        return results
    
    for col in common_columns:
        if verbose:
            print(f"\nAnalyzing {col}:")
        
        # Get data for this column, removing NaN values
        data1 = df1[col].dropna()
        data2 = df2[col].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            continue
            
        col_results = {}
        
        # Basic statistics for original data
        col_results['group1_mean'] = data1.mean()
        col_results['group1_std'] = data1.std()
        col_results['group1_n'] = len(data1)
        col_results['group2_mean'] = data2.mean()
        col_results['group2_std'] = data2.std()
        col_results['group2_n'] = len(data2)
        col_results['mean_difference'] = data2.mean() - data1.mean()
        col_results['effect_size_cohens_d'] = (data2.mean() - data1.mean()) / np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
        
        # Test normality on original data
        if len(data1) <= 50:
            _, p_norm1 = stats.shapiro(data1)
            test1_name = "Shapiro-Wilk"
        else:
            _, _, p_norm1 = stats.anderson(data1, dist='norm')
            test1_name = "Anderson-Darling"
            p_norm1 = p_norm1 if isinstance(p_norm1, float) else 0.05
            
        if len(data2) <= 50:
            _, p_norm2 = stats.shapiro(data2)
            test2_name = "Shapiro-Wilk"
        else:
            _, _, p_norm2 = stats.anderson(data2, dist='norm')
            test2_name = "Anderson-Darling"
            p_norm2 = p_norm2 if isinstance(p_norm2, float) else 0.05
        
        col_results['normality_p1'] = p_norm1
        col_results['normality_p2'] = p_norm2
        col_results['normality_test1'] = test1_name
        col_results['normality_test2'] = test2_name
        is_normal_original = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        if verbose:
            print(f"  Original data normality: {group1_name} p = {p_norm1:.2e}, {group2_name} p = {p_norm2:.2e}")
        
        # Initialize variables for log-transformed data
        log_data1, log_data2 = None, None
        is_normal_log = False
        log_transform_applied = False
        
        # Try log transformation if original data is not normal
        if not is_normal_original and try_log_transform:
            if verbose:
                print(f"  Trying log transformation for {col}:")
            
            log_data1, transformed1, info1 = check_and_apply_log_transform(data1, col, group1_name, verbose)
            log_data2, transformed2, info2 = check_and_apply_log_transform(data2, col, group2_name, verbose)
            
            if transformed1 and transformed2:
                # Test normality of log-transformed data
                if len(log_data1) <= 50:
                    _, p_log1 = stats.shapiro(log_data1)
                else:
                    _, _, p_log1 = stats.anderson(log_data1, dist='norm')
                    p_log1 = p_log1 if isinstance(p_log1, float) else 0.05
                    
                if len(log_data2) <= 50:
                    _, p_log2 = stats.shapiro(log_data2)
                else:
                    _, _, p_log2 = stats.anderson(log_data2, dist='norm')
                    p_log2 = p_log2 if isinstance(p_log2, float) else 0.05
                
                is_normal_log = (p_log1 > 0.05) and (p_log2 > 0.05)
                log_transform_applied = True
                
                col_results['log_normality_p1'] = p_log1
                col_results['log_normality_p2'] = p_log2
                col_results['log_transform_info'] = f"{info1}; {info2}"
                
                if verbose:
                    print(f"  Log-transformed normality: {group1_name} p = {p_log1:.2e}, {group2_name} p = {p_log2:.2e}")
        
        # Always perform non-parametric test
        u_stat, p_nonparam = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        col_results['nonparametric_test'] = "Mann-Whitney U"
        col_results['nonparametric_statistic'] = u_stat
        col_results['nonparametric_p'] = p_nonparam
        col_results['nonparametric_significant'] = p_nonparam < 0.05
        
        # Perform parametric tests
        # Test on original data if normal
        if is_normal_original:
            # Levene's test for equal variances
            _, p_levene = stats.levene(data1, data2)
            col_results['levene_p'] = p_levene
            equal_variances = p_levene > 0.05
            
            if equal_variances:
                t_stat, p_param = stats.ttest_ind(data1, data2, equal_var=True)
                parametric_test = "Independent t-test (equal variances)"
            else:
                t_stat, p_param = stats.ttest_ind(data1, data2, equal_var=False)
                parametric_test = "Welch's t-test (unequal variances)"
            
            col_results['parametric_test'] = parametric_test
            col_results['parametric_statistic'] = t_stat
            col_results['parametric_p'] = p_param
            col_results['parametric_significant'] = p_param < 0.05
            col_results['parametric_data_used'] = "original"
            
            # Confidence interval for mean difference
            pooled_se = np.sqrt(data1.var()/len(data1) + data2.var()/len(data2))
            df_param = len(data1) + len(data2) - 2
            t_crit = stats.t.ppf(0.975, df_param)
            ci_lower = col_results['mean_difference'] - t_crit * pooled_se
            ci_upper = col_results['mean_difference'] + t_crit * pooled_se
            col_results['ci_95_lower'] = ci_lower
            col_results['ci_95_upper'] = ci_upper
        
        # Test on log-transformed data if it became normal
        elif is_normal_log and log_transform_applied:
            # Levene's test for equal variances on log data
            _, p_levene_log = stats.levene(log_data1, log_data2)
            col_results['log_levene_p'] = p_levene_log
            equal_variances_log = p_levene_log > 0.05
            
            if equal_variances_log:
                t_stat_log, p_param_log = stats.ttest_ind(log_data1, log_data2, equal_var=True)
                parametric_test_log = "Independent t-test on log-transformed data (equal variances)"
            else:
                t_stat_log, p_param_log = stats.ttest_ind(log_data1, log_data2, equal_var=False)
                parametric_test_log = "Welch's t-test on log-transformed data (unequal variances)"
            
            col_results['parametric_test'] = parametric_test_log
            col_results['parametric_statistic'] = t_stat_log
            col_results['parametric_p'] = p_param_log
            col_results['parametric_significant'] = p_param_log < 0.05
            col_results['parametric_data_used'] = "log-transformed"
            
            # Statistics for log-transformed data
            col_results['log_group1_mean'] = np.mean(log_data1)
            col_results['log_group1_std'] = np.std(log_data1)
            col_results['log_group2_mean'] = np.mean(log_data2)
            col_results['log_group2_std'] = np.std(log_data2)
            col_results['log_mean_difference'] = np.mean(log_data2) - np.mean(log_data1)
            
            # Confidence interval for log-transformed mean difference
            pooled_se_log = np.sqrt(np.var(log_data1)/len(log_data1) + np.var(log_data2)/len(log_data2))
            df_log = len(log_data1) + len(log_data2) - 2
            t_crit_log = stats.t.ppf(0.975, df_log)
            ci_lower_log = col_results['log_mean_difference'] - t_crit_log * pooled_se_log
            ci_upper_log = col_results['log_mean_difference'] + t_crit_log * pooled_se_log
            col_results['log_ci_95_lower'] = ci_lower_log
            col_results['log_ci_95_upper'] = ci_upper_log
        else:
            # No parametric test possible
            col_results['parametric_test'] = "Not applicable (non-normal data)"
            col_results['parametric_p'] = np.nan
            col_results['parametric_significant'] = False
            col_results['parametric_data_used'] = "none"
        
        # Choose the primary result (parametric if available and significant, otherwise non-parametric)
        if 'parametric_p' in col_results and not np.isnan(col_results['parametric_p']):
            col_results['primary_p_value'] = col_results['parametric_p']
            col_results['primary_test'] = col_results['parametric_test']
            col_results['primary_significant'] = col_results['parametric_significant']
        else:
            col_results['primary_p_value'] = col_results['nonparametric_p']
            col_results['primary_test'] = col_results['nonparametric_test']
            col_results['primary_significant'] = col_results['nonparametric_significant']
        
        results[col] = col_results
    
    return results

def create_normality_plots(df1: pd.DataFrame, df2: pd.DataFrame, group1_name: str, group2_name: str, 
                          output_dir: str = None, file_prefix: str = ""):
    """
    Create QQ plots and histograms to assess normality of data.
    
    Args:
        df1: DataFrame with scores from first group
        df2: DataFrame with scores from second group
        group1_name: Name of first group
        group2_name: Name of second group
        output_dir: Directory to save plots
        file_prefix: Prefix for output files
    """
    # Get common columns
    common_columns = list(set(df1.columns) & set(df2.columns))
    
    if not common_columns:
        print("No common columns to plot")
        return
    
    for col in common_columns:
        # Get data for this column, removing NaN values
        data1 = df1[col].dropna()
        data2 = df2[col].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            continue
        
        # Create figure with subplots for original data
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Normality Assessment: {col}', fontsize=16, fontweight='bold')
        
        # Original data plots
        # QQ plot for group 1
        probplot(data1, dist="norm", plot=axes[0,0])
        axes[0,0].set_title(f'QQ Plot - {group1_name} (Original)')
        axes[0,0].grid(True, alpha=0.3)
        
        # QQ plot for group 2
        probplot(data2, dist="norm", plot=axes[0,1])
        axes[0,1].set_title(f'QQ Plot - {group2_name} (Original)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Histogram for group 1
        axes[1,0].hist(data1, bins=min(20, len(data1)//2), alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[1,0].set_title(f'Histogram - {group1_name} (Original)')
        axes[1,0].set_xlabel(col)
        axes[1,0].set_ylabel('Density')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add normal curve overlay
        x_range = np.linspace(data1.min(), data1.max(), 100)
        normal_curve = stats.norm.pdf(x_range, data1.mean(), data1.std())
        axes[1,0].plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal curve')
        axes[1,0].legend()
        
        # Histogram for group 2
        axes[1,1].hist(data2, bins=min(20, len(data2)//2), alpha=0.7, density=True, color='lightcoral', edgecolor='black')
        axes[1,1].set_title(f'Histogram - {group2_name} (Original)')
        axes[1,1].set_xlabel(col)
        axes[1,1].set_ylabel('Density')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add normal curve overlay
        x_range = np.linspace(data2.min(), data2.max(), 100)
        normal_curve = stats.norm.pdf(x_range, data2.mean(), data2.std())
        axes[1,1].plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal curve')
        axes[1,1].legend()
        
        # Log-transformed plots (if possible)
        if np.all(data1 > 0) and np.all(data2 > 0):
            log_data1 = np.log(data1)
            log_data2 = np.log(data2)
            
            # QQ plot for log-transformed data comparison
            probplot(log_data1, dist="norm", plot=axes[0,2])
            probplot(log_data2, dist="norm", plot=axes[0,2])
            axes[0,2].set_title(f'QQ Plot - Log-transformed (Both Groups)')
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].legend([f'{group1_name} (log)', f'{group2_name} (log)'])
            
            # Combined histogram for log-transformed data
            axes[1,2].hist(log_data1, bins=min(20, len(log_data1)//2), alpha=0.5, density=True, 
                          color='skyblue', edgecolor='black', label=f'{group1_name} (log)')
            axes[1,2].hist(log_data2, bins=min(20, len(log_data2)//2), alpha=0.5, density=True, 
                          color='lightcoral', edgecolor='black', label=f'{group2_name} (log)')
            axes[1,2].set_title(f'Histogram - Log-transformed')
            axes[1,2].set_xlabel(f'log({col})')
            axes[1,2].set_ylabel('Density')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].legend()
        else:
            # Remove the third column if log transform not possible
            axes[0,2].text(0.5, 0.5, 'Log transform\nnot possible\n(non-positive values)', 
                          ha='center', va='center', transform=axes[0,2].transAxes, fontsize=12)
            axes[0,2].set_xticks([])
            axes[0,2].set_yticks([])
            
            axes[1,2].text(0.5, 0.5, 'Log transform\nnot possible\n(non-positive values)', 
                          ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
            axes[1,2].set_xticks([])
            axes[1,2].set_yticks([])
        
        plt.tight_layout()
        
        if output_dir:
            filename = f'{file_prefix}normality_assessment_{col}.png' if file_prefix else f'normality_assessment_{col}.png'
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Normality plots for {col} saved to: {plot_path}")
        
        plt.show()



def create_comparison_plots(df1: pd.DataFrame, df2: pd.DataFrame, group1_name: str, group2_name: str, 
                          output_dir: str = None, file_prefix: str = "", mean_line_width: float = 0.3):
    """
    Create individual violin plots comparing the two groups for each score type.
    All plots will have the same size and axis dimensions.
    
    Args:
        df1: DataFrame with scores from first group
        df2: DataFrame with scores from second group
        group1_name: Name of first group
        group2_name: Name of second group
        output_dir: Directory to save plots
        file_prefix: Prefix for output files
        mean_line_width: Width of the mean line bars (default: 0.3)
    """
    # Get common columns
    common_columns = list(set(df1.columns) & set(df2.columns))
    
    if not common_columns:
        print("No common columns to plot")
        return
    
    # Create combined dataframe for plotting
    df1_plot = df1[common_columns].copy()
    df1_plot['group'] = group1_name
    df2_plot = df2[common_columns].copy()
    df2_plot['group'] = group2_name
    
    combined_df = pd.concat([df1_plot, df2_plot], ignore_index=True)
    
    # Melt for easier plotting
    melted_df = combined_df.melt(id_vars=['group'], var_name='score_type', value_name='score')
    
    # Define colors for the two groups
    group_colors = {
        group1_name: '#fbfe71', # #168aad #daf284 #fbfe71
        group2_name: '#76c893' 
    }
    
    # Score name mappings
    score_name_mapping = {
        'mean_plddt': 'pLDDT',
        'ptm': 'pTM',
        'chain_chain_clashes': 'Chain-chain clashes',
        'iptm': 'ipTM',
        'aggregate_score': 'Aggregate score',
        'has_inter_chain_clshes': 'Inter-chain clashes',
        'per_chain_ptm': 'Per chain pTM',
        'per_chain_pair_iptm': 'Per chain pair ipTM'
    }
    
    # Set font size globally
    plt.rcParams.update({'font.size': 18})
    
    # Calculate global y-axis limits for consistency
    y_limits = {}
    for col in common_columns:
        col_data = melted_df[melted_df['score_type'] == col]['score'].dropna()
        if len(col_data) > 0:
            y_min = col_data.min()
            y_max = col_data.max()
            y_range = y_max - y_min
            margin = y_range * 0.1  # 10% margin
            y_limits[col] = (y_min - margin, y_max + margin)
    
    # Standard subplot parameters for consistent sizing
    subplot_params = {
        'left': 0.15,     # left margin
        'bottom': 0.15,   # bottom margin  
        'right': 0.95,    # right margin
        'top': 0.95,      # top margin
        'wspace': 0.2,    # width spacing
        'hspace': 0.2     # height spacing
    }
    
    # Create individual plots for each score type
    for col in common_columns:
        # Create new figure for each score type with consistent parameters
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(**subplot_params)
        
        # Get data for this specific score type
        col_data = melted_df[melted_df['score_type'] == col]
        
        # Create violin plot with custom colors
        colors_list = [group_colors[group] for group in [group1_name, group2_name]]
        sns.violinplot(data=col_data, x='group', y='score', ax=ax, 
                      palette=colors_list, inner=None, order=[group1_name, group2_name])
        
        # Set ylabel to score name
        ylabel = score_name_mapping.get(col, col)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        
        # Set consistent y-axis limits
        if col in y_limits:
            ax.set_ylim(y_limits[col])
        
        # Set consistent x-axis limits and ticks
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group1_name, group2_name])
        
        # Remove top and right spines
        sns.despine(ax=ax, top=True, right=True)
        
        # Add mean bars (horizontal lines) for each group
        for i, group in enumerate([group1_name, group2_name]):
            group_data = col_data[col_data['group'] == group]['score']
            if len(group_data) > 0:
                mean_val = group_data.mean()
                # Add horizontal line as mean bar
                ax.hlines(y=mean_val, xmin=i-mean_line_width, xmax=i+mean_line_width, 
                         colors='black', linestyles='dashed', linewidth=1)
    
        
        # Apply tight layout without changing the subplot parameters
        plt.tight_layout()
        fig.subplots_adjust(**subplot_params)  # Re-apply after tight_layout
        
        # Save individual plot
        if output_dir:
            filename = f'{file_prefix}comparison_{col}_violin.png' if file_prefix else f'comparison_{col}_violin.png'
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Violin plot for {col} saved to: {plot_path}")
        
        plt.show()
        plt.close()



def print_comparison_results(results: Dict, group1_name: str, group2_name: str):
    """
    Print formatted comparison results with both parametric and non-parametric tests.
    
    Args:
        results: Dictionary of statistical test results
        group1_name: Name of first group
        group2_name: Name of second group
    """
    print("\n" + "=" * 100)
    print(f"STATISTICAL COMPARISON: {group1_name} vs {group2_name}")
    print("=" * 100)
    
    if not results:
        print("No common score types found for comparison!")
        return
    
    # Summary table
    print(f"\n{'Score Type':<20} {'Group1 Mean':<12} {'Group2 Mean':<12} {'Difference':<12} {'Parametric P':<15} {'Non-param P':<15} {'Primary Test'}")
    print("-" * 140)
    
    significant_results = []
    
    for score_type, result in results.items():
        param_p_str = f"{result['parametric_p']:.2e}" if not np.isnan(result.get('parametric_p', np.nan)) else "N/A"
        nonparam_p_str = f"{result['nonparametric_p']:.2e}"
        
        primary_sig = "Yes***" if result['primary_p_value'] < 0.001 else "Yes**" if result['primary_p_value'] < 0.01 else "Yes*" if result['primary_p_value'] < 0.05 else "No"
        
        # Truncate test name for display
        primary_test_short = result['primary_test'][:25] + "..." if len(result['primary_test']) > 28 else result['primary_test']
        
        print(f"{score_type:<20} {result['group1_mean']:<12.6f} {result['group2_mean']:<12.6f} {result['mean_difference']:<12.6f} {param_p_str:<15} {nonparam_p_str:<15} {primary_test_short}")
        
        if result['primary_significant']:
            significant_results.append((score_type, result))
    
    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")
    
    # Detailed results for significant findings
    if significant_results:
        print(f"\n" + "=" * 100)
        print("DETAILED RESULTS FOR SIGNIFICANT DIFFERENCES:")
        print("=" * 100)
        
        for score_type, result in significant_results:
            print(f"\n{score_type.upper()}:")
            print(f"  {group1_name}: {result['group1_mean']:.6f} ± {result['group1_std']:.6f} (n={result['group1_n']})")
            print(f"  {group2_name}: {result['group2_mean']:.6f} ± {result['group2_std']:.6f} (n={result['group2_n']})")
            print(f"  Mean difference: {result['mean_difference']:.6f}")
            print(f"  Effect size (Cohen's d): {result['effect_size_cohens_d']:.4f}")
            
            # Parametric test results
            if not np.isnan(result.get('parametric_p', np.nan)):
                print(f"\n  PARAMETRIC TEST:")
                print(f"    Test: {result['parametric_test']}")
                print(f"    Data used: {result['parametric_data_used']}")
                print(f"    P-value: {result['parametric_p']:.2e}")
                print(f"    Significant: {'Yes' if result['parametric_significant'] else 'No'}")
                
                if result['parametric_data_used'] == 'log-transformed':
                    print(f"    Log-transformed {group1_name}: {result['log_group1_mean']:.6f} ± {result['log_group1_std']:.6f}")
                    print(f"    Log-transformed {group2_name}: {result['log_group2_mean']:.6f} ± {result['log_group2_std']:.6f}")
                    print(f"    Log difference: {result['log_mean_difference']:.6f}")
                    if 'log_ci_95_lower' in result:
                        print(f"    95% CI for log difference: [{result['log_ci_95_lower']:.6f}, {result['log_ci_95_upper']:.6f}]")
                elif 'ci_95_lower' in result:
                    print(f"    95% CI for difference: [{result['ci_95_lower']:.6f}, {result['ci_95_upper']:.6f}]")
            
            # Non-parametric test results
            print(f"\n  NON-PARAMETRIC TEST:")
            print(f"    Test: {result['nonparametric_test']}")
            print(f"    Statistic: {result['nonparametric_statistic']:.2f}")
            print(f"    P-value: {result['nonparametric_p']:.2e}")
            print(f"    Significant: {'Yes' if result['nonparametric_significant'] else 'No'}")
            
            # Primary recommendation
            print(f"\n  PRIMARY RESULT: {result['primary_test']}")
            print(f"    P-value: {result['primary_p_value']:.2e}")
            
            # Interpret effect size
            abs_d = abs(result['effect_size_cohens_d'])
            if abs_d < 0.2:
                effect_size_interp = "negligible"
            elif abs_d < 0.5:
                effect_size_interp = "small"
            elif abs_d < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
            print(f"    Effect size interpretation: {effect_size_interp}")
            
            # Log transformation info
            if 'log_transform_info' in result:
                print(f"    Log transformation: {result['log_transform_info']}")
    
    # Multiple testing correction warning
    if len(results) > 1:
        print(f"\n" + "=" * 100)
        print("MULTIPLE TESTING CORRECTION:")
        print("=" * 100)
        print(f"Note: {len(results)} statistical tests were performed.")
        print("Consider applying Bonferroni correction for multiple comparisons.")
        bonferroni_alpha = 0.05 / len(results)
        print(f"Bonferroni-corrected significance level: α = {bonferroni_alpha:.2e}")
        
        # Check both parametric and non-parametric results
        bonferroni_significant_param = [(name, result) for name, result in results.items() 
                                       if not np.isnan(result.get('parametric_p', np.nan)) and result['parametric_p'] < bonferroni_alpha]
        bonferroni_significant_nonparam = [(name, result) for name, result in results.items() 
                                          if result['nonparametric_p'] < bonferroni_alpha]
        
        if bonferroni_significant_param or bonferroni_significant_nonparam:
            print(f"\nSignificant after Bonferroni correction:")
            all_bonf_sig = set()
            
            if bonferroni_significant_param:
                print("  Parametric tests:")
                for name, result in bonferroni_significant_param:
                    print(f"    {name}: p = {result['parametric_p']:.2e} ({result['parametric_test']})")
                    all_bonf_sig.add(name)
            
            if bonferroni_significant_nonparam:
                print("  Non-parametric tests:")
                for name, result in bonferroni_significant_nonparam:
                    if name not in all_bonf_sig:  # Don't duplicate if both tests are significant
                        print(f"    {name}: p = {result['nonparametric_p']:.2e} ({result['nonparametric_test']})")
        else:
            print("No results remain significant after Bonferroni correction.")

def create_summary_report(df1: pd.DataFrame, df2: pd.DataFrame, results: Dict, group1_name: str, group2_name: str, 
                         output_file: str = None, file_prefix: str = ""):
    """
    Create a comprehensive summary report.
    
    Args:
        df1: DataFrame with scores from first group
        df2: DataFrame with scores from second group
        results: Statistical test results
        group1_name: Name of first group
        group2_name: Name of second group
        output_file: Path to save the report
        file_prefix: Prefix for output files
    """
    report_lines = []
    
    report_lines.append("CHAI-1 COMPARATIVE ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Group 1: {group1_name}")
    report_lines.append(f"Group 2: {group2_name}")
    report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Sample sizes
    report_lines.append("SAMPLE SIZES:")
    report_lines.append(f"  {group1_name}: {len(df1)} sequences")
    report_lines.append(f"  {group2_name}: {len(df2)} sequences")
    report_lines.append("")
    
    # Score types
    common_cols = set(df1.columns) & set(df2.columns)
    report_lines.append(f"COMMON SCORE TYPES ({len(common_cols)}):")
    for col in sorted(common_cols):
        report_lines.append(f"  - {col}")
    report_lines.append("")
    
    # Statistical results summary
    if results:
        param_significant_count = sum(1 for r in results.values() if r.get('parametric_significant', False))
        nonparam_significant_count = sum(1 for r in results.values() if r.get('nonparametric_significant', False))
        primary_significant_count = sum(1 for r in results.values() if r['primary_significant'])
        
        report_lines.append("STATISTICAL RESULTS SUMMARY:")
        report_lines.append(f"  Total comparisons: {len(results)}")
        report_lines.append(f"  Primary significant differences (p<0.05): {primary_significant_count}")
        report_lines.append(f"  Parametric test significant: {param_significant_count}")
        report_lines.append(f"  Non-parametric test significant: {nonparam_significant_count}")
        report_lines.append("")
        
        for score_type, result in results.items():
            report_lines.append(f"{score_type.upper()}:")
            report_lines.append(f"  {group1_name}: {result['group1_mean']:.6f} ± {result['group1_std']:.6f}")
            report_lines.append(f"  {group2_name}: {result['group2_mean']:.6f} ± {result['group2_std']:.6f}")
            report_lines.append(f"  Difference: {result['mean_difference']:.6f}")
            
            # Parametric results
            if not np.isnan(result.get('parametric_p', np.nan)):
                report_lines.append(f"  Parametric test: {result['parametric_test']}")
                report_lines.append(f"  Parametric p-value: {result['parametric_p']:.2e}")
                report_lines.append(f"  Parametric significant: {'Yes' if result['parametric_significant'] else 'No'}")
            else:
                report_lines.append(f"  Parametric test: Not applicable")
            
            # Non-parametric results
            report_lines.append(f"  Non-parametric test: {result['nonparametric_test']}")
            report_lines.append(f"  Non-parametric p-value: {result['nonparametric_p']:.2e}")
            report_lines.append(f"  Non-parametric significant: {'Yes' if result['nonparametric_significant'] else 'No'}")
            
            # Primary result
            report_lines.append(f"  Primary test used: {result['primary_test']}")
            report_lines.append(f"  Primary p-value: {result['primary_p_value']:.2e}")
            report_lines.append(f"  Primary significant: {'Yes' if result['primary_significant'] else 'No'}")
            report_lines.append(f"  Effect size (Cohen's d): {result['effect_size_cohens_d']:.4f}")
            
            # Log transformation info
            if 'log_transform_info' in result:
                report_lines.append(f"  Log transformation: Applied")
                report_lines.append(f"  Log transform details: {result['log_transform_info']}")
            
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        filename = f'{file_prefix}comparison_report.txt' if file_prefix else 'comparison_report.txt'
        full_path = os.path.join(os.path.dirname(output_file), filename) if os.path.dirname(output_file) else filename
        with open(full_path, 'w') as f:
            f.write(report_text)
        print(f"Summary report saved to: {full_path}")
    
    return report_text

def main():
    """Main function to handle command line arguments."""
    
    parser = argparse.ArgumentParser(description='Compare CHAI-1 folding scores between two directories')
    parser.add_argument('directory1', help='First directory containing folded sequences')
    parser.add_argument('directory2', help='Second directory containing folded sequences')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--npz-pattern', default='scores.model_idx_0.npz',
                       help='Pattern to match NPZ files')
    parser.add_argument('--cif-pattern', default='pred.model_idx_0.cif',
                       help='Pattern to match CIF files')
    parser.add_argument('--group1-name', default='Group1',
                       help='Name for first group in outputs')
    parser.add_argument('--group2-name', default='Group2',
                       help='Name for second group in outputs')
    parser.add_argument('--prefix', default='',
                       help='Prefix for all output files and figures')
    parser.add_argument('--log-transform', action='store_true', default=True,
                       help='Try log transformation for non-normal data (default: True)')
    parser.add_argument('--no-log-transform', action='store_true',
                       help='Disable log transformation attempts')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    parser.add_argument('--normality-only', action='store_true',
                       help='Only create normality assessment plots')
    
    args = parser.parse_args()
    
    # Handle log transform arguments
    try_log_transform = args.log_transform and not args.no_log_transform
    
    # Check if directories exist
    for i, directory in enumerate([args.directory1, args.directory2], 1):
        if not os.path.exists(directory):
            print(f"Error: Directory {i} '{directory}' not found!")
            sys.exit(1)
    
    verbose = not args.quiet
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Add prefix formatting
    file_prefix = f"{args.prefix}_" if args.prefix and not args.prefix.endswith('_') else args.prefix
    
    print("CHAI-1 COMPARATIVE SCORE ANALYZER (Enhanced)")
    print("=" * 60)
    print(f"Directory 1 ({args.group1_name}): {args.directory1}")
    print(f"Directory 2 ({args.group2_name}): {args.directory2}")
    if file_prefix:
        print(f"File prefix: {file_prefix}")
    print(f"Log transformation: {'Enabled' if try_log_transform else 'Disabled'}")
    print("=" * 60)
    
    # Analyze both directories
    print("\nAnalyzing first directory...")
    df1 = analyze_scores_single_directory(args.directory1, verbose, args.npz_pattern, args.cif_pattern)
    
    print("\nAnalyzing second directory...")
    df2 = analyze_scores_single_directory(args.directory2, verbose, args.npz_pattern, args.cif_pattern)
    
    if df1 is None:
        print(f"Error: No data found in {args.directory1}")
        sys.exit(1)
    
    if df2 is None:
        print(f"Error: No data found in {args.directory2}")
        sys.exit(1)
    
    # Perform statistical comparison
    if not args.normality_only:
        print("\nPerforming statistical comparison...")
        print("Both parametric and non-parametric tests will be performed.")
        if try_log_transform:
            print("Log transformation will be attempted for non-normal data.")
        
        results = perform_statistical_tests(df1, df2, args.group1_name, args.group2_name, 
                                          try_log_transform=try_log_transform, verbose=verbose)
        
        # Print results
        print_comparison_results(results, args.group1_name, args.group2_name)
    else:
        results = {}
    
    # Create normality assessment plots
    if not args.no_plots:
        try:
            print("\nCreating normality assessment plots...")
            create_normality_plots(df1, df2, args.group1_name, args.group2_name, args.output, file_prefix)
        except Exception as e:
            print(f"Warning: Could not create normality plots: {e}")
    
    # Create comparison plots (skip if normality-only mode)
    if not args.no_plots and not args.normality_only:
        try:
            print("\nCreating comparison plots...")
            create_comparison_plots(df1, df2, args.group1_name, args.group2_name, args.output, file_prefix)
        except Exception as e:
            print(f"Warning: Could not create comparison plots: {e}")
    
    # Save detailed results (skip if normality-only mode)
    if args.output and not args.normality_only:
        # Save individual dataframes
        df1_filename = f'{file_prefix}{args.group1_name}_scores.csv' if file_prefix else f'{args.group1_name}_scores.csv'
        df2_filename = f'{file_prefix}{args.group2_name}_scores.csv' if file_prefix else f'{args.group2_name}_scores.csv'
        
        df1.to_csv(os.path.join(args.output, df1_filename))
        df2.to_csv(os.path.join(args.output, df2_filename))
        
        # Save statistical results
        if results:
            results_filename = f'{file_prefix}statistical_comparison.csv' if file_prefix else 'statistical_comparison.csv'
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.to_csv(os.path.join(args.output, results_filename))
        
        # Create summary report
        report_filename = f'{file_prefix}comparison_report.txt' if file_prefix else 'comparison_report.txt'
        report_file = os.path.join(args.output, report_filename)
        create_summary_report(df1, df2, results, args.group1_name, args.group2_name, report_file, file_prefix)
        
        print(f"\nAll results saved to: {args.output}")
        if file_prefix:
            print(f"All files use prefix: {file_prefix}")
    
    if not args.normality_only:
        print(f"\nComparative analysis completed!")
        print("Both parametric and non-parametric tests were performed for robust comparison.")
        if try_log_transform:
            print("Log transformation was attempted where appropriate for non-normal data.")
    else:
        print(f"\nNormality assessment completed!")
        print("Use the QQ plots and histograms to assess normality visually.")
        print("If data appears normal in QQ plots, t-tests are appropriate.")
        print("If data deviates from the diagonal line in QQ plots, use non-parametric tests.")

if __name__ == "__main__":
    main()