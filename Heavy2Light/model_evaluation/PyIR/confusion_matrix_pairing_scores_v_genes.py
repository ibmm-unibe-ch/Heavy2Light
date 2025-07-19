import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import re

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


def load_and_clean_data_with_pairing_filter(csv_file_path: str, min_pairing_score: float = 0.5,
                                           true_col: str = "true_v_gene_family_call", 
                                           gen_col: str = "gen_v_gene_family_call",
                                           family_col: str = "true_v_gene_family_call",
                                           family_filter: str = None) -> pd.DataFrame:
    """
    Load the CSV file and clean the data for analysis with pairing score filter.
    
    Args:
        csv_file_path: Path to the merged CSV file
        min_pairing_score: Minimum pairing score threshold (default: 0.5)
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
        family_filter: Optional V gene family to filter for (e.g. 'IGKV1', 'IGLV2')
        
    Returns:
        Cleaned DataFrame with non-null V gene family calls and pairing score >= threshold
    """
    df = pd.read_csv(csv_file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Check if pairing_scores column exists
    if 'pairing_scores' not in df.columns:
        print("Warning: 'pairing_scores' column not found in dataset!")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()
    
    # Check if specified columns exist
    missing_cols = []
    if true_col not in df.columns:
        missing_cols.append(true_col)
    if gen_col not in df.columns:
        missing_cols.append(gen_col)
    
    if missing_cols:
        print(f"Warning: The following columns not found in dataset: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()
    
    print(f"Rows with pairing_scores: {df['pairing_scores'].notna().sum()}")
    print(f"Pairing scores range: {df['pairing_scores'].min():.3f} to {df['pairing_scores'].max():.3f}")
    
    # Filter by pairing score
    pairing_filtered_df = df[df['pairing_scores'] >= min_pairing_score]
    print(f"After filtering for pairing_scores >= {min_pairing_score}: {pairing_filtered_df.shape}")
    
    print(f"Rows with {gen_col}: {pairing_filtered_df[gen_col].notna().sum()}")
    print(f"Rows with {true_col}: {pairing_filtered_df[true_col].notna().sum()}")
    
    # Keep only rows where both gen and true V gene family calls are available
    clean_df = pairing_filtered_df.dropna(subset=[gen_col, true_col])

    # Remove IGHV3 from analysis (if you still want this filter)
    if 'IGHV3' in clean_df[gen_col].values or 'IGHV3' in clean_df[true_col].values:
        clean_df = clean_df[
            (clean_df[gen_col] != 'IGHV3') & 
            (clean_df[true_col] != 'IGHV3')
        ]
        print("Note: IGHV3 entries removed from analysis")

    # # Filter out entries that have only "IGKV1" (without suffix)
    # before_igkv1_filter = len(clean_df)
    # clean_df = clean_df[
    #     (clean_df[gen_col] != 'IGKV1') & 
    #     (clean_df[true_col] != 'IGKV1') &
    #     (clean_df[gen_col] != 'IGKV3') &
    #     (clean_df[true_col] != 'IGKV3')
    # ]
    # after_igkv1_filter = len(clean_df)
    # if before_igkv1_filter != after_igkv1_filter:
    #     print(f"Note: Removed {before_igkv1_filter - after_igkv1_filter} entries with only 'IGKV1' (without suffix)")
    
    
    # Apply family filter if specified
    if family_filter:
        print(f"\nApplying V gene family filter: {family_filter}")
        before_filter = len(clean_df)
        
        # Filter to include only genes that belong to the specified family
        # For example, if family_filter is 'IGKV1', this will match IGKV1, IGKV1-16, IGKV1-27, etc.
        clean_df = clean_df[
            clean_df[true_col].str.startswith(family_filter) & 
            clean_df[gen_col].str.startswith(family_filter)
        ]
        
        after_filter = len(clean_df)
        print(f"After filtering for {family_filter} family genes: {after_filter} rows (was {before_filter})")
        
        if after_filter == 0:
            print(f"Warning: No data found for family {family_filter}")
            available_families = sorted(clean_df[true_col].unique())
            print(f"Available genes: {available_families}")
    
    print(f"Final clean dataset shape (both calls available, pairing >= {min_pairing_score}): {clean_df.shape}")
    
    # Print pairing score statistics for final dataset
    if len(clean_df) > 0:
        print(f"Final dataset pairing scores - Mean: {clean_df['pairing_scores'].mean():.3f}, "
              f"Median: {clean_df['pairing_scores'].median():.3f}, "
              f"Std: {clean_df['pairing_scores'].std():.3f}")
    
    return clean_df

def sort_v_genes_by_chain_and_family(labels):
    """
    Sort V genes: IGKV1, IGKV2... then IGLV1, IGLV2...
    """
    def sort_key(gene):
        # Extract chain type (IGK or IGL) and family number
        if 'IGKV' in gene:
            chain_priority = 1  # IGK first
            match = re.search(r'IGKV(\d+)', gene)
        elif 'IGLV' in gene:
            chain_priority = 2  # IGL second
            match = re.search(r'IGLV(\d+)', gene)
        else:
            return (999, 999, gene)  # Put unknown patterns at end
        
        family_num = int(match.group(1)) if match else 999
        return (chain_priority, family_num, gene)
    
    return sorted(labels, key=sort_key)

def get_family_from_gene(df, gene_col, family_col):
    """
    Create mapping from gene calls to their families and get family order for genes.
    """
    # Create gene to family mapping
    gene_family_map = df.groupby(gene_col)[family_col].first().to_dict()
    
    # Get unique families and sort them (IGKV first, then IGLV)
    unique_families = df[family_col].unique()
    
    def family_sort_key(family):
        if 'IGKV' in family:
            chain_priority = 1
            match = re.search(r'IGKV(\d+)', family)
        elif 'IGLV' in family:
            chain_priority = 2
            match = re.search(r'IGLV(\d+)', family)
        else:
            return (999, 999)
        
        family_num = int(match.group(1)) if match else 999
        return (chain_priority, family_num)
    
    unique_families = sorted(unique_families, key=family_sort_key)
    
    return gene_family_map, unique_families



def generate_confusion_matrix_with_pairing(df: pd.DataFrame, save_path: str = None, 
                                         figsize: tuple = (18, 14), min_pairing_score: float = 0.5,
                                         true_col: str = "true_v_gene_family_call", 
                                         gen_col: str = "gen_v_gene_family_call",
                                        true_family_col: str = "true_v_gene_family_call",
                                        gen_family_col: str = "gen_v_gene_family_call",
                                         use_rainbow: bool = False):
    """
    Generate and display confusion matrix for V gene family calls with pairing score filter.
    
    Args:
        df: DataFrame with specified columns and pairing_scores
        save_path: Optional path to save the confusion matrix plot
        figsize: Figure size for the plot
        min_pairing_score: Minimum pairing score used for filtering
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    true_labels = df[true_col]
    pred_labels = df[gen_col]
    
    df_filtered = df[~df[true_col].str.contains('IGH', na=False) & 
                     ~df[gen_col].str.contains('IGH', na=False)].copy()
    
    # Get all unique labels
    all_labels = sorted(list(set(true_labels.unique()) | set(pred_labels.unique())))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Create the plot
    plt.figure(figsize=figsize)

    true_gene_family_map, unique_families = get_family_from_gene(df_filtered, true_col, true_family_col)

    # Choose colormap
    if use_rainbow:
        base_cmap = plt.cm.jet(np.linspace(0, 1, len(unique_families)))
    else:
        base_cmap = plt.cm.Spectral

    
    # Create heatmap
    sns.heatmap(cm, 
                annot=False, 
                fmt='d', 
                cmap=base_cmap,
                xticklabels=all_labels,
                yticklabels=all_labels,
                cbar_kws={'label': 'Count'})
    
    title = f'{gen_col.replace("_", " ").title()} vs {true_col.replace("_", " ").title()} Confusion Matrix (Pairing Score ≥ {min_pairing_score})\n'
    title += f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) | n = {len(df)}'
    plt.title(title, fontsize=26, fontweight='bold')
    plt.xlabel(f'Generated V gene', fontsize=22, fontweight='bold')
    plt.ylabel(f'True V gene', fontsize=22, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=True)
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, all_labels, accuracy


def generate_confusion_matrix_single(df: pd.DataFrame, save_path: str = None, 
                                   figsize: tuple = (18, 14), threshold: float = 0.5,
                                   true_col: str = "true_v_gene_family_call", 
                                   gen_col: str = "gen_v_gene_family_call",
                                   true_family_col: str = "true_v_gene_family_call",
                                   gen_family_col: str = "gen_v_gene_family_call",
                                   use_rainbow: bool = False, is_low: bool = False):
    """
    Generate single confusion matrix with family-based colors and transparency.
    """
    # Filter out rows with IGH genes if needed
    df_filtered = df[~df[true_col].str.contains('IGH', na=False) & 
                     ~df[gen_col].str.contains('IGH', na=False)].copy()
    
    true_labels = df_filtered[true_col]
    pred_labels = df_filtered[gen_col]
    
    # Get gene to family mappings and unique families
    true_gene_family_map, unique_families = get_family_from_gene(df_filtered, true_col, true_family_col)
    gen_gene_family_map, _ = get_family_from_gene(df_filtered, gen_col, gen_family_col)
    
    # Get all unique labels and sort them by chain type and family
    all_labels_unsorted = list(set(true_labels.unique()) | set(pred_labels.unique()))
    all_labels = sort_v_genes_by_chain_and_family(all_labels_unsorted)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Create family color mapping
    if use_rainbow:
        family_colors = plt.cm.jet(np.linspace(0, 1, len(unique_families)))
    else:
        family_colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_families)))
    
    family_color_map = {family: family_colors[i] for i, family in enumerate(unique_families)}
    
    # Create custom colored matrix with transparency based on values
    colored_matrix = np.zeros((len(all_labels), len(all_labels), 4))  # RGBA
    
    # Normalize confusion matrix values for transparency (0-1 range)
    cm_normalized = cm / cm.max() if cm.max() > 0 else cm
    
    # Apply power scaling to increase contrast
    cm_contrast = np.power(cm_normalized, 0.3)
    
    for i, true_gene in enumerate(all_labels):
        for j, pred_gene in enumerate(all_labels):
            # Get family for true gene (row)
            true_family = true_gene_family_map.get(true_gene)
            
            if true_family and true_family in family_color_map:
                base_color = family_color_map[true_family][:3]  # RGB only
                # Increased transparency range and contrast
                alpha = 0.1 + 0.9 * cm_contrast[i, j]  # Range from 0.1 to 1.0
                colored_matrix[i, j] = [base_color[0], base_color[1], base_color[2], alpha]
            else:
                # Default gray for unknown families
                alpha = 0.1 + 0.9 * cm_contrast[i, j]
                colored_matrix[i, j] = [0.5, 0.5, 0.5, alpha]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the custom colored matrix
    im = ax.imshow(colored_matrix, aspect='auto', interpolation='nearest')
    
    # Add white borders around each tile
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                               fill=False, edgecolor='white', 
                               linewidth=1, alpha=1)
            ax.add_patch(rect)
    
    # Remove all spines and styling for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set labels and ticks
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=90, ha='center', fontsize=18)
    ax.set_yticklabels(all_labels, rotation=0, ha='right', fontsize=18)
    
    # Create title based on pairing score range
    if is_low:
        title = f'{gen_col.replace("_", " ").title()} vs {true_col.replace("_", " ").title()} Confusion Matrix (Pairing Score < {threshold})\n'
    else:
        title = f'{gen_col.replace("_", " ").title()} vs {true_col.replace("_", " ").title()} Confusion Matrix (Pairing Score ≥ {threshold})\n'
    
    title += f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) | n = {len(df)}'
    
    # Add colormap info to title
    cmap_name = "Rainbow Family-Colored" if use_rainbow else "Spectral Family-Colored"
    title += f' | {cmap_name}'
    
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_xlabel(f'Generated V gene', fontsize=26, fontweight='bold')
    ax.set_ylabel(f'True V gene', fontsize=26, fontweight='bold')


    # Create a separate colorbar showing the value scale
    # Create a dummy mappable for the colorbar
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=cm.max())
    sm = ScalarMappable(norm=norm, cmap=plt.cm.Greys)
    sm.set_array([])

    # cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    # cbar.set_label('Count (Transparency)', rotation=270, labelpad=20, fontsize=12)
    # cbar.outline.set_visible(False)
    
    # Create legend for family colors
    legend_elements = []
    for family in unique_families:
        if family in family_color_map:
            color = family_color_map[family]
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, 
                                               edgecolor='black', alpha=0.8, 
                                               label=family))
    
    # Place legend outside the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
             title='V Gene Families', title_fontsize=12, fontsize=10)
    
    # Remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    # Adjust layout
    plt.tight_layout()

    # Adjust layout
    plt.tight_layout()
    
    # Add the missing save and show functionality
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=True)
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, all_labels, accuracy



def generate_dual_pairing_confusion_matrices(df_all: pd.DataFrame, threshold: float = 0.5,
                                           save_path_high: str = None, save_path_low: str = None,
                                           figsize: tuple = (18, 14),
                                           true_col: str = "true_v_gene_family_call", 
                                           gen_col: str = "gen_v_gene_family_call",
                                           true_family_col: str = "true_v_gene_family_call",
                                           gen_family_col: str = "gen_v_gene_family_call",
                                           use_rainbow: bool = False,
                                           family_filter: str = None):
    """
    Generate two confusion matrices: one for pairing scores >= threshold and one for < threshold.
    """
    # Apply family filter first if specified
    if family_filter:
        print(f"Applying family filter: {family_filter}")
        df_all = df_all[df_all[true_col].str.startswith(family_filter) &
                        df_all[gen_col].str.startswith(family_filter)]
        print(f"Rows after family filter: {len(df_all)}")

    
    # Split data by pairing score threshold
    df_high = df_all[df_all['pairing_scores'] >= threshold].dropna(subset=[gen_col, true_col])
    df_low = df_all[df_all['pairing_scores'] < threshold].dropna(subset=[gen_col, true_col])
    
    # Remove IGHV3 if present
    for df_subset in [df_high, df_low]:
        if len(df_subset) > 0 and ('IGHV3' in df_subset[gen_col].values or 'IGHV3' in df_subset[true_col].values):
            df_subset = df_subset[
                (df_subset[gen_col] != 'IGHV3') & 
                (df_subset[true_col] != 'IGHV3')
            ]

    # Remove entries with only "IGKV1" (without suffix) from both subsets
    # for df_subset_name, df_subset in [('df_high', df_high), ('df_low', df_low)]:
    #     if len(df_subset) > 0:
    #         before_filter = len(df_subset)
    #         if df_subset_name == 'df_high':
    #             df_high = df_subset[
    #                 (df_subset[gen_col] != 'IGKV1') & 
    #                 (df_subset[true_col] != 'IGKV1') &
    #                 (df_subset[gen_col] != 'IGKV3') &
    #                 (df_subset[true_col] != 'IGKV3')
    #             ]
    #             after_filter = len(df_high)
    #         else:
    #             df_low = df_subset[
    #                 (df_subset[gen_col] != 'IGKV1') & 
    #                 (df_subset[true_col] != 'IGKV1') &
    #                 (df_subset[gen_col] != 'IGKV3') &
    #                 (df_subset[true_col] != 'IGKV3')
    #             ]
    #             after_filter = len(df_low)
            
    #         if before_filter != after_filter:
    #             print(f"Removed {before_filter - after_filter} 'IGKV1' and 'IGKV3' entries from {df_subset_name}")
    
    print(f"High pairing score (≥{threshold}): {len(df_high)} samples")
    print(f"Low pairing score (<{threshold}): {len(df_low)} samples")
    
    # Generate confusion matrices for both subsets
    if len(df_high) > 0:
        print(f"\nGenerating confusion matrix for pairing score ≥ {threshold}...")
        generate_confusion_matrix_single(df_high, save_path_high, figsize, threshold, 
                                        true_col, gen_col, true_family_col, gen_family_col,
                                        use_rainbow, is_low=False)
    else:
        print(f"No data available for high pairing scores (≥{threshold})")
    
    if len(df_low) > 0:
        print(f"\nGenerating confusion matrix for pairing score < {threshold}...")
        generate_confusion_matrix_single(df_low, save_path_low, figsize, threshold, 
                                        true_col, gen_col, true_family_col, gen_family_col,
                                        use_rainbow, is_low=True)
    else:
        print(f"No data available for low pairing scores (<{threshold})")
    
    return df_high, df_low



def analyze_pairing_score_impact(df_all: pd.DataFrame, df_filtered: pd.DataFrame, 
                                min_pairing_score: float = 0.5,
                                true_col: str = "true_v_gene_family_call", 
                                gen_col: str = "gen_v_gene_family_call"):
    """
    Analyze the impact of pairing score filtering on prediction accuracy.
    
    Args:
        df_all: DataFrame with all data (before pairing score filtering)
        df_filtered: DataFrame after pairing score filtering
        min_pairing_score: The pairing score threshold used
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    print(f"\n" + "="*70)
    print(f"PAIRING SCORE FILTER IMPACT ANALYSIS")
    print("="*70)
    
    # Calculate accuracies
    if len(df_all) > 0:
        all_accuracy = (df_all[true_col] == df_all[gen_col]).mean()
    else:
        all_accuracy = 0
    
    if len(df_filtered) > 0:
        filtered_accuracy = (df_filtered[true_col] == df_filtered[gen_col]).mean()
    else:
        filtered_accuracy = 0
    
    # Print comparison
    print(f"All data (no pairing filter):")
    print(f"  Sample size: {len(df_all)}")
    print(f"  {gen_col.replace('_', ' ').title()} accuracy: {all_accuracy:.3f} ({all_accuracy*100:.1f}%)")
    
    print(f"\nFiltered data (pairing score ≥ {min_pairing_score}):")
    print(f"  Sample size: {len(df_filtered)}")
    print(f"  {gen_col.replace('_', ' ').title()} accuracy: {filtered_accuracy:.3f} ({filtered_accuracy*100:.1f}%)")
    
    if len(df_all) > 0:
        retention_rate = len(df_filtered) / len(df_all)
        print(f"  Data retention rate: {retention_rate:.3f} ({retention_rate*100:.1f}%)")
    
    accuracy_improvement = filtered_accuracy - all_accuracy
    print(f"\nAccuracy improvement: {accuracy_improvement:.3f} ({accuracy_improvement*100:.1f} percentage points)")
    
    if accuracy_improvement > 0:
        print("✓ Pairing score filtering IMPROVED prediction accuracy")
    elif accuracy_improvement < 0:
        print("✗ Pairing score filtering DECREASED prediction accuracy")
    else:
        print("→ Pairing score filtering had NO EFFECT on prediction accuracy")


def create_pairing_score_distribution_plot(df: pd.DataFrame, save_path: str = None,
                                         true_col: str = "true_v_gene_family_call", 
                                         gen_col: str = "gen_v_gene_family_call"):
    """
    Create plots showing pairing score distributions and their relationship to prediction accuracy.
    
    Args:
        df: DataFrame with pairing_scores and specified columns
        save_path: Optional path to save the plot
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    if 'pairing_scores' not in df.columns:
        print("No pairing_scores column found for distribution analysis.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Overall pairing score distribution
    ax1.hist(df['pairing_scores'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax1.set_xlabel('Pairing Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Pairing Score Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pairing scores: Correct vs Incorrect predictions
    correct_mask = df[true_col] == df[gen_col]
    correct_scores = df[correct_mask]['pairing_scores']
    incorrect_scores = df[~correct_mask]['pairing_scores']
    
    ax2.hist(correct_scores, bins=30, alpha=0.7, label=f'Correct (n={len(correct_scores)})', 
             color='green', density=True)
    ax2.hist(incorrect_scores, bins=30, alpha=0.7, label=f'Incorrect (n={len(incorrect_scores)})', 
             color='red', density=True)
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax2.set_xlabel('Pairing Score')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Pairing Scores: Correct vs Incorrect {gen_col.replace("_", " ").title()} Predictions', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy by pairing score bins
    score_bins = np.arange(0, 1.1, 0.1)
    bin_accuracies = []
    bin_counts = []
    bin_centers = []
    
    for i in range(len(score_bins)-1):
        bin_mask = (df['pairing_scores'] >= score_bins[i]) & (df['pairing_scores'] < score_bins[i+1])
        bin_df = df[bin_mask]
        
        if len(bin_df) > 0:
            bin_accuracy = (bin_df[true_col] == bin_df[gen_col]).mean()
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(len(bin_df))
            bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
    
    if bin_accuracies:
        bars = ax3.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='orange')
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
        ax3.set_xlabel('Pairing Score Bin')
        ax3.set_ylabel(f'{gen_col.replace("_", " ").title()} Accuracy')
        ax3.set_title(f'{gen_col.replace("_", " ").title()} Accuracy by Pairing Score Bin', fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 4. Box plot: Pairing scores by prediction accuracy
    box_data = [correct_scores, incorrect_scores]
    box_labels = ['Correct', 'Incorrect']
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax4.set_ylabel('Pairing Score')
    ax4.set_title(f'Pairing Score Distribution by {gen_col.replace("_", " ").title()} Prediction Accuracy', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=True)
        print(f"Pairing score analysis plot saved to: {save_path}")
    
    plt.show()

def generate_detailed_analysis(df: pd.DataFrame, true_col: str = "true_v_gene_family_call", 
                              gen_col: str = "gen_v_gene_family_call"):
    """
    Generate detailed analysis including classification report and match statistics.
    
    Args:
        df: DataFrame with specified columns
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    true_labels = df[true_col]
    pred_labels = df[gen_col]
    
    print("\n" + "="*60)
    print(f"DETAILED {gen_col.replace('_', ' ').upper()} ANALYSIS")
    print("="*60)
    
    # Overall statistics
    total_pairs = len(df)
    matches = (true_labels == pred_labels).sum()
    accuracy = matches / total_pairs
    
    print(f"\nOverall Statistics:")
    print(f"Total sequence pairs: {total_pairs}")
    print(f"Exact matches: {matches}")
    print(f"Mismatches: {total_pairs - matches}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Distribution of true labels
    print(f"\nTrue {true_col.replace('_', ' ').title()} Distribution:")
    true_dist = Counter(true_labels)
    for family, count in sorted(true_dist.items()):
        percentage = (count / total_pairs) * 100
        print(f"  {family}: {count} ({percentage:.1f}%)")
    
    # Distribution of generated labels
    print(f"\nGenerated {gen_col.replace('_', ' ').title()} Distribution:")
    pred_dist = Counter(pred_labels)
    for family, count in sorted(pred_dist.items()):
        percentage = (count / total_pairs) * 100
        print(f"  {family}: {count} ({percentage:.1f}%)")
    
    # Per-class accuracy
    print(f"\nPer-Class Performance:")
    unique_labels = sorted(list(set(true_labels.unique()) | set(pred_labels.unique())))
    
    for label in unique_labels:
        true_mask = (true_labels == label)
        pred_mask = (pred_labels == label)
        
        if true_mask.sum() > 0:  # Only if there are true instances of this class
            class_matches = ((true_labels == label) & (pred_labels == label)).sum()
            class_total = true_mask.sum()
            class_accuracy = class_matches / class_total
            
            # Also calculate precision and recall
            precision = class_matches / pred_mask.sum() if pred_mask.sum() > 0 else 0
            recall = class_accuracy  # Same as class accuracy in this context
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {label}:")
            print(f"    True instances: {class_total}")
            print(f"    Correctly predicted: {class_matches}")
            print(f"    Recall: {recall:.3f}")
            print(f"    Precision: {precision:.3f}")
            print(f"    F1-score: {f1:.3f}")
    
    # Classification report
    print(f"\nSklearn Classification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

def analyze_mismatches(df: pd.DataFrame, true_col: str = "true_v_gene_family_call", 
                      gen_col: str = "gen_v_gene_family_call"):
    """
    Analyze and display the mismatched predictions.
    
    Args:
        df: DataFrame with specified columns
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    mismatches = df[df[true_col] != df[gen_col]].copy()

    
    if len(mismatches) == 0:
        print("\nNo mismatches found - perfect accuracy!")
        return
    
    print(f"\n" + "="*60)
    print("MISMATCH ANALYSIS")
    print("="*60)
    
    print(f"\nTotal mismatches: {len(mismatches)}")
    
    # Most common mismatch patterns
    mismatch_patterns = mismatches.groupby([true_col, gen_col]).size().reset_index(name='count')
    mismatch_patterns = mismatch_patterns.sort_values('count', ascending=False)
    
    print(f"\nMost common mismatch patterns:")
    for _, row in mismatch_patterns.head(10).iterrows():
        true_fam = row[true_col]
        gen_fam = row[gen_col]
        count = row['count']
        print(f"  {true_fam} → {gen_fam}: {count} times")
    
    # Show some example mismatches with additional context
    print(f"\nExample mismatches (showing first 10):")
    # Update display_cols to use dynamic column names
    display_cols = []
    
    # Add identifier column if it exists
    id_cols = ['base_fasta_id', 'id', 'sequence_id', 'fasta_id']
    for id_col in id_cols:
        if id_col in mismatches.columns:
            display_cols.append(id_col)
            break
    
    # Add the main columns
    display_cols.extend([true_col, gen_col])
    
    # Add other relevant columns if they exist
    optional_cols = ['similarity', 'BLOSUM_score', 'pairing_scores']
    for col in optional_cols:
        if col in mismatches.columns:
            display_cols.append(col)
    
    available_cols = [col for col in display_cols if col in mismatches.columns]
    
    if available_cols:
        print(mismatches[available_cols].head(10).to_string(index=False))


def main(true_col: str = 'true_v_gene_family_call', gen_col: str = 'gen_v_gene_family_call',
         min_pairing_score: float = 0.5, family_filter: str = None, family_col: str = "true_v_gene_family_call"):
    """
    Main function to run the confusion matrix analysis with pairing score filter.
    
    Args:
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
        min_pairing_score: Minimum pairing score threshold
        family_filter: Optional V gene family to filter for (e.g. 'IGKV1', 'IGLV2')
    """
    #csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/updated_merged_pairing_vgene_result_bert2gpt_full_complete_ids_mapping_unique_nt_trimmed_gene_hit_locus.csv"
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/pairing_result_matching_seqs_multiple_light_seqs_203276_cls_predictions_parsed_reformatted_rel_cols_v_genes.csv"
    output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/maturity_matching_family_full_test_set_families"  

    # Add family filter to output directory name if specified
    if family_filter:
        output_dir = f"{output_dir}_{family_filter}"
    
    # Ensure output directory exists
    import os  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        family_text = f" (Family: {family_filter})" if family_filter else ""
        print("="*80)
        print(f"{gen_col.replace('_', ' ').upper()} CONFUSION MATRIX WITH PAIRING SCORE FILTER (≥ {min_pairing_score}){family_text}")
        print("="*80)
        
        # Load data without pairing filter for comparison
        print("Loading all data (no pairing filter)...")
        df_all = pd.read_csv(csv_file_path)
        df_all_clean = df_all.dropna(subset=[gen_col, true_col])
        
        # Remove IGHV3 if present
        if 'IGHV3' in df_all_clean[gen_col].values or 'IGHV3' in df_all_clean[true_col].values:
            df_all_clean = df_all_clean[
                (df_all_clean[gen_col] != 'IGHV3') & 
                (df_all_clean[true_col] != 'IGHV3')
            ]
        
        # Remove entries with only "IGKV1" (without suffix)
        # before_igkv1_filter = len(df_all_clean)
        # df_all_clean = df_all_clean[
        #     (df_all_clean[gen_col] != 'IGKV1') & 
        #     (df_all_clean[true_col] != 'IGKV1') &
        #     (df_all_clean[gen_col] != 'IGKV3') &
        #     (df_all_clean[true_col] != 'IGKV3')
        # ]
        # after_igkv1_filter = len(df_all_clean)
        # if before_igkv1_filter != after_igkv1_filter:
        #     print(f"Removed {before_igkv1_filter - after_igkv1_filter} entries with only 'IGKV1' and 'IGKV3' (without suffix) from all data")
        
        
        # Apply family filter to all data if specified
        if family_filter:
            # Use startswith to match all genes in the family
            df_all_clean = df_all_clean[
                df_all_clean[true_col].str.startswith(family_filter) & 
                df_all_clean[gen_col].str.startswith(family_filter)
            ]
            print(f"After applying family filter {family_filter}: {len(df_all_clean)} samples")
        
        
        # Load data with pairing filter
        print(f"\nLoading data with pairing score filter (≥ {min_pairing_score})...")
        df_filtered = load_and_clean_data_with_pairing_filter(csv_file_path, min_pairing_score, true_col, gen_col, family_col, family_filter)
        
        if len(df_filtered) == 0:
            print("No data available after filtering. Please check your data and parameters.")
            return
        
        # Analyze impact of pairing filter
        analyze_pairing_score_impact(df_all_clean, df_filtered, min_pairing_score, true_col, gen_col)
        
        # Generate confusion matrix for filtered data
        print(f"\nGenerating confusion matrices for pairing score threshold {min_pairing_score}...")
        
        # Update file names to include family filter
        family_suffix = f"_{family_filter}" if family_filter else ""
        
        # Generate matrices for high vs low pairing scores with Spectral colormap
        print("=== SPECTRAL COLORMAP ===")
        df_high_spectral, df_low_spectral = generate_dual_pairing_confusion_matrices(
            df_all_clean, min_pairing_score,
            save_path_high=f"{output_dir}/spectral_high_pairing_{gen_col}_min_{min_pairing_score}{family_suffix}.png",
            save_path_low=f"{output_dir}/spectral_low_pairing_{gen_col}_min_{min_pairing_score}{family_suffix}.png",
            true_col=true_col, gen_col=gen_col, use_rainbow=False, family_filter=family_filter
        )
        
        # Generate matrices for high vs low pairing scores with Rainbow colormap
        print("\n=== RAINBOW COLORMAP ===")
        df_high_rainbow, df_low_rainbow = generate_dual_pairing_confusion_matrices(
            df_all_clean, min_pairing_score,
            save_path_high=f"{output_dir}/rainbow_high_pairing_{gen_col}_min_{min_pairing_score}{family_suffix}.png",
            save_path_low=f"{output_dir}/rainbow_low_pairing_{gen_col}_min_{min_pairing_score}{family_suffix}.png",
            true_col=true_col, gen_col=gen_col, use_rainbow=True, family_filter=family_filter
        )
        
        # Generate detailed analysis
        print(f"\nDetailed analysis for pairing score ≥ {min_pairing_score}...")
        generate_detailed_analysis(df_filtered, true_col, gen_col)
        analyze_mismatches(df_filtered, true_col, gen_col)
        
        # Create pairing score analysis plots
        print("\nCreating pairing score distribution analysis...")
        create_pairing_score_distribution_plot(df_all_clean, 
                                             f"{output_dir}/pairing_score_analysis_{gen_col}_min_pairing_{min_pairing_score}{family_suffix}.png",
                                             true_col, gen_col)
        
        print(f"\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Function to test different pairing score thresholds
def test_multiple_thresholds(csv_file_path: str, thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                           true_col: str = "true_v_gene_family_call", 
                           gen_col: str = "gen_v_gene_family_call"):
    """
    Test multiple pairing score thresholds and compare accuracies.
    
    Args:
        csv_file_path: Path to the merged CSV file
        thresholds: List of pairing score thresholds to test
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
    """
    print("="*80)
    print("TESTING MULTIPLE PAIRING SCORE THRESHOLDS")
    print("="*80)
    
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        df_filtered = load_and_clean_data_with_pairing_filter(csv_file_path, threshold, true_col, gen_col)
        
        if len(df_filtered) > 0:
            accuracy = (df_filtered[true_col] == df_filtered[gen_col]).mean()
            results.append({
                'threshold': threshold,
                'sample_size': len(df_filtered),
                'accuracy': accuracy
            })
            print(f"  Sample size: {len(df_filtered)}, Accuracy: {accuracy:.3f}")
        else:
            print(f"  No data available at threshold {threshold}")
    
    # Summary table
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nSUMMARY OF THRESHOLD TESTING:")
        print(results_df.to_string(index=False))
        
        # Find optimal threshold
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_threshold = results_df.loc[best_accuracy_idx, 'threshold']
        best_accuracy = results_df.loc[best_accuracy_idx, 'accuracy']
        
        print(f"\nBest threshold: {best_threshold} (accuracy: {best_accuracy:.3f})")

if __name__ == "__main__":
    # Example usage with different columns:
    
    # For V gene family analysis (default)
    #main()

    # Analyze only IGKV1 family
    main()
    #main(family_filter='IGKV1')



    # For V gene simple analysis
    # main(true_col='true_v_gene_simple', gen_col='gen_v_gene_simple')
    
    # For J gene analysis
    # main(true_col='true_j_gene_call', gen_col='gen_j_gene_call')
    
    # Test multiple thresholds for V gene family
    # test_multiple_thresholds("merged_pairing_vgene.csv")
    
    # Test multiple thresholds for V gene simple
    # test_multiple_thresholds("merged_pairing_vgene.csv", 
    #                         true_col='true_v_gene_simple', 
    #                         gen_col='gen_v_gene_simple')
