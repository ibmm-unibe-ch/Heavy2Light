import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import warnings
from scipy import stats
import re
from matplotlib.colors import ListedColormap
import colorcet as cc

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

def load_and_clean_data(csv_file_path: str, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call") -> pd.DataFrame:
    """
    Load the CSV file and clean the data for analysis.
    
    Args:
        csv_file_path: Path to the updated mapping CSV file
        
    Returns:
        Cleaned DataFrame with non-null V gene family calls
    """
    df = pd.read_csv(csv_file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Rows with {gen_col}: {df[gen_col].notna().sum()}")
    print(f"Rows with {true_col}: {df[true_col].notna().sum()}")
    
    # Keep only rows where both gen and true V gene family calls are available
    clean_df = df.dropna(subset=[gen_col, true_col])

    # Remove IGHV3 
    clean_df = clean_df[
        (clean_df[gen_col] != 'IGHV3') & 
        (clean_df[true_col] != 'IGHV3')
    ]
    
    print(f"Clean dataset shape (both calls available): {clean_df.shape}")
    
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


def generate_confusion_matrix(df: pd.DataFrame, true_col: str, gen_col: str, 
                            save_path: str = None, figsize: tuple = (14, 12), 
                            show_values: bool = True, 
                            true_family_col: str = "true_v_gene_family_call",
                            gen_family_col: str = "gen_v_gene_family_call"):
    """
    Generate and display confusion matrix with family-based colors and value-based transparency.
    """
    # Filter out rows with IGH genes
    df_filtered = df[~df[true_col].str.contains('IGH', na=False) & 
                     ~df[gen_col].str.contains('IGH', na=False)].copy()
    
    print(f"Filtered out IGH genes. Remaining rows: {len(df_filtered)} (was {len(df)})")
    
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
    
    # Create family color mapping - use seaborn HLS palette for distinct colors
    #family_colors = sns.color_palette("hls", len(unique_families))
    family_colors = plt.cm.jet(np.linspace(0, 1, len(unique_families)))

    family_color_map = {family: family_colors[i] for i, family in enumerate(unique_families)}
    
    # Create custom colored matrix with transparency based on values
    colored_matrix = np.zeros((len(all_labels), len(all_labels), 4))  # RGBA
    
    # Normalize confusion matrix values for transparency (0-1 range)
    cm_normalized = cm / cm.max() if cm.max() > 0 else cm
    
    # Apply power scaling to increase contrast - makes low values much more transparent
    cm_contrast = np.power(cm_normalized, 0.3)  # Square the values for more contrast
    
    for i, true_gene in enumerate(all_labels):
        for j, pred_gene in enumerate(all_labels):
            # Get family for true gene (row)
            true_family = true_gene_family_map.get(true_gene)
            
            if true_family and true_family in family_color_map:
                base_color = family_color_map[true_family][:3]  # RGB only
                # Increased transparency range and contrast
                alpha = 0.1 + 0.9 * cm_contrast[i, j]  # Range from 0.1 to 1.0 with more contrast
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
            # Add rectangle border around each cell
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                               fill=False, edgecolor='white', 
                               linewidth=2, alpha=1)
            ax.add_patch(rect)
    
    # Remove all spines and ticks styling for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set labels and ticks with clean styling
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=90, ha='center', fontsize=10)
    ax.set_yticklabels(all_labels, rotation=0, ha='right', fontsize=10)
    
    # Add text annotations if requested
    if show_values:
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                # Choose text color based on background brightness
                alpha = colored_matrix[i, j, 3]
                avg_color = np.mean(colored_matrix[i, j, :3])
                text_color = 'white' if (alpha * avg_color) < 0.5 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Clean, minimal title and labels
    ax.set_title(f'Confusion Matrix (Family-Colored)\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'Predicted {gen_col.replace("_", " ").title()}', 
                 fontsize=14, fontweight='normal')
    ax.set_ylabel(f'True {true_col.replace("_", " ").title()}', 
                 fontsize=14, fontweight='normal')
    
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
             title='V Gene Families', title_fontsize=20, fontsize=18)
    
    # Remove tick marks for cleaner look
    ax.tick_params(axis='both', which='both', length=0)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, all_labels, accuracy


def generate_detailed_analysis(df: pd.DataFrame, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call"):
    """
    Generate detailed analysis including classification report and match statistics.
    
    Args:
        df: DataFrame with V gene family calls
    """
    true_labels = df[true_col]
    pred_labels = df[gen_col]
    
    print("\n" + "="*60)
    print("DETAILED V GENE FAMILY ANALYSIS")
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
    print(f"\nTrue V Gene Family Distribution:")
    true_dist = Counter(true_labels)
    for family, count in sorted(true_dist.items()):
        percentage = (count / total_pairs) * 100
        print(f"  {family}: {count} ({percentage:.1f}%)")
    
    # Distribution of generated labels
    print(f"\nGenerated V Gene Family Distribution:")
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

def analyze_mismatches(df: pd.DataFrame, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call"):
    """
    Analyze and display the mismatched predictions.
    
    Args:
        df: DataFrame with V gene family calls
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
    display_cols = ['base_fasta_id', true_col, gen_col]
    
    # Add other columns if they exist
    if true_col.replace('family_call', 'call') in mismatches.columns:
        display_cols.append(true_col.replace('family_call', 'call'))
    if gen_col.replace('family_call', 'call') in mismatches.columns:
        display_cols.append(gen_col.replace('family_call', 'call'))
    
    available_cols = [col for col in display_cols if col in mismatches.columns]
    if 'similarity' in mismatches.columns:
        available_cols.append('similarity')
    if 'BLOSUM_score' in mismatches.columns:
        available_cols.append('BLOSUM_score')
    
    print(mismatches[available_cols].head(10).to_string(index=False))

def create_accuracy_pie_chart(df: pd.DataFrame, true_col: str, gen_col: str, 
                            save_path: str = None, figsize: tuple = (8, 8)):
    """Create accuracy pie chart."""
    matches = (df[true_col] == df[gen_col]).sum()
    mismatches = len(df) - matches
    accuracy = matches / len(df)
    
    plt.figure(figsize=figsize)
    plt.pie([matches, mismatches], 
            labels=[f'Matches\n({matches})', f'Mismatches\n({mismatches})'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title(f'Overall Prediction Accuracy\n{accuracy:.3f} ({accuracy*100:.1f}%)', 
              fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy pie chart saved to: {save_path}")
    
    plt.show()

def create_true_distribution_plot(df: pd.DataFrame, true_col: str, 
                                save_path: str = None, figsize: tuple = (12, 8)):
    """Create true label distribution plot."""
    true_dist = df[true_col].value_counts()
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(true_dist)), true_dist.values, color='skyblue', alpha=0.8)
    plt.xticks(range(len(true_dist)), true_dist.index, rotation=45, ha='right')
    plt.title(f'True {true_col.replace("_", " ").title()} Distribution', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Count', fontsize=14)
    plt.xlabel(true_col.replace("_", " ").title(), fontsize=14)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(true_dist.values)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"True distribution plot saved to: {save_path}")
    
    plt.show()

def create_generated_distribution_plot(df: pd.DataFrame, gen_col: str, 
                                     save_path: str = None, figsize: tuple = (12, 8)):
    """Create generated label distribution plot."""
    pred_dist = df[gen_col].value_counts()
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(pred_dist)), pred_dist.values, color='lightcoral', alpha=0.8)
    plt.xticks(range(len(pred_dist)), pred_dist.index, rotation=45, ha='right')
    plt.title(f'Generated {gen_col.replace("_", " ").title()} Distribution', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Count', fontsize=14)
    plt.xlabel(gen_col.replace("_", " ").title(), fontsize=14)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(pred_dist.values)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generated distribution plot saved to: {save_path}")
    
    plt.show()

def create_per_class_accuracy_plot(df: pd.DataFrame, true_col: str, gen_col: str, 
                                 save_path: str = None, figsize: tuple = (14, 8)):
    """Create per-class accuracy plot."""
    unique_labels = sorted(df[true_col].unique())
    accuracies = []
    counts = []
    
    for label in unique_labels:
        label_df = df[df[true_col] == label]
        if len(label_df) > 0:
            acc = (label_df[true_col] == label_df[gen_col]).mean()
            accuracies.append(acc)
            counts.append(len(label_df))
        else:
            accuracies.append(0)
            counts.append(0)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(unique_labels)), accuracies, color='gold', alpha=0.8)
    plt.xticks(range(len(unique_labels)), unique_labels, rotation=45, ha='right')
    plt.title(f'Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel(true_col.replace("_", " ").title(), fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on bars with sample counts
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    
    plt.show()

# 3. Create wrapper function to generate all 4 summary plots
def create_all_summary_plots(df: pd.DataFrame, true_col: str, gen_col: str, 
                           output_dir: str = None, figsize: tuple = (12, 8)):
    """
    Create all 4 summary plots as separate figures.
    
    Args:
        df: DataFrame with specified columns
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
        output_dir: Directory to save plots (if None, plots won't be saved)
        figsize: Default figure size for plots
    """
    print("\nGenerating summary plots...")
    
    # Generate save paths if output directory is provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        accuracy_path = f"{output_dir}/accuracy_pie_chart.png"
        true_dist_path = f"{output_dir}/true_distribution.png"
        gen_dist_path = f"{output_dir}/generated_distribution.png"
        per_class_path = f"{output_dir}/per_class_accuracy.png"
    else:
        accuracy_path = true_dist_path = gen_dist_path = per_class_path = None
    
    # Create individual plots
    create_accuracy_pie_chart(df, true_col, gen_col, accuracy_path, figsize)
    create_true_distribution_plot(df, true_col, true_dist_path, figsize)
    create_generated_distribution_plot(df, gen_col, gen_dist_path, figsize)
    create_per_class_accuracy_plot(df, true_col, gen_col, per_class_path, figsize)



def analyze_vgene_vs_similarity(df: pd.DataFrame, save_path: str = None, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call", figsize: tuple = (16, 12)):
    """
    Analyze and plot V gene family accuracies vs calculated_similarity values.
    
    Args:
        df: DataFrame with V gene family calls and calculated_similarity
        save_path: Optional path to save the plot
    """
    print("\n" + "="*60)
    print("V GENE FAMILY vs SIMILARITY ANALYSIS")
    print("="*60)
    
    # Calculate per-family statistics
    family_stats = []
    
    unique_families = sorted(df[true_col].unique())
    
    for family in unique_families:
        family_df = df[df[true_col] == family]
        
        if len(family_df) > 0:
            # V gene family accuracy
            matches = (family_df[true_col] == family_df[gen_col]).sum()
            v_gene_accuracy = matches / len(family_df)
            
            # Similarity statistics
            if 'calculated_similarity' in family_df.columns:
                avg_similarity = family_df['calculated_similarity'].mean()
                median_similarity = family_df['calculated_similarity'].median()
                std_similarity = family_df['calculated_similarity'].std()
            else:
                avg_similarity = median_similarity = std_similarity = np.nan
            
            family_stats.append({
                'family': family,
                'count': len(family_df),
                'v_gene_accuracy': v_gene_accuracy,
                'avg_similarity': avg_similarity,
                'median_similarity': median_similarity,
                'std_similarity': std_similarity
            })
    
    stats_df = pd.DataFrame(family_stats)
    
    # Print statistics table
    print(f"\nPer-Family Statistics:")
    print(f"{'Family':<10} {'Count':<8} {'V Gene Acc':<12} {'Avg Sim':<10} {'Med Sim':<10} {'Std Sim':<10}")
    print("-" * 70)
    
    for _, row in stats_df.iterrows():
        print(f"{row['family']:<10} {row['count']:<8} {row['v_gene_accuracy']:<12.3f} "
              f"{row['avg_similarity']:<10.3f} {row['median_similarity']:<10.3f} {row['std_similarity']:<10.3f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. V Gene Family Accuracy by Family
    bars1 = ax1.bar(range(len(stats_df)), stats_df['v_gene_accuracy'], 
                    color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(stats_df)))
    ax1.set_xticklabels(stats_df['family'], rotation=45, ha='right')
    ax1.set_ylabel('V Gene Family Accuracy')
    ax1.set_title('V Gene Family Prediction Accuracy by Family', fontweight='bold')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average Similarity by Family
    if not stats_df['avg_similarity'].isna().all():
        bars2 = ax2.bar(range(len(stats_df)), stats_df['avg_similarity'], 
                        color='orange', alpha=0.7)
        ax2.set_xticks(range(len(stats_df)))
        ax2.set_xticklabels(stats_df['family'], rotation=45, ha='right')
        ax2.set_ylabel('Average Calculated Similarity')
        ax2.set_title('Average Sequence Similarity by V Gene Family', fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No similarity data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Average Sequence Similarity by V Gene Family', fontweight='bold')
    
    # 3. Scatter plot: V Gene Accuracy vs Average Similarity
    if not stats_df['avg_similarity'].isna().all():
        scatter = ax3.scatter(stats_df['avg_similarity'], stats_df['v_gene_accuracy'], 
                             s=stats_df['count']*3, alpha=0.6, c=range(len(stats_df)), cmap='viridis')
        
        # Add family labels
        for i, row in stats_df.iterrows():
            if not np.isnan(row['avg_similarity']):
                ax3.annotate(row['family'], (row['avg_similarity'], row['v_gene_accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Average Calculated Similarity')
        ax3.set_ylabel('V Gene Family Accuracy')
        ax3.set_title('V Gene Accuracy vs Sequence Similarity\n(Bubble size = sample count)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No similarity data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('V Gene Accuracy vs Sequence Similarity', fontweight='bold')
    
    # 4. Sample size by family
    bars4 = ax4.bar(range(len(stats_df)), stats_df['count'], 
                    color='lightcoral', alpha=0.7)
    ax4.set_xticks(range(len(stats_df)))
    ax4.set_xticklabels(stats_df['family'], rotation=45, ha='right')
    ax4.set_ylabel('Sample Count')
    ax4.set_title('Sample Size by V Gene Family', fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(stats_df['count'])*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"V Gene vs Similarity analysis plot saved to: {save_path}")
    
    plt.show()
    
    # Calculate correlations
    if not stats_df['avg_similarity'].isna().all():
        # Remove NaN values for correlation
        clean_stats = stats_df.dropna(subset=['avg_similarity', 'v_gene_accuracy'])
        
        if len(clean_stats) > 1:
            correlation = clean_stats['avg_similarity'].corr(clean_stats['v_gene_accuracy'])
            print(f"\nCorrelation between average similarity and V gene accuracy: {correlation:.3f}")
            
            # Weighted correlation (by sample size)
            weights = clean_stats['count']
            weighted_corr = np.corrcoef(clean_stats['avg_similarity'], clean_stats['v_gene_accuracy'])[0, 1]
            print(f"Sample-weighted correlation: {weighted_corr:.3f}")
    
    return stats_df

def create_similarity_distribution_plot(df: pd.DataFrame, save_path: str = None, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call"):
    """
    Create plots showing similarity distributions for correct vs incorrect V gene predictions.
    
    Args:
        df: DataFrame with V gene family calls and calculated_similarity
        save_path: Optional path to save the plot
    """
    if 'calculated_similarity' not in df.columns:
        print("No calculated_similarity column found for distribution analysis.")
        return
    
    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Separate correct and incorrect predictions
    correct_mask = df[true_col] == df[gen_col]
    correct_similarities = df[correct_mask]['calculated_similarity']
    incorrect_similarities = df[~correct_mask]['calculated_similarity']
    
    print(f"Correct predictions: {len(correct_similarities)} (avg similarity: {correct_similarities.mean():.3f})")
    print(f"Incorrect predictions: {len(incorrect_similarities)} (avg similarity: {incorrect_similarities.mean():.3f})")
    
    # Statistical test
    from scipy import stats
    statistic, p_value = stats.mannwhitneyu(correct_similarities, incorrect_similarities, alternative='two-sided')
    print(f"Mann-Whitney U test p-value: {p_value:.6f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Histogram comparison
    ax1.hist(correct_similarities, bins=30, alpha=0.7, label=f'Correct (n={len(correct_similarities)})', 
             color='green', density=True)
    ax1.hist(incorrect_similarities, bins=30, alpha=0.7, label=f'Incorrect (n={len(incorrect_similarities)})', 
             color='red', density=True)
    ax1.set_xlabel('Calculated Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('Similarity Distribution: Correct vs Incorrect V Gene Predictions', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    box_data = [correct_similarities, incorrect_similarities]
    box_labels = ['Correct', 'Incorrect']
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax2.set_ylabel('Calculated Similarity')
    ax2.set_title('Similarity Distribution Box Plot', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity distribution plot saved to: {save_path}")
    
    plt.show()


def main(true_col: str = 'true_v_gene_family_call', gen_col: str = 'gen_v_gene_family_call',
         show_cm_values: bool = False, summary_figsize: tuple = (55, 44), matrix_figsize = (28, 24)):
    """
    Main function to run the confusion matrix analysis.
    
    Args:
        true_col: Column name for true/reference values
        gen_col: Column name for generated/predicted values
        show_cm_values: Whether to show values in confusion matrix cells
        summary_figsize: Figure size for summary plots
        matrix_figsize: Figure size for confusion matrix
    """
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/all_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping.csv"
    output_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs"
    confusion_matrix_save_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/rainbow_v_gene_family_call_compl_no_d_v_gene_family_confusion_matrix.png"  
        

    print("Loading and cleaning data...")
    df = load_and_clean_data(csv_file_path, true_col, gen_col)
        
    if len(df) == 0:
        print("No data available for analysis. Please check that your CSV file contains the required columns.")
        return
        
    print("\nGenerating confusion matrix...")
    cm, labels, accuracy = generate_confusion_matrix(df, true_col, gen_col, 
                                                        confusion_matrix_save_path,
                                                        matrix_figsize, 
                                                        show_values=show_cm_values)
        
    print("\nGenerating detailed analysis...")
    generate_detailed_analysis(df, true_col, gen_col)
        
    print("\nAnalyzing mismatches...")
    analyze_mismatches(df, true_col, gen_col)
        
    print("\nGenerating summary plots...")
    create_all_summary_plots(df, true_col, gen_col, output_path, summary_figsize)
        
    print("\nAnalyzing vs Similarity...")
    analyze_vgene_vs_similarity(df, f"{output_path}/rainbow_colored_fam_v_gene_family_call_compl_no_d_v_gene_family_analysis.png", true_col, gen_col, summary_figsize)
        
    print("\nAnalyzing Similarity Distributions...")
    create_similarity_distribution_plot(df, f"{output_path}/rainbow_colored_fam_v_gene_family_call_compl_no_d_v_gene_family_similarity_distribution.png", true_col, gen_col)
        
    print(f"\nAnalysis completed successfully!")
    print(f"Overall Prediction Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
if __name__ == "__main__":
    main()