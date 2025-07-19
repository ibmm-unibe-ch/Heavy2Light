import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import re

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


def load_and_clean_data(csv_file_path: str, btype_filter: str = None, true_col: str = "true_v_gene_family_call", gen_col: str = "gen_v_gene_family_call") -> pd.DataFrame:
    """
    Load the CSV file and clean the data for analysis.
    
    Args:
        csv_file_path: Path to the updated mapping CSV file
        btype_filter: Optional filter for BType column ('Naive-B-Cells', 'Memory-B-Cells', or None for all)
        
    Returns:
        Cleaned DataFrame with non-null V gene family calls
    """
    df = pd.read_csv(csv_file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Rows with {gen_col}: {df[gen_col].notna().sum()}")
    print(f"Rows with {true_col}: {df[true_col].notna().sum()}")
    
    # Filter by BType if specified
    if btype_filter:
        if 'BType' not in df.columns:
            print(f"Warning: BType column not found in dataset!")
            return pd.DataFrame()
        
        df = df[df['BType'] == btype_filter]
        print(f"After filtering for {btype_filter}: {df.shape}")

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



def generate_confusion_matrix(df: pd.DataFrame, 
                            save_path: str = None, 
                            figsize: tuple = (10, 8), 
                            title_suffix: str = "", 
                            show_values: bool = True, 
                            show_family_rectangles: bool = True,  
                            tile_border_width: float = 0.5,      
                            true_col: str = "true_v_gene_family_call",
                            gen_col: str = "gen_v_gene_family_call",
                            true_family_col: str = "true_v_gene_family_call",
                            gen_family_col: str = "gen_v_gene_family_call"):
    """
    Generate and display confusion matrix for V gene family calls.
    Args:
        df: DataFrame with gen_v_gene_family_call and true_v_gene_family_call columns
        save_path: Optional path to save the confusion matrix plot
        figsize: Figure size for the plot
        title_suffix: Additional text to add to the title
    """

    # Filter out rows with IGH genes
    df_filtered = df[~df[true_col].str.contains('IGH', na=False) & 
                     ~df[gen_col].str.contains('IGH', na=False)].copy()
    
    print(f"Filtered out IGH genes. Remaining rows: {len(df_filtered)} (was {len(df)})")
    
    true_labels = df_filtered[true_col]
    pred_labels = df_filtered[gen_col]

    # Get gene to family mappings
    true_gene_family_map, _ = get_family_from_gene(df_filtered, true_col, true_family_col)
    gen_gene_family_map, unique_families = get_family_from_gene(df_filtered, gen_col, gen_family_col)
    
    
    # Get all unique labels and sort them by chain type and family
    all_labels_unsorted = list(set(true_labels.unique()) | set(pred_labels.unique()))
    all_labels = sort_v_genes_by_chain_and_family(all_labels_unsorted)
    
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Create color mapping for V gene families
    family_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_families)))
    family_color_map = {family: family_colors[i] for i, family in enumerate(unique_families)}
    
    base_cmap = plt.cm.Viridis
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(cm, cmap=base_cmap, aspect='auto')

    # Add family-based coloring by drawing rectangles
    ax = plt.gca()

    # Add borders around each tile if requested
    if tile_border_width > 0:
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                   fill=False, edgecolor='white', 
                                   linewidth=tile_border_width, alpha=1.0)
                ax.add_patch(rect)
    
    # Group genes by family for coloring
    current_family = None
    start_idx = 0
    
    if show_family_rectangles:
        # Group genes by family for coloring
        current_family = None
        start_idx = 0
        
        for i, gene in enumerate(all_labels):
            # Get family for this gene
            gene_family = true_gene_family_map.get(gene) or gen_gene_family_map.get(gene)
            
            if gene_family != current_family:
                if current_family is not None and current_family in family_color_map:
                    # Draw rectangle for previous family group
                    rect_color = family_color_map[current_family]
                    
                    # Add colored border for this family group
                    rect = plt.Rectangle((start_idx-0.5, start_idx-0.5), 
                                       i-start_idx, i-start_idx,
                                       fill=False, edgecolor=rect_color, 
                                       linewidth=3, alpha=0.8)
                    ax.add_patch(rect)
                
                current_family = gene_family
                start_idx = i
        
        if current_family is not None and current_family in family_color_map:
            rect_color = family_color_map[current_family]
            rect = plt.Rectangle((start_idx-0.5, start_idx-0.5), 
                               len(all_labels)-start_idx, len(all_labels)-start_idx,
                               fill=False, edgecolor=rect_color, 
                               linewidth=3, alpha=0.8)
            ax.add_patch(rect)
    
    # Add annotations if requested
    if show_values:
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=8)
    
    # Set labels and ticks
    plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right', fontsize=20)  # Change fontsize here
    plt.yticks(range(len(all_labels)), all_labels, rotation=0, fontsize=20)  
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Add title and labels
    plt.title(f'Confusion Matrix (V Gene Family Ordered) {title_suffix}\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
              fontsize=24, fontweight='bold')
    plt.xlabel(f'Generated V gene', fontsize=22, fontweight='bold')
    plt.ylabel(f'True V gene', fontsize=22, fontweight='bold')
    
    # Create legend for family colors
    legend_elements = []
    for family in unique_families:
        if family in family_color_map:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=family_color_map[family], 
                                               edgecolor='black', alpha=0.8, 
                                               label=family))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
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

def create_summary_statistics_plot(df: pd.DataFrame, true_col: str, gen_col: str, save_path: str = None):
    """
    Create a summary plot showing accuracy and distribution statistics.
    
    Args:
        df: DataFrame with V gene family calls
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall accuracy pie chart
    matches = (df[true_col] == df[gen_col]).sum()
    mismatches = len(df) - matches
    
    ax1.pie([matches, mismatches], 
            labels=[f'Matches\n({matches})', f'Mismatches\n({mismatches})'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Overall V Gene Family Prediction Accuracy', fontweight='bold')
    
    # 2. True label distribution
    true_dist = df[true_col].value_counts()
    ax2.bar(range(len(true_dist)), true_dist.values, color='skyblue')
    ax2.set_xticks(range(len(true_dist)))
    ax2.set_xticklabels(true_dist.index, rotation=45, ha='right')
    ax2.set_title('True V Gene Family Distribution', fontweight='bold')
    ax2.set_ylabel('Count')
    
    # 3. Generated label distribution
    pred_dist = df[gen_col].value_counts()
    ax3.bar(range(len(pred_dist)), pred_dist.values, color='lightcoral')
    ax3.set_xticks(range(len(pred_dist)))
    ax3.set_xticklabels(pred_dist.index, rotation=45, ha='right')
    ax3.set_title('Generated V Gene Family Distribution', fontweight='bold')
    ax3.set_ylabel('Count')
    
    # 4. Per-class accuracy
    unique_labels = sorted(df[true_col].unique())
    accuracies = []
    
    for label in unique_labels:
        label_df = df[df[true_col] == label]
        if len(label_df) > 0:
            acc = (label_df[true_col] == label_df[gen_col]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    bars = ax4.bar(range(len(unique_labels)), accuracies, color='gold')
    ax4.set_xticks(range(len(unique_labels)))
    ax4.set_xticklabels(unique_labels, rotation=45, ha='right')
    ax4.set_title('Per-Class Accuracy', fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary statistics plot saved to: {save_path}")
    
    plt.show()

def main(true_col: str = 'true_v_gene_simple', gen_col: str = 'gen_v_gene_simple',
         show_cm_values: bool = False, summary_figsize: tuple = (55, 44), matrix_figsize = (35, 25), tile_border_width: float = 0):
    """
    Main function to run the confusion matrix analysis for both B cell types.
    """
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/all_v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping_merged_dataset_naive_memory.csv"

    output_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/naive_memory_all_genes"

    # make directory if it doesn't exist
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        print("="*80)
        print("V GENE FAMILY CONFUSION MATRIX ANALYSIS BY B CELL TYPE")
        print("="*80)
        
        # Analysis for Naive B Cells
        print("\n" + "="*50)
        print("NAIVE B CELLS ANALYSIS")
        print("="*50)
        
        naive_df = load_and_clean_data(csv_file_path=csv_file_path, btype_filter="Naive-B-Cells", true_col=true_col, gen_col=gen_col)
        
        if len(naive_df) > 0:
            print(f"\nGenerating confusion matrix for Naive B Cells...")
            cm_naive, labels_naive, accuracy_naive = generate_confusion_matrix(
            df=naive_df, 
            save_path=f"{output_path}/no_rect_naive_b_cells_confusion_matrix.png",
            title_suffix=" - Naive B Cells",
            true_col=true_col,
            gen_col=gen_col,
            show_values=show_cm_values,
            figsize=matrix_figsize,
            show_family_rectangles=False,  
            tile_border_width=tile_border_width,        
        )
            
            print(f"\nDetailed analysis for Naive B Cells...")
            generate_detailed_analysis(naive_df)
            
            print(f"\nMismatch analysis for Naive B Cells...")
            analyze_mismatches(naive_df)
            
            print(f"\nSummary plot for Naive B Cells...")
            create_summary_statistics_plot(naive_df, true_col, gen_col, "naive_b_cells_summary.png")
        else:
            print("No data available for Naive B Cells analysis.")
        
        # Analysis for Memory B Cells
        print("\n" + "="*50)
        print("MEMORY B CELLS ANALYSIS")
        print("="*50)
        
        memory_df = load_and_clean_data(csv_file_path, btype_filter="Memory-B-Cells")
        
        if len(memory_df) > 0:
            print(f"\nGenerating confusion matrix for Memory B Cells...")
            cm_memory, labels_memory, accuracy_memory = generate_confusion_matrix(
                df=memory_df, 
                save_path=f"{output_path}/no_rect_memory_b_cells_confusion_matrix.png",
                title_suffix=" - Memory B Cells",
                true_col=true_col,
                gen_col=gen_col,
                show_values=show_cm_values,
                figsize=matrix_figsize,
                show_family_rectangles=False,  
                tile_border_width=tile_border_width,       
            )
            
            print(f"\nDetailed analysis for Memory B Cells...")
            generate_detailed_analysis(memory_df)
            
            print(f"\nMismatch analysis for Memory B Cells...")
            analyze_mismatches(memory_df)
            
            print(f"\nSummary plot for Memory B Cells...")
            create_summary_statistics_plot(memory_df, true_col, gen_col, "memory_b_cells_summary.png")
        else:
            print("No data available for Memory B Cells analysis.")
        
        # Comparison summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        if len(naive_df) > 0 and len(memory_df) > 0:
            print(f"Naive B Cells:")
            print(f"  Sample size: {len(naive_df)}")
            print(f"  Accuracy: {accuracy_naive:.3f} ({accuracy_naive*100:.1f}%)")
            
            print(f"\nMemory B Cells:")
            print(f"  Sample size: {len(memory_df)}")
            print(f"  Accuracy: {accuracy_memory:.3f} ({accuracy_memory*100:.1f}%)")
            
            print(f"\nDifference in accuracy: {abs(accuracy_naive - accuracy_memory):.3f}")
            
            if accuracy_naive > accuracy_memory:
                print("Naive B Cells have higher prediction accuracy.")
            elif accuracy_memory > accuracy_naive:
                print("Memory B Cells have higher prediction accuracy.")
            else:
                print("Both cell types have equal prediction accuracy.")
        
        print(f"\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # Additional function to run analysis for specific B cell type
# def analyze_btype(csv_file_path: str, btype: str):
#     """
#     Run analysis for a specific B cell type.
    
#     Args:
#         csv_file_path: Path to the CSV file
#         btype: B cell type to analyze ("Naive-B-Cells" or "Memory-B-Cells")
#     """
#     print(f"Analyzing {btype}...")
    
#     df = load_and_clean_data(csv_file_path, btype_filter=btype)
    
#     if len(df) == 0:
#         print(f"No data available for {btype}")
#         return
    
#     # Generate confusion matrix
#     cm, labels, accuracy = generate_confusion_matrix(
#         df, 
#         save_path=f"{btype.lower().replace('-', '_')}_confusion_matrix.png",
#         title_suffix=f" - {btype}"
#     )
    
#     # Detailed analysis
#     generate_detailed_analysis(df)
#     analyze_mismatches(df)
#     create_summary_statistics_plot(df, f"{btype.lower().replace('-', '_')}_summary.png")
    
#     return df, accuracy

if __name__ == "__main__":
    main()

# Additional utility function for quick analysis
# def quick_analysis(csv_file_path: str):
#     """
#     Quick analysis function that just prints basic statistics.
    
#     Args:
#         csv_file_path: Path to the CSV file
#     """
#     df = pd.read_csv(csv_file_path)
#     clean_df = df.dropna(subset=['gen_v_gene_family_call', 'true_v_gene_family_call'])
    
#     if len(clean_df) == 0:
#         print("No valid data for analysis.")
#         return
    
#     matches = (clean_df['true_v_gene_family_call'] == clean_df['gen_v_gene_family_call']).sum()
#     total = len(clean_df)
#     accuracy = matches / total
    
#     print(f"Quick V Gene Family Analysis:")
#     print(f"  Total pairs: {total}")
#     print(f"  Matches: {matches}")
#     print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
#     print(f"  Unique true families: {clean_df['true_v_gene_family_call'].nunique()}")
#     print(f"  Unique generated families: {clean_df['gen_v_gene_family_call'].nunique()}")

# Uncomment to run quick analysis
# quick_analysis("updated_mapping.csv")