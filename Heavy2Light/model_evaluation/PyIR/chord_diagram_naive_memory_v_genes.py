import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import warnings



def load_and_clean_data(csv_file_path: str, btype_filter: str = None) -> pd.DataFrame:
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
    
    # Filter by BType if specified
    if btype_filter:
        if 'BType' not in df.columns:
            print(f"Warning: BType column not found in dataset!")
            return pd.DataFrame()
        
        df = df[df['BType'] == btype_filter]
        print(f"After filtering for {btype_filter}: {df.shape}")
    
    print(f"Rows with gen_v_gene_family_call: {df['gen_v_gene_family_call'].notna().sum()}")
    print(f"Rows with true_v_gene_family_call: {df['true_v_gene_family_call'].notna().sum()}")
    
    # Keep only rows where both gen and true V gene family calls are available
    clean_df = df.dropna(subset=['gen_v_gene_family_call', 'true_v_gene_family_call'])

    # Remove IGHV3 from analysis
    clean_df = clean_df[
        (clean_df['gen_v_gene_family_call'] != 'IGHV3') & 
        (clean_df['true_v_gene_family_call'] != 'IGHV3')
    ]
    
    print(f"Clean dataset shape (both calls available, IGHV3 removed): {clean_df.shape}")
    
    return clean_df

def create_chord_diagram_data(df: pd.DataFrame):
    """
    Prepare data for chord diagram from V gene family predictions.
    
    Args:
        df: DataFrame with true_v_gene_family_call and gen_v_gene_family_call columns
        
    Returns:
        Tuple of (matrix, labels, flow_data)
    """
    true_labels = df['true_v_gene_family_call']
    pred_labels = df['gen_v_gene_family_call']
    
    # Get all unique labels
    all_labels = sorted(list(set(true_labels.unique()) | set(pred_labels.unique())))
    n_labels = len(all_labels)
    
    # Create mapping from label to index
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    # Create flow matrix
    matrix = np.zeros((n_labels, n_labels))
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        matrix[true_idx][pred_idx] += 1
    
    # Create flow data for easier processing
    flow_data = []
    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            if matrix[i][j] > 0:
                flow_data.append({
                    'source': true_label,
                    'target': pred_label,
                    'value': matrix[i][j],
                    'source_idx': i,
                    'target_idx': j
                })
    
    return matrix, all_labels, flow_data

def draw_chord(ax, angle1, angle2, thickness, color1, color2, alpha=0.3):
    """
    Draw a chord (bezier curve) between two angles.
    """
    # Points on the circle
    x1 = np.cos(angle1)
    y1 = np.sin(angle1)
    x2 = np.cos(angle2)
    y2 = np.sin(angle2)
    
    # Control points for bezier curve
    control_factor = 0.3
    cx1 = control_factor * x1
    cy1 = control_factor * y1
    cx2 = control_factor * x2
    cy2 = control_factor * y2
    
    # Create bezier path
    verts = [(x1, y1), (cx1, cy1), (cx2, cy2), (x2, y2)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color1, 
                     linewidth=thickness, alpha=alpha)
    ax.add_patch(patch)

def create_chord_legend(ax, labels, colors, matrix):
    """
    Create a legend for the chord diagram.
    """
    # Calculate accuracy for each label
    accuracies = []
    for i, label in enumerate(labels):
        if matrix[i].sum() > 0:
            accuracy = matrix[i][i] / matrix[i].sum()
            accuracies.append(f"{label}: {accuracy:.2f}")
        else:
            accuracies.append(f"{label}: N/A")
    
    # Add legend box
    legend_text = "Per-class accuracy:\n" + "\n".join(accuracies)
    ax.text(-1.15, -0.8, legend_text, fontsize=9, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
           verticalalignment='top')

def draw_chord_diagram(matrix, labels, save_path=None, figsize=(12, 12), title_suffix=""):
    """
    Draw a chord diagram using matplotlib.
    
    Args:
        matrix: Flow matrix between categories
        labels: List of category labels
        save_path: Optional path to save the plot
        figsize: Figure size
        title_suffix: Additional text for title
    """
    n_labels = len(labels)
    
    # Calculate totals for each label
    row_sums = matrix.sum(axis=1)  # True label totals
    col_sums = matrix.sum(axis=0)  # Predicted label totals
    total_sum = matrix.sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors for each label
    colors = plt.cm.Set3(np.linspace(0, 1, n_labels))
    
    # Calculate angles for each segment
    angles = []
    start_angle = 0
    
    for i, label in enumerate(labels):
        # Size proportional to total occurrences (true + predicted)
        segment_size = (row_sums[i] + col_sums[i]) / (2 * total_sum) * 2 * np.pi
        angles.append((start_angle, start_angle + segment_size))
        start_angle += segment_size
    
    # Draw segments
    radius = 1.0
    segment_width = 0.1
    
    for i, (label, (start, end)) in enumerate(zip(labels, angles)):
        # Draw arc segment
        arc = Arc((0, 0), 2*radius, 2*radius, 
                 theta1=np.degrees(start), theta2=np.degrees(end),
                 linewidth=segment_width*100, color=colors[i], alpha=0.8)
        ax.add_patch(arc)
        
        # Add label
        mid_angle = (start + end) / 2
        label_radius = radius + 0.15
        x = label_radius * np.cos(mid_angle)
        y = label_radius * np.sin(mid_angle)
        
        # Adjust text rotation
        rotation = np.degrees(mid_angle)
        if mid_angle > np.pi/2 and mid_angle < 3*np.pi/2:
            rotation += 180
        
        ax.text(x, y, label, ha='center', va='center', 
               rotation=rotation, fontsize=10, fontweight='bold')
    
    # Draw flows (chords)
    for i in range(n_labels):
        for j in range(n_labels):
            if matrix[i][j] > 0:
                # Calculate chord thickness based on flow value
                thickness = matrix[i][j] / total_sum * 200  # Thick chords
                
                # Get angles for source and target
                source_start, source_end = angles[i]
                target_start, target_end = angles[j]
                
                source_mid = (source_start + source_end) / 2
                target_mid = (target_start + target_end) / 2
                
                # Draw bezier curve
                draw_chord(ax, source_mid, target_mid, thickness, 
                          colors[i], colors[j], alpha=0.3)
    
    # Add title
    accuracy = np.trace(matrix) / total_sum
    plt.title(f'V Gene Family Chord Diagram{title_suffix}\n'
             f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) | Total: {int(total_sum)}',
             fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    create_chord_legend(ax, labels, colors, matrix)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chord diagram saved to: {save_path}")
    
    plt.show()

def generate_detailed_analysis(df: pd.DataFrame):
    """
    Generate detailed analysis including classification report and match statistics.
    
    Args:
        df: DataFrame with V gene family calls
    """
    true_labels = df['true_v_gene_family_call']
    pred_labels = df['gen_v_gene_family_call']
    
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

def analyze_mismatches(df: pd.DataFrame):
    """
    Analyze and display the mismatched predictions.
    
    Args:
        df: DataFrame with V gene family calls
    """
    mismatches = df[df['true_v_gene_family_call'] != df['gen_v_gene_family_call']].copy()
    
    if len(mismatches) == 0:
        print("\nNo mismatches found - perfect accuracy!")
        return
    
    print(f"\n" + "="*60)
    print("MISMATCH ANALYSIS")
    print("="*60)
    
    print(f"\nTotal mismatches: {len(mismatches)}")
    
    # Most common mismatch patterns
    mismatch_patterns = mismatches.groupby(['true_v_gene_family_call', 'gen_v_gene_family_call']).size().reset_index(name='count')
    mismatch_patterns = mismatch_patterns.sort_values('count', ascending=False)
    
    print(f"\nMost common mismatch patterns:")
    for _, row in mismatch_patterns.head(10).iterrows():
        true_fam = row['true_v_gene_family_call']
        gen_fam = row['gen_v_gene_family_call']
        count = row['count']
        print(f"  {true_fam} → {gen_fam}: {count} times")
    
    # Show some example mismatches with additional context
    print(f"\nExample mismatches (showing first 10):")
    display_cols = ['base_fasta_id', 'true_v_gene_family_call', 'gen_v_gene_family_call', 
                   'true_v_gene_call', 'gen_v_gene_call']
    
    available_cols = [col for col in display_cols if col in mismatches.columns]
    if 'similarity' in mismatches.columns:
        available_cols.append('similarity')
    if 'BLOSUM_score' in mismatches.columns:
        available_cols.append('BLOSUM_score')
    
    print(mismatches[available_cols].head(10).to_string(index=False))

def create_summary_statistics_plot(df: pd.DataFrame, save_path: str = None):
    """
    Create a summary plot showing accuracy and distribution statistics.
    
    Args:
        df: DataFrame with V gene family calls
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall accuracy pie chart
    matches = (df['true_v_gene_family_call'] == df['gen_v_gene_family_call']).sum()
    mismatches = len(df) - matches
    
    ax1.pie([matches, mismatches], 
            labels=[f'Matches\n({matches})', f'Mismatches\n({mismatches})'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Overall V Gene Family Prediction Accuracy', fontweight='bold')
    
    # 2. True label distribution
    true_dist = df['true_v_gene_family_call'].value_counts()
    ax2.bar(range(len(true_dist)), true_dist.values, color='skyblue')
    ax2.set_xticks(range(len(true_dist)))
    ax2.set_xticklabels(true_dist.index, rotation=45, ha='right')
    ax2.set_title('True V Gene Family Distribution', fontweight='bold')
    ax2.set_ylabel('Count')
    
    # 3. Generated label distribution
    pred_dist = df['gen_v_gene_family_call'].value_counts()
    ax3.bar(range(len(pred_dist)), pred_dist.values, color='lightcoral')
    ax3.set_xticks(range(len(pred_dist)))
    ax3.set_xticklabels(pred_dist.index, rotation=45, ha='right')
    ax3.set_title('Generated V Gene Family Distribution', fontweight='bold')
    ax3.set_ylabel('Count')
    
    # 4. Per-class accuracy
    unique_labels = sorted(df['true_v_gene_family_call'].unique())
    accuracies = []
    
    for label in unique_labels:
        label_df = df[df['true_v_gene_family_call'] == label]
        if len(label_df) > 0:
            acc = (label_df['true_v_gene_family_call'] == label_df['gen_v_gene_family_call']).mean()
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

def main():
    """
    Main function to run the chord diagram analysis for both B cell types.
    """
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping_merged_dataset.csv"
    output_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs"
    
    try:
        print("="*80)
        print("V GENE FAMILY CHORD DIAGRAM ANALYSIS BY B CELL TYPE")
        print("="*80)
        
        # Analysis for Naive B Cells
        print("\n" + "="*50)
        print("NAIVE B CELLS ANALYSIS")
        print("="*50)
        
        naive_df = load_and_clean_data(csv_file_path, btype_filter="Naive-B-Cells")
        
        if len(naive_df) > 0:
            print(f"\nGenerating chord diagram for Naive B Cells...")
            matrix_naive, labels_naive, flow_data_naive = create_chord_diagram_data(naive_df)
            draw_chord_diagram(matrix_naive, labels_naive, 
                             save_path=f"{output_path}/naive_b_cells_chord_diagram.png",
                             title_suffix=" - Naive B Cells")
            
            print(f"\nDetailed analysis for Naive B Cells...")
            generate_detailed_analysis(naive_df)
            
            print(f"\nMismatch analysis for Naive B Cells...")
            analyze_mismatches(naive_df)
            
            print(f"\nSummary plot for Naive B Cells...")
            create_summary_statistics_plot(naive_df, f"{output_path}/naive_b_cells_summary.png")
            
            accuracy_naive = (naive_df['true_v_gene_family_call'] == naive_df['gen_v_gene_family_call']).mean()
        else:
            print("No data available for Naive B Cells analysis.")
            accuracy_naive = 0
        
        # Analysis for Memory B Cells
        print("\n" + "="*50)
        print("MEMORY B CELLS ANALYSIS")
        print("="*50)
        
        memory_df = load_and_clean_data(csv_file_path, btype_filter="Memory-B-Cells")
        
        if len(memory_df) > 0:
            print(f"\nGenerating chord diagram for Memory B Cells...")
            matrix_memory, labels_memory, flow_data_memory = create_chord_diagram_data(memory_df)
            draw_chord_diagram(matrix_memory, labels_memory, 
                             save_path=f"{output_path}/memory_b_cells_chord_diagram.png",
                             title_suffix=" - Memory B Cells")
            
            print(f"\nDetailed analysis for Memory B Cells...")
            generate_detailed_analysis(memory_df)
            
            print(f"\nMismatch analysis for Memory B Cells...")
            analyze_mismatches(memory_df)
            
            print(f"\nSummary plot for Memory B Cells...")
            create_summary_statistics_plot(memory_df, f"{output_path}/memory_b_cells_summary.png")
            
            accuracy_memory = (memory_df['true_v_gene_family_call'] == memory_df['gen_v_gene_family_call']).mean()
        else:
            print("No data available for Memory B Cells analysis.")
            accuracy_memory = 0
        
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
        
        print(f"\nChord diagram analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Additional function to run analysis for specific B cell type
def analyze_btype_chord(csv_file_path: str, btype: str, output_path: str = ""):
    """
    Run chord diagram analysis for a specific B cell type.
    
    Args:
        csv_file_path: Path to the CSV file
        btype: B cell type to analyze ("Naive-B-Cells" or "Memory-B-Cells")
        output_path: Output directory path
    """
    print(f"Analyzing {btype} with chord diagram...")
    
    df = load_and_clean_data(csv_file_path, btype_filter=btype)
    
    if len(df) == 0:
        print(f"No data available for {btype}")
        return
    
    # Generate chord diagram
    matrix, labels, flow_data = create_chord_diagram_data(df)
    filename = f"{btype.lower().replace('-', '_')}_chord_diagram.png"
    save_path = f"{output_path}/{filename}" if output_path else filename
    
    draw_chord_diagram(matrix, labels, save_path=save_path, title_suffix=f" - {btype}")
    
    # Detailed analysis
    generate_detailed_analysis(df)
    analyze_mismatches(df)
    
    summary_filename = f"{btype.lower().replace('-', '_')}_summary.png"
    summary_save_path = f"{output_path}/{summary_filename}" if output_path else summary_filename
    create_summary_statistics_plot(df, summary_save_path)
    
    accuracy = (df['true_v_gene_family_call'] == df['gen_v_gene_family_call']).mean()
    return df, accuracy



if __name__ == "__main__":
    main()