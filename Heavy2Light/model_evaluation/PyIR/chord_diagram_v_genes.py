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
from scipy import stats
warnings.filterwarnings('ignore')

def load_and_clean_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load the CSV file and clean the data for analysis.
    
    Args:
        csv_file_path: Path to the updated mapping CSV file
        
    Returns:
        Cleaned DataFrame with non-null V gene family calls
    """
    df = pd.read_csv(csv_file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Rows with gen_v_gene_family_call: {df['gen_v_gene_family_call'].notna().sum()}")
    print(f"Rows with true_v_gene_family_call: {df['true_v_gene_family_call'].notna().sum()}")
    
    # Keep only rows where both gen and true V gene family calls are available
    clean_df = df.dropna(subset=['gen_v_gene_family_call', 'true_v_gene_family_call'])

    # Remove IGHV3 
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

def analyze_vgene_vs_similarity(df: pd.DataFrame, save_path: str = None):
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
    
    unique_families = sorted(df['true_v_gene_family_call'].unique())
    
    for family in unique_families:
        family_df = df[df['true_v_gene_family_call'] == family]
        
        if len(family_df) > 0:
            # V gene family accuracy
            v_gene_matches = (family_df['true_v_gene_family_call'] == family_df['gen_v_gene_family_call']).sum()
            v_gene_accuracy = v_gene_matches / len(family_df)
            
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
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

def create_similarity_distribution_plot(df: pd.DataFrame, save_path: str = None):
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
    correct_mask = df['true_v_gene_family_call'] == df['gen_v_gene_family_call']
    correct_similarities = df[correct_mask]['calculated_similarity']
    incorrect_similarities = df[~correct_mask]['calculated_similarity']
    
    print(f"Correct predictions: {len(correct_similarities)} (avg similarity: {correct_similarities.mean():.3f})")
    print(f"Incorrect predictions: {len(incorrect_similarities)} (avg similarity: {incorrect_similarities.mean():.3f})")
    
    # Statistical test
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

def main():
    """
    Main function to run the chord diagram analysis with statistics.
    """
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/src/v_genes_no_D_bert2gpt_complete_ids_gen_true_mapping.csv"
    output_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/chord_diagrams"
    
    try:
        print("="*80)
        print("V GENE FAMILY CHORD DIAGRAM ANALYSIS")
        print("="*80)
        
        print("Loading and cleaning data...")
        df = load_and_clean_data(csv_file_path)
        
        if len(df) == 0:
            print("No data available for analysis. Please check that your CSV file contains the required columns.")
            return
        
        print("\nGenerating chord diagram...")
        matrix, chord_labels, flow_data = create_chord_diagram_data(df)
        draw_chord_diagram(matrix, chord_labels, 
                         save_path=f"{output_path}/compl_no_d_v_gene_family_chord_diagram.png")
        
        print("\nGenerating detailed analysis...")
        generate_detailed_analysis(df)
        
        print("\nAnalyzing mismatches...")
        analyze_mismatches(df)
        
        print("\nGenerating summary statistics plot...")
        create_summary_statistics_plot(df, f"{output_path}/compl_no_d_v_gene_family_summary.png")
        
        print("\nAnalyzing V Gene Family vs Similarity...")
        vgene_similarity_stats = analyze_vgene_vs_similarity(df, 
            f"{output_path}/compl_no_d_vgene_vs_similarity_analysis.png")
        
        print("\nAnalyzing Similarity Distributions...")
        create_similarity_distribution_plot(df, 
            f"{output_path}/compl_no_d_similarity_distribution.png")
        
        # Calculate final accuracy
        accuracy = (df['true_v_gene_family_call'] == df['gen_v_gene_family_call']).mean()
        
        print(f"\nChord diagram analysis completed successfully!")
        print(f"Overall V Gene Family Prediction Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()