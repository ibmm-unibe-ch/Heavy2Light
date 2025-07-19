import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
import seaborn as sns
from collections import Counter


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

def draw_chord_diagram(matrix, labels, save_path=None, figsize=(12, 12)):
    """
    Draw a chord diagram using matplotlib.
    
    Args:
        matrix: Flow matrix between categories
        labels: List of category labels
        save_path: Optional path to save the plot
        figsize: Figure size
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
                thickness = matrix[i][j] / total_sum * 200  # Scale factor
                
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
    plt.title(f'V Gene Family Prediction Flow Diagram\n'
             f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) | Total: {int(total_sum)}',
             fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    create_chord_legend(ax, labels, colors, matrix)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chord diagram saved to: {save_path}")
    
    plt.show()

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
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
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
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.9),
           verticalalignment='top')

def main():
    """
    Main function to create chord diagrams for V gene family predictions.
    """
    csv_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/confusion_matrix_outputs/merged_pairing_vgene_pairing_result_bert2gpt_full_complete_ids_mapping_unique_nt_trimmed_gene_hit_locus.csv"  # Your data file
    output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/unconditioned_gen_seqs/PyIR/chord_diagrams"               
    
    # Filter settings
    min_pairing_score = 0.5
    
    try:
        print("="*60)
        print("V GENE FAMILY CHORD DIAGRAM GENERATION")
        print("="*60)
        
        # Load and filter data
        df = pd.read_csv(csv_file_path)
        
        # Apply pairing score filter if column exists
        if 'pairing_scores' in df.columns:
            print(f"Filtering for pairing scores >= {min_pairing_score}")
            df = df[df['pairing_scores'] >= min_pairing_score]
        
        # Clean data
        df_clean = df.dropna(subset=['gen_v_gene_family_call', 'true_v_gene_family_call'])
        
        # Remove IGHV3 if desired
        df_clean = df_clean[
            (df_clean['gen_v_gene_family_call'] != 'IGHV3') & 
            (df_clean['true_v_gene_family_call'] != 'IGHV3')
        ]
        
        print(f"Final dataset shape: {df_clean.shape}")
        
        if len(df_clean) == 0:
            print("No data available for chord diagram generation.")
            return
        
        # Prepare chord diagram data
        print("Preparing chord diagram data...")
        matrix, labels, flow_data = create_chord_diagram_data(df_clean)
        
        print(f"Number of V gene families: {len(labels)}")
        print(f"V gene families: {', '.join(labels)}")
        
        # Create matplotlib chord diagram
        print("Creating matplotlib chord diagram...")
        draw_chord_diagram(matrix, labels, f"{output_dir}/vgene_chord_diagram_{min_pairing_score}.png")

        # Print summary statistics
        accuracy = np.trace(matrix) / matrix.sum()
        print(f"\nChord diagram generation completed!")
        print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Total predictions: {int(matrix.sum())}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()