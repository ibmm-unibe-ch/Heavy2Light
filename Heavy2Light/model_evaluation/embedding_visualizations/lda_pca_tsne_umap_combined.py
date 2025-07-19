import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
import colorcet as cc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    BertTokenizer, 
    RobertaForMaskedLM,
    EncoderDecoderModel,
    BertModel
)
from adapters import init

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run dimensionality reduction on embeddings')
    
    # Required arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to input data file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['bert_unpaired_heavy', 'bert_unpaired_light', 'gpt2_unpaired_light', 'bert2gpt'], help='Type of model to use')
    
    # Input data configuration
    parser.add_argument('--target_column', type=str, required=True, help='Target column to use for coloring')
    parser.add_argument('--sequence_column', type=str, required=True, help='Column containing sequences to embed')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--plot_prefix', type=str, default='dim_reduction', help='Prefix for plot filenames')
    parser.add_argument('--plot_title', type=str, default=None, help='Plot title')
    
    # Adapter configuration (optional)
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to adapter')
    parser.add_argument('--adapter_name', type=str, default=None, help='Name of adapter')
    
    # Visualization options
    parser.add_argument('--title_fontsize', type=int, default=18, help='Font size for title and axis labels')
    parser.add_argument('--label_fontsize', type=int, default=16, help='Font size for tick labels and legend')
    parser.add_argument('--tick_fontsize', type=int, default=14, help='Font size for tick labels')
    parser.add_argument('--legend_title', type=str, default=None, help='Title for the legend')
    parser.add_argument('--plot_height', type=int, default=10, help='Height of the plot in inches')
    parser.add_argument('--plot_width', type=int, default=10, help='Width of the plot in inches')
    parser.add_argument('--dpi', type=int, default=450, help='DPI for saved figures')
    parser.add_argument('--transparent', action='store_true', help='Save plots with transparent background')
    parser.add_argument('--format', type=str, default='svg', choices=['svg', 'png', 'pdf'], help='Output file format')
    parser.add_argument('--marker_scale', type=float, default=1.0, help='Scale for size of dots in the legend')
    parser.add_argument('--new_label_name_1', type=str, default=None, help='New name for label 1 in legend')
    parser.add_argument('--new_label_name_2', type=str, default=None, help='New name for label 2 in legend')
    parser.add_argument('--x_min', type=float, default=None, help='Minimum x-axis value for plots')
    parser.add_argument('--x_max', type=float, default=None, help='Maximum x-axis value for plots')
    parser.add_argument('--y_min', type=float, default=None, help='Minimum y-axis value for plots')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum y-axis value for plots')
    parser.add_argument('--xaxis_bins', type=int, default=10, help='Number of bins for x-axis')
    parser.add_argument('--yaxis_bins', type=int, default=10, help='Number of bins for y-axis')

    
    # Dimensionality reduction parameters
    parser.add_argument('--methods', nargs='+', default=['pca', 'tsne', 'umap', 'lda'], 
                        choices=['pca', 'tsne', 'umap', 'lda'], help='Methods to use')
    parser.add_argument('--tsne_perplexity', type=int, default=30, help='Perplexity for t-SNE')
    parser.add_argument('--umap_neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help='Minimum distance for UMAP')
    parser.add_argument('--n_components', type=int, default=2, help='Number of components for dimensionality reduction')
    
    # Device configuration
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    # Model loading options
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')

    # filtering options
    parser.add_argument('--include_for_analysis', nargs='+', default=None, 
                        help='Values to include for LDA analysis (needs multiple classes)')
    parser.add_argument('--exclude_from_plots', nargs='+', default=None, 
                        help='Values to exclude from final plots (but keep for LDA computation)')
    parser.add_argument('--remove_wrong_heavy_values', action='store_true', 
                        help='Remove all values containing "H" in the target column')
    
    parser.add_argument('--legend_order', nargs='+', default=None, 
                        help='Custom order for legend labels (e.g., IGHV1 IGHV2 IGHV3 IGHV4 IGHV5 IGHV6 IGHV7)')
    
    return parser.parse_args()

# Initialize device
def initialize_device(device_arg):
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# Load model and tokenizer
def initialize_model_and_tokenizer(args, device):
    print(f"Loading model from {args.model_path}")
    
    if args.model_type == 'bert_unpaired_heavy' or args.model_type == 'bert_unpaired_light':
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_path)
    elif args.model_type == 'gpt2_unpaired_light':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path)
    elif args.model_type == 'bert2gpt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = EncoderDecoderModel.from_pretrained(args.model_path)
        if args.adapter_path and args.adapter_name:
            init(model)
            model.load_adapter(args.adapter_path)
            model.set_active_adapters(args.adapter_name)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model.to(device)
    print(f"Model loaded to {device}")
    return model, tokenizer

def apply_analysis_filter(df, target_column, include_values=None):
    """
    Filter dataframe for analysis (LDA computation) - first stage filtering
    """
    if include_values:
        print(f"Including for analysis: {include_values}")
        df = df[df[target_column].isin(include_values)]
        print(f"After analysis filter: {len(df)} samples with {len(df[target_column].unique())} unique classes")
        print(f"Classes for analysis: {df[target_column].unique()}")
    return df

def apply_heavy_filter(df, target_column, remove_heavy=False):
    """
    Filter out values containing 'H' if remove_heavy is True
    """
    if remove_heavy:
        original_count = len(df)
        df = df[~df[target_column].str.contains('H', na=False)]
        removed_count = original_count - len(df)
        print(f"Removed {removed_count} samples containing 'H' in {target_column}")
        print(f"After heavy filter: {len(df)} samples remaining")
        if len(df) > 0:
            print(f"Remaining classes: {df[target_column].unique()}")
    return df

def apply_plot_filter(result_df, target_column, exclude_values=None):
    """
    Filter results for plotting - second stage filtering
    """
    if exclude_values:
        print(f"Excluding from plots: {exclude_values}")
        original_count = len(result_df)
        result_df = result_df[~result_df[target_column].isin(exclude_values)]
        print(f"After plot filter: {len(result_df)} samples (removed {original_count - len(result_df)} samples)")
        print(f"Classes for plotting: {result_df[target_column].unique()}")
    return result_df


# Get embeddings from model
def get_embeddings(model, tokenizer, sequences, model_type, device, max_length=512):
    embeddings = []
    model.eval()
    
    if not sequences:
        raise ValueError("Input sequences are empty!")
    
    with torch.no_grad():
        for seq in sequences:
            if not isinstance(seq, str):
                print(f"Skipping non-string sequence: {seq}")
                continue
                
            inputs = tokenizer(
                seq, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            if inputs["input_ids"].size(1) == 0:
                print(f"Skipping sequence due to tokenization error: {seq}")
                continue
            
            # Get hidden states based on model type
            if model_type == 'bert_unpaired_heavy' or model_type == 'bert_unpaired_light':
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
            elif model_type == 'gpt2_unpaired_light':
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
            elif model_type == 'bert2gpt':
                # For encoder-decoder models, you can use either encoder or decoder
                # This example uses the encoder part
                outputs = model.encoder(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
            
            # Mean pooling to get sentence embedding
            mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled_output)
    
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were generated!")
    
    return np.vstack(embeddings)  # Stack into a numpy array

# Perform dimensionality reduction
def perform_dimensionality_reduction(embeddings, labels, methods, args):
    results = {}
    encoded_labels = None
    
    # Create a LabelEncoder for LDA
    if 'lda' in methods:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
    
    # Perform PCA
    if 'pca' in methods:
        print("Performing PCA...")
        pca = PCA(n_components=args.n_components)
        results['pca'] = pca.fit_transform(embeddings)
    
    # Perform t-SNE
    if 'tsne' in methods:
        print("Performing t-SNE...")
        tsne = TSNE(
            n_components=args.n_components, 
            random_state=42, 
            perplexity=args.tsne_perplexity
        )
        results['tsne'] = tsne.fit_transform(embeddings)
    
    # Perform UMAP
    if 'umap' in methods:
        print("Performing UMAP...")
        umap_reducer = umap.UMAP(
            n_components=args.n_components, 
            random_state=42,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist
        )
        results['umap'] = umap_reducer.fit_transform(embeddings)
    
    # Perform LDA
    if 'lda' in methods:
        print("Performing LDA...")
        # Get number of unique classes
        n_classes = len(np.unique(encoded_labels))
        # LDA's n_components must be at most n_classes - 1
        lda_n_components = min(args.n_components, n_classes - 1)
        if lda_n_components < 1:
            print("Warning: Cannot perform LDA with only one class")
        else:
            lda = LDA(n_components=lda_n_components)
            results['lda'] = lda.fit_transform(embeddings, encoded_labels)
    
    return results

def plot_results(results, labels, args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # set plot size
    plt.figure(figsize=(args.plot_height, args.plot_width))
    
    # Get a color palette with enough colors
    unique_labels = np.unique(labels)
    
    # Apply custom legend order if provided
    if args.legend_order:
        print(f"Applying custom legend order: {args.legend_order}")
        # Filter custom order to only include labels that exist in the data
        available_labels = set(unique_labels)
        ordered_labels = [label for label in args.legend_order if label in available_labels]
        
        # Add any remaining labels not specified in custom order (sorted)
        remaining_labels = sorted(available_labels - set(ordered_labels))
        unique_labels = ordered_labels + remaining_labels
        
        print(f"Final legend order: {unique_labels}")
    else:
        unique_labels = sorted(unique_labels)  # Default alphabetical sort

    n_colors = len(unique_labels)
    print(f"Using color palette with {n_colors} colors for {len(unique_labels)} unique labels")
    
    # Use colorcet palette which has many distinguishable colors
    palette = sns.color_palette(cc.glasbey, n_colors=n_colors)
    color_mapping = dict(zip(unique_labels, palette))
    
    # Set default legend title if not provided
    legend_title = args.legend_title if args.legend_title else args.target_column
    
    # Plot each result
    for method_name, result in results.items():
        print(f"Plotting {method_name.upper()} results...")
        
        # Create DataFrame for plotting
        result_df = pd.DataFrame(result, columns=[f"Component {i+1}" for i in range(result.shape[1])])
        result_df['Label'] = labels

        # Apply plot filtering (second stage filtering)
        if args.exclude_from_plots:
            result_df = apply_plot_filter(result_df, 'Label', args.exclude_from_plots)
            
            # If no data left after filtering, skip this method
            if len(result_df) == 0:
                print(f"No data left for plotting {method_name} after filtering. Skipping...")
                continue
        
        with sns.axes_style("white"):
            # Create JointGrid
            g = sns.JointGrid(
                data=result_df, 
                x="Component 1", 
                y="Component 2", 
                height=args.plot_height
            )
            
            # Plot main scatter WITH HUE_ORDER
            sns.scatterplot(
                data=result_df, 
                x="Component 1", 
                y="Component 2", 
                hue="Label", 
                hue_order=unique_labels,  # This enforces the legend order
                palette=color_mapping,    # Use consistent color mapping
                ax=g.ax_joint
            )
            
            # Plot marginal density plots WITH HUE_ORDER
            sns.kdeplot(
                data=result_df, 
                x="Component 1", 
                hue="Label", 
                hue_order=unique_labels,  # Consistent ordering
                palette=color_mapping,    # Consistent colors
                ax=g.ax_marg_x, 
                legend=False, 
                common_norm=True
            )
            
            sns.kdeplot(
                data=result_df, 
                y="Component 2", 
                hue="Label", 
                hue_order=unique_labels,  # Consistent ordering
                palette=color_mapping,    # Consistent colors
                ax=g.ax_marg_y, 
                legend=False, 
                common_norm=True
            )
            
        # Set axis titles with specified font size
        g.ax_joint.set_xlabel("Component 1", fontsize=args.title_fontsize)
        g.ax_joint.set_ylabel("Component 2", fontsize=args.title_fontsize)

        # Add these lines to control tick label font size
        g.ax_joint.tick_params(axis='x', labelsize=args.tick_fontsize) 
        g.ax_joint.tick_params(axis='y', labelsize=args.tick_fontsize)

        g.ax_joint.set_xlim(args.x_min, args.x_max)
        g.ax_joint.set_ylim(args.y_min, args.y_max)

        from matplotlib.ticker import MaxNLocator
        g.ax_joint.xaxis.set_major_locator(MaxNLocator(nbins=args.xaxis_bins))
        g.ax_joint.yaxis.set_major_locator(MaxNLocator(nbins=args.yaxis_bins))
        
        # Handle legend
        handles, labels_from_plot = g.ax_joint.get_legend_handles_labels()

        # Remove the automatically added legend
        if g.ax_joint.get_legend():
            g.ax_joint.get_legend().remove()
        
        # Create a new legend outside the plot
        legend_config = g.fig.legend(
            handles=handles, 
            labels=labels_from_plot,  # These will now be in the correct order
            title=legend_title, 
            title_fontsize=args.title_fontsize, 
            fontsize=args.label_fontsize,
            loc='center left',  
            bbox_to_anchor=(1.02, 0.25),  # Right of plot, centered vertically
            ncol=1,  
            frameon=False, 
            borderaxespad=0.,
            markerscale=args.marker_scale
        )

        if args.new_label_name_1 is not None and len(legend_config.get_texts()) > 0:
            legend_config.get_texts()[0].set_text(args.new_label_name_1)
        if args.new_label_name_2 is not None and len(legend_config.get_texts()) > 1:
            legend_config.get_texts()[1].set_text(args.new_label_name_2)
        
        # Adjust the layout to make room for the legend
        plt.subplots_adjust(bottom=0.15)
        
        # Set plot title if provided
        if args.plot_title:
            g.fig.suptitle(f"{args.plot_title} - {method_name.upper()}", fontsize=args.title_fontsize)
        
        # Save the plot
        output_file = os.path.join(
            args.output_dir, 
            f"{args.plot_prefix}_{method_name}.{args.format}"
        )
        
        g.savefig(
            output_file, 
            format=args.format, 
            dpi=args.dpi, 
            transparent=args.transparent
        )
        
        print(f"Saved plot to {output_file}")
        plt.close()

# Main function
def main():
    args = parse_args()
    
    # Initialize device
    device = initialize_device(args.device)
    
    # Load model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args, device)
    
    # Load data
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)

    # Apply heavy filter if requested
    if args.remove_wrong_heavy_values:
        df = apply_heavy_filter(df, args.target_column, args.remove_wrong_heavy_values)
        
        # Check if we still have data after filtering
        if len(df) == 0:
            raise ValueError("No data remaining after heavy filtering!")


    # Apply first stage filtering (for analysis including LDA)
    if args.include_for_analysis:
        df = apply_analysis_filter(df, args.target_column, args.include_for_analysis)
        
        # Check if we still have data after filtering
        if len(df) == 0:
            raise ValueError("No data remaining after analysis filtering!")
    
    
    # Extract sequences and labels
    sequences = df[args.sequence_column].tolist()
    labels = df[args.target_column].tolist()
    
    print(f"Loaded {len(sequences)} sequences with {len(set(labels))} unique labels")
    
    # Get embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(
        model, 
        tokenizer, 
        sequences, 
        args.model_type, 
        device, 
        args.max_length
    )
    
    print(f"Generated embeddings with shape {embeddings.shape}")
    
    # Perform dimensionality reduction
    results = perform_dimensionality_reduction(
        embeddings, 
        labels, 
        args.methods, 
        args
    )
    
    # Plot results
    plot_results(results, labels, args)
    
    print("All operations completed successfully!")

if __name__ == "__main__":
    main()