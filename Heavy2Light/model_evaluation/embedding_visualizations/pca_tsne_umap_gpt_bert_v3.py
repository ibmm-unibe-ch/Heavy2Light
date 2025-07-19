from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig, AutoModel
import torch
import pandas as pd
from adapters import init
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import colorcet as cc
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"embedding_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('embedding_analysis')

logger.info("Starting script execution")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def initialize_model_and_tokenizer(model_path, device, model_type="auto"):
    """
    Initialize model and tokenizer with better error handling
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        model_type: Type of model to load ("auto", "encoder_decoder", or "bert")
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Initializing {model_type} model from {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if model_type == "encoder_decoder":
            model = EncoderDecoderModel.from_pretrained(model_path)
            logger.info("Loaded EncoderDecoderModel")
        elif model_type == "bert":
            model = AutoModel.from_pretrained(model_path)
            logger.info("Loaded BERT-like model")
        else:  # Default to AutoModel
            model = AutoModel.from_pretrained(model_path)
            logger.info("Loaded AutoModel")
            
        model.to(device)
        logger.info(f"Model loaded to {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

# # Model configuration gpt-2 epoch 41
# run_name = "gpt_unpaired_epoch_41"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-6058816"
# model_name = "gpt2_unpaired_light"

# # gpt light unpaired epoch 80
# run_name="gpt_unpaired_epoch_80"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-11822080"
# model_name = "gpt2_unpaired_light"

# # BERT light model small
# run_name="bert_unpaired_light_small"
# model_path =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

# BERT heavy model small
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
run_name="bert_unpaired_heavy_small"


#model_name = "bert2gpt"
#model_name="gpt2_unpaired_light"
#model_name="bert_model_light"
model_name="bert_unpaired_heavy"


# Initialize model and tokenizer
logger.info(f"Loading model: {model_name} from path: {model_path}")
model, tokenizer = initialize_model_and_tokenizer(model_path, device)

# Load data
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set_1000.csv"
logger.info(f"Loading data from: {test_file_path}")

try:
    test_df = pd.read_csv(test_file_path)
    logger.info(f"Successfully loaded dataset with {len(test_df)} rows")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

#light_sequences = test_df["sequence_alignment_aa_light"].tolist()
heavy_sequences = test_df["sequence_alignment_aa_heavy"].tolist()

#labels = test_df['v_gene_light_family'].tolist()
labels = test_df['v_gene_heavy_family'].tolist()


logger.info(f"Extracted {len(heavy_sequences)} heavy_sequences sequences and labels")
logger.info(f"Unique labels: {set(labels)}")

def get_last_layer_embeddings(model, tokenizer, sequences, device, model_name):
    """Extract embeddings from the model with improved error handling and logging"""
    embeddings = []
    processed_count = 0
    skipped_count = 0
    total_sequences = len(sequences)
    start_time = time.time()
    
    logger.info(f"Starting embedding extraction for {total_sequences} sequences")
    
    model.to(device)
    model.eval()
    
    if not sequences:
        logger.error("Input sequences are empty!")
        raise ValueError("Input sequences are empty!")

    with torch.no_grad():
        for idx, seq in enumerate(sequences):
            # Log progress at intervals
            if idx > 0 and idx % 100 == 0:
                elapsed = time.time() - start_time
                time_per_seq = elapsed / idx
                remaining = time_per_seq * (total_sequences - idx)
                logger.info(f"Processed {idx}/{total_sequences} sequences - ETA: {remaining:.1f}s")
            
            try:
                inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                
                if inputs["input_ids"].size(1) == 0:
                    logger.warning(f"Skipping sequence due to tokenization error: {seq[:50]}...")
                    skipped_count += 1
                    continue
                
                # Forward pass through the model
                if model_name == "gpt2_unpaired_light":
                    outputs = model(**inputs, output_hidden_states=True)
                elif model_name == "bert2gpt":
                    outputs = model.decoder(**inputs, output_hidden_states=True)
                elif model_name == "bert_model_light":
                    outputs = model(**inputs, output_hidden_states=True)
                elif model_name == "bert_unpaired_heavy":
                    outputs = model(**inputs, output_hidden_states=True)
                else:
                    logger.error(f"Invalid model name: {model_name}")
                    raise ValueError(f"Invalid model name: {model_name}")
                
                last_hidden_states = outputs.hidden_states[-1]

                if last_hidden_states is None or last_hidden_states.size(0) == 0:
                    logger.warning(f"Skipping sequence due to model output error: {seq[:50]}...")
                    skipped_count += 1
                    continue
                
                # Apply mean pooling on the hidden states
                embedding = last_hidden_states.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing sequence {idx}: {str(e)}")
                skipped_count += 1
                continue
    
    total_time = time.time() - start_time
    logger.info(f"Embedding extraction complete. Processed: {processed_count}, Skipped: {skipped_count}, Time: {total_time:.2f}s")
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings were generated!")
        raise ValueError("No valid embeddings were generated!")
    
    # Stack embeddings into a numpy array
    stacked_embeddings = np.vstack(embeddings)
    logger.info(f"Generated embeddings with shape: {stacked_embeddings.shape}")
    
    return stacked_embeddings

# Create output directory
plot_save_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/{model_name}/{run_name}"
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)
    logger.info(f"Created output directory: {plot_save_dir}")

# Get embeddings
logger.info("Extracting embeddings")
try:
    embeddings = get_last_layer_embeddings(model, tokenizer, heavy_sequences, device, model_name)
    logger.info(f"Successfully extracted embeddings with shape {embeddings.shape}")
except Exception as e:
    logger.error(f"Error extracting embeddings: {str(e)}")
    raise

def plot_histogram_line(ax, data, color, bins=20, vertical=False):
    """
    Plot a line representation of a histogram without bars
    
    Args:
        ax: Matplotlib axis to plot on
        data: Data to plot
        color: Color for the line
        bins: Number of bins for the histogram
        vertical: Whether to plot vertically
    """
    # Calculate histogram data
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist / hist.max()
    
    if vertical:
        # Plot vertical line histogram
        ax.plot(hist, bin_centers, color=color, linewidth=1.5)
    else:
        # Plot horizontal line histogram
        ax.plot(bin_centers, hist, color=color, linewidth=1.5)




def plot_dimensionality_reduction(result, labels, method_name, plot_save_prefix, model_name, show_marginals=True):
    """
    Plot dimensionality reduction results with histograms as marginals
    
    Args:
        result: 2D array of dimensionality reduction results
        labels: Labels for coloring the points
        method_name: Name of the dimensionality reduction method
        plot_save_prefix: Prefix for saving the plot
        model_name: Name of the model used
        show_marginals: Whether to show marginal distributions
    """
    logger.info(f"Creating {method_name} plot with marginals={show_marginals}")
    
    # Get a unique set of labels for coloring
    unique_labels = list(set(labels))
    color_count = len(unique_labels)
    logger.info(f"Plotting with {color_count} unique labels")
    
    # Use a colorblind-friendly palette with enough colors
    if color_count <= 10:
        palette = sns.color_palette("colorblind", n_colors=color_count)
    else:
        palette = sns.color_palette(cc.glasbey, n_colors=color_count)
    
    # Create a color dictionary for consistent colors
    color_dict = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    # Set seaborn style without grid
    with sns.axes_style("white"):  # Changed from "whitegrid" to "white" to remove grid
        # Set larger font sizes for all text elements
        sns.set(font_scale=1.5)  # Increased font scale
        
        # Create a DataFrame for the plot
        result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
        result_df['Label'] = labels
        
        # Create a JointGrid for the plot
        g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)
        
        # Plot the scatterplot in the main axes
        scatter = sns.scatterplot(
            data=result_df, 
            x="Component 1", 
            y="Component 2", 
            hue="Label", 
            palette=palette, 
            s=30, 
            alpha=0.7,
            edgecolor='white',
            ax=g.ax_joint
        )
        
        # Increase font size for axis labels
        g.ax_joint.set_xlabel("Component 1", fontsize=18)  # Increased to 18
        g.ax_joint.set_ylabel("Component 2", fontsize=18)  # Increased to 18
        
        # Remove axis ticks and numbers from main plot
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        g.ax_joint.set_xticklabels([])
        g.ax_joint.set_yticklabels([])
        
        # Add a title with larger font
        g.ax_joint.set_title(f"{method_name} Visualization", fontsize=18, fontweight='bold')
        
        # Plot histograms on the marginals if requested
        if show_marginals:
            # Clear the marginal axes first
            g.ax_marg_x.clear()
            g.ax_marg_y.clear()
            
            # Define histogram bins (same for all histograms for consistency)
            n_bins = 20
            x_min, x_max = result_df["Component 1"].min(), result_df["Component 1"].max()
            y_min, y_max = result_df["Component 2"].min(), result_df["Component 2"].max()
            x_bins = np.linspace(x_min, x_max, n_bins + 1)
            y_bins = np.linspace(y_min, y_max, n_bins + 1)
            
            # Plot separate histograms for each label with no fill
            for label in unique_labels:
                mask = result_df['Label'] == label
                # Get the exact same color as used in the scatter plot
                color = color_dict[label]
                
                # Plot histograms with no fill for x-axis marginal
                sns.histplot(
                    data=result_df.loc[mask],
                    x="Component 1",
                    ax=g.ax_marg_x,
                    bins=x_bins,
                    color=color,
                    element="step",    
                    fill=False,        
                    linewidth=1.5,     
                    legend=False
                )
                
                # Plot histograms with no fill for y-axis marginal
                sns.histplot(
                    data=result_df.loc[mask],
                    y="Component 2",
                    ax=g.ax_marg_y,
                    bins=y_bins,
                    color=color,
                    element="step",    
                    fill=False,        
                    linewidth=1.5,     
                    legend=False
                )
            
            # Remove axis labels from marginal plots
            g.ax_marg_x.set_xlabel("")
            g.ax_marg_x.set_ylabel("")
            g.ax_marg_y.set_xlabel("")
            g.ax_marg_y.set_ylabel("")
            
            # Remove axis ticks and numbers from marginal plots
            g.ax_marg_x.set_xticks([])
            g.ax_marg_x.set_yticks([])
            g.ax_marg_y.set_xticks([])
            g.ax_marg_y.set_yticks([])
        
        # Remove the legend from the scatterplot
        if g.ax_joint.get_legend():
            g.ax_joint.get_legend().remove()
        
        # Create a legend for the figure with larger font
        handles, labels = g.ax_joint.get_legend_handles_labels()
        g.fig.legend(
            handles=handles, 
            labels=labels, 
            title='V Gene Family', 
            title_fontsize=18,  # Set to 18 as requested
            fontsize=16,  # Increased to 16 as requested
            markerscale=2.0,
            loc='center right', 
            bbox_to_anchor=(1.15, 0.5), 
            frameon=False,
            handlelength=2.5,
            handleheight=1.5,
            borderpad=1.0,
            labelspacing=1.2
        )
        
        # Adjust the layout to make room for the legend
        plt.subplots_adjust(right=0.8)
        
        # Save the figure as SVG
        output_file = f'{plot_save_dir}/{plot_save_prefix}_{method_name}{"_with_marginals" if show_marginals else ""}.svg'
        plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=300)
        logger.info(f"Saved SVG plot to {output_file}")
        
        # Also save as PNG for backward compatibility
        output_file_png = f'{plot_save_dir}/{plot_save_prefix}_{method_name}{"_with_marginals" if show_marginals else ""}.png'
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PNG plot to {output_file_png}")
        
        # Close the figure to save memory
        plt.close()

# Processing pipeline for dimensionality reduction
plot_save_prefix = "v3_small_test_gpt_2"

# Perform PCA
logger.info("Starting PCA")
try:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot PCA results with and without marginals
    for show_marginals in [True, False]:
        plot_dimensionality_reduction(
            result=pca_result,
            labels=labels,
            method_name='PCA',
            plot_save_prefix=plot_save_prefix,
            model_name=model_name,
            show_marginals=show_marginals
        )
except Exception as e:
    logger.error(f"Error in PCA: {str(e)}")

# Perform t-SNE
logger.info("Starting t-SNE")
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    logger.info("t-SNE initialized, beginning fit_transform (this may take some time)")
    start_time = time.time()
    tsne_result = tsne.fit_transform(embeddings)
    logger.info(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    
    # Plot t-SNE results with and without marginals
    for show_marginals in [True, False]:
        plot_dimensionality_reduction(
            result=tsne_result,
            labels=labels,
            method_name='t-SNE',
            plot_save_prefix=plot_save_prefix,
            model_name=model_name,
            show_marginals=show_marginals
        )
except Exception as e:
    logger.error(f"Error in t-SNE: {str(e)}")

# Perform UMAP
logger.info("Starting UMAP")
try:
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    logger.info("UMAP initialized, beginning fit_transform")
    start_time = time.time()
    umap_result = umap_reducer.fit_transform(embeddings)
    logger.info(f"UMAP completed in {time.time() - start_time:.2f} seconds")
    
    # Plot UMAP results with and without marginals
    for show_marginals in [True, False]:
        plot_dimensionality_reduction(
            result=umap_result,
            labels=labels,
            method_name='UMAP',
            plot_save_prefix=plot_save_prefix,
            model_name=model_name,
            show_marginals=show_marginals
        )
except Exception as e:
    logger.error(f"Error in UMAP: {str(e)}")

logger.info("Script execution completed")