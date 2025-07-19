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



# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Initialize the model, tokenizer, and generation configuration
def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    return model, tokenizer, generation_config

# # Define paths
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# tokenizer_path = f"{model_path}/checkpoint-336040"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# adapter_name = "heavy2light_adapter"

# run_name="full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/full_PLAbDab_healthy_human_nucleus_bert2gpt_nucleus_healthy_human_PLAbDab_max_new_tokens_115_num_epochs_50_bert_like_tokenizer_unpaired_epo_41-7"
# tokenizer_path=f"{model_path}/checkpoint-367750"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

# gpt light unpaired epoch 41
run_name="gpt_unpaired_epoch_41"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-6058816"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # gpt light unpaired epoch 80
# run_name="gpt_unpaired_epoch_80"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-11822080"
# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # BERT light model small
# run_name="bert_unpaired_light_small"
# small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"
# model_name = "light_model_bert"
# tokenizer = AutoTokenizer.from_pretrained(small_light_decoder)
# model = AutoModel.from_pretrained(small_light_decoder)

# # BERT heavy model small
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
# run_name="bert_unpaired_heavy_small"
# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize model and tokenizer
#model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

#model_name = "bert2gpt"
model_name="gpt2_unpaired_light"
#model_name="bert_model_light"
#model_name="bert_unpaired_heavy"


# Load small test data
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# load small test data with locus (kappa or lambda)
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/small_data_heavyseplight_locus_no_dupl_spaces_rm.csv"

# load FULL test data
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated.txt"

# load FULL test data with locus (kappa or lambda)
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus.csv"

# sequence_heavy,locus_heavy,v_call_heavy,sequence_alignment_heavy,sequence_alignment_aa_heavy,germline_alignment_aa_heavy,cdr3_aa_heavy,sequence_light,
# locus_light,v_call_light,sequence_alignment_light,sequence_alignment_aa_light,germline_alignment_aa_light,cdr3_aa_light,sequence_alignment_heavy_sep_light,
# BType,Disease,Species,Subject,Author,Age,sequence_alignment_aa_light_1,generated_sequence_light,input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,
# calculated_similarity,general_v_gene_heavy,general_v_gene_light,v_gene_heavy_family,v_gene_light_family,alignment_germline,group
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set.csv"
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set_small_100.csv"

# sequence_heavy,locus_heavy,v_call_heavy_x,sequence_alignment_heavy,sequence_alignment_aa_heavy_x,germline_alignment_aa_heavy,cdr3_aa_heavy,sequence_light,locus_light_x,
# v_call_light_x,sequence_alignment_light,sequence_alignment_aa_light_x,germline_alignment_aa_light,cdr3_aa_light,sequence_alignment_heavy_sep_light,BType_x,Disease,Species,
# Subject,Author,Age,sequence_alignment_aa_light_1,generated_sequence_light,input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,calculated_similarity,
# general_v_gene_heavy,general_v_gene_light,v_gene_heavy_family_x,v_gene_light_family_x,alignment_germline,group,sequence_alignment_aa_heavy_y,sequence_alignment_aa_light_y,
# BType_y,locus_light_y,v_call_heavy_y,v_call_light_y,j_call_heavy,j_call_light,v_gene_heavy_family_y,v_gene_light_family_y,j_gene_heavy_family,j_gene_light_family
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/data/bert2gpt_df_merged_final_test_set_with_full_paired.csv"

# paired and unpaired artificial dataset
#sequence_heavy,locus_heavy,v_call_heavy,sequence_alignment_heavy,sequence_alignment_aa_heavy,germline_alignment_aa_heavy,
#cdr3_aa_heavy,sequence_light,locus_light,v_call_light,sequence_alignment_light,sequence_alignment_aa_light,germline_alignment_aa_light,
#cdr3_aa_light,sequence_alignment_heavy_sep_light,BType,Disease,Species,Subject,Author,Age,sequence_alignment_aa_light_1,generated_sequence_light,
#input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,calculated_similarity,general_v_gene_heavy,general_v_gene_light,v_gene_heavy_family,
#v_gene_light_family,alignment_germline,group,final_sequence_alignment_aa_light,paired_digit,full_sequence_heavy_light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/src/paired_artif_unpaired_dataset_bert2gpt_df_merged_final_test_set.csv"


# load test file as csv
test_df = pd.read_csv(test_file_path)


# def load_data(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             data.append(line.strip())
#     sequences = []
#     for entry in data:
#         split_entry = entry.split('[SEP]')
#         if len(split_entry) == 2:
#             sequences.append(split_entry)
#         else:
#             print(f"Skipping invalid entry: {entry}")
#     df = pd.DataFrame(sequences, columns=['heavy', 'light'])
#     return df

# test_df = load_data(test_file_path)
#heavy_sequences = test_df["input_heavy_sequence"].tolist()
light_sequences = test_df["sequence_alignment_aa_light_1"].tolist()

#full_sequences = test_df["full_sequence_heavy_light"].tolist()
#labels = test_df['locus_light'].tolist()
#labels = test_df['v_call_light'].tolist()
#labels = test_df['paired_digit'].tolist()
labels = test_df['j_gene_heavy_family'].tolist()



# # Function to extract embeddings from the last layer
# def get_last_layer_embeddings(model, tokenizer, sequences, device):
#     embeddings = []
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for seq in sequences:
#             inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
#             outputs = model.encoder(**inputs)
#             last_hidden_states = outputs.last_hidden_state
#             embeddings.append(last_hidden_states.mean(dim=1).cpu().numpy())  # Mean pooling
#     return np.vstack(embeddings)  # Stack into a numpy array

def get_last_layer_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    model.to(device)
    model.eval()
    if not sequences:
        raise ValueError("Input sequences are empty!")

    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            if inputs["input_ids"].size(1) == 0:
                print(f"Skipping sequence due to tokenization error: {seq}")
                continue
            # Forward pass through the decoder (GPT model)
            if model_name == "gpt2_unpaired_light":
                outputs = model(**inputs, output_hidden_states=True)
            elif model_name == "bert2gpt":
                outputs = model.decoder(**inputs, output_hidden_states=True)  # Adjust for your specific model implementation
            elif model_name == "bert_model_light":
                outputs = model(**inputs, output_hidden_states=True)
            elif model_name == "bert_unpaired_heavy":
                outputs = model(**inputs, output_hidden_states=True)
            else:
                raise ValueError("Invalid model name!")
             
            last_hidden_states = outputs.hidden_states[-1]

            if last_hidden_states is None or last_hidden_states.size(0) == 0:
                print(f"Skipping sequence due to model output error: {seq}")
                continue
            embeddings.append(last_hidden_states.mean(dim=1).cpu().numpy())  # Mean pooling

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were generated!")
    
    return np.vstack(embeddings)  # Stack into a numpy array

# Get embeddings from the last layer
#embeddings = get_last_layer_embeddings(model, tokenizer, heavy_sequences, device)

# for gpt upaired light use light seqs as input
embeddings = get_last_layer_embeddings(model, tokenizer, light_sequences, device)

#embeddings = get_last_layer_embeddings(model, tokenizer, full_sequences, device)

# # Apply UMAP
# umap_reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = umap_reducer.fit_transform(embeddings)

# # Plot UMAP result with labels
# plt.figure(figsize=(8, 6))
# for label in set(labels):
#     indices = [i for i, l in enumerate(labels) if l == label]
#     plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=label, alpha=0.5, s=5)
# plt.title(f'UMAP of Last Layer Embeddings {model_type}')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.legend()
# plt.show()
# plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/{model_type}_{run_name}.png')

plot_save_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/{model_name}/{run_name}"

# Create the directory if it does not exist
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)


# def plot_dimensionality_reduction(result, labels, method_name, plot_title_target=None, plot_save_prefix=None, model_name=None):
#     # Convert result to a DataFrame
#     result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
#     result_df['Label'] = labels  # Add labels for coloring

#     # Create the plot using Seaborn
#     g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)

#     palette="Paired"
    
#     # Plot scatter plot with hue (based on the label column)
#     g = g.plot(sns.scatterplot, sns.histplot, data=result_df, hue="Label", palette=palette, legend=False)

#     # legend outside of plot
#     handles, labels = g.ax_joint.get_legend_handles_labels()
#     g.figure.legend(handles=handles, labels=labels, title='B-Cell Type', loc='upper right', bbox_to_anchor=(1.05, 1), frameon=False, borderaxespad=0.)

#     plt.subplots_adjust(top=0.85, right=0.85) 

#     # Manually add a single legend, renaming it to 'J Gene Family'
#     # handles, labels = g.ax_joint.get_legend_handles_labels()
#     # g.ax_joint.legend(handles=handles, labels=labels, title='B-Cell Type', loc='lower right', frameon=False)

#     # handles, labels = g.ax_joint.get_legend_handles_labels()
#     # g.ax_joint.legend(handles=handles, labels=labels, title="V Gene Light", bbox_to_anchor=(1.05, 1), loc="upper left")

#     # Customize the plot title
#     #g.figure.suptitle(f'Differentiation of {plot_title_target} via {method_name} of Last Layer Embeddings', y=1.02)

#     # Save the plot
#     g.savefig(f'{plot_save_dir}/{plot_save_prefix}_pal_{palette}_{method_name}.png')

#     # Show the plot
#     plt.show()


# def plot_dimensionality_reduction(result, labels, method_name, plot_title_target=None, plot_save_prefix=None, model_name=None):
#     # Convert result to a DataFrame

#     palette = sns.color_palette(cc.glasbey, n_colors=25)

#     with sns.axes_style("whitegrid"):
#         result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
#         result_df['Label'] = labels  # Add labels for coloring

#         # Create a JointGrid with proper dimensions
#         g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)
        
#         # Plot the joint and marginal distributions
#         sns.scatterplot(data=result_df, x="Component 1", y="Component 2", 
#                     hue="Label", palette=palette, ax=g.ax_joint)
        
#         # Increase font size for axis labels
#         g.ax_joint.set_xlabel("Component 1", fontsize=16)
#         g.ax_joint.set_ylabel("Component 2", fontsize=16)
#         g.ax_joint.tick_params(labelsize=12)
        
#         # Add histograms on the margins
#         sns.histplot(data=result_df, x="Component 1", ax=g.ax_marg_x, hue="Label", 
#                     palette=palette, legend=False, fill=False)
#         sns.histplot(data=result_df, y="Component 2", ax=g.ax_marg_y, hue="Label", 
#                     palette=palette, legend=False, fill=False)
        
#         # Now create the legend at the figure level and place it outside
#         handles, labels = g.ax_joint.get_legend_handles_labels()
        
#         # Remove the automatically added legend if it exists
#         if g.ax_joint.get_legend():
#             g.ax_joint.get_legend().remove()
        
#         # Add legend to the figure, not the joint axis
#         g.fig.legend(handles=handles, labels=labels, title='V Gene Family', 
#                     title_fontsize=18, fontsize=14,
#                     loc='upper center', bbox_to_anchor=(0.5, -0.02), 
#                     ncol=6,
#                     frameon=False, borderaxespad=0.)
        
#         # Adjust the layout to make room for the legend
#         plt.subplots_adjust(top=0.95, right=0.85)
        
#         # Save the plot
#         g.savefig(f'{plot_save_dir}/{plot_save_prefix}_pal_custom_{method_name}.svg', format="svg", dpi=450)
        
#         # Show the plot
#         plt.show()


# def plot_dimensionality_reduction(result, labels, method_name, plot_title_target=None, plot_save_prefix=None, model_name=None):
#     # Convert result to a DataFrame
#     palette = sns.color_palette(cc.glasbey, n_colors=25)

#     with sns.axes_style("white"):
#         result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
#         result_df['Label'] = labels  # Add labels for coloring

#         # Create a JointGrid with proper dimensions
#         g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)
        
#         # Plot the joint and marginal distributions
#         sns.scatterplot(data=result_df, x="Component 1", y="Component 2", 
#                     hue="Label", palette=palette, ax=g.ax_joint)
        
#         # Increase font size for axis labels
#         g.ax_joint.set_xlabel("Component 1", fontsize=18)
#         g.ax_joint.set_ylabel("Component 2", fontsize=18)
#         g.ax_joint.tick_params(labelsize=12)
        
#         # Add grid to the main plot
#         #g.ax_joint.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        
#         # # Clear the marginal axes first to prevent overplotting
#         # g.ax_marg_x.clear()
#         # g.ax_marg_y.clear()
        
#         # # Add KDE plots on the margins instead of histograms
#         # # For x-axis marginal
#         # sns.kdeplot(
#         #     data=result_df, 
#         #     x="Component 1", 
#         #     ax=g.ax_marg_x, 
#         #     hue="Label", 
#         #     palette=palette, 
#         #     legend=False,
#         #     linewidth=1.5,
#         #     common_norm=False,  # Each KDE curve normalized independently
#         #     bw_adjust=1.0,      # Bandwidth adjustment (1.0 is default, <1 for more details, >1 for smoother)
#         # )
        
#         # # For y-axis marginal
#         # sns.kdeplot(
#         #     data=result_df, 
#         #     y="Component 2", 
#         #     ax=g.ax_marg_y, 
#         #     hue="Label", 
#         #     palette=palette, 
#         #     legend=False,
#         #     linewidth=1.5,
#         #     common_norm=False,  # Each KDE curve normalized independently
#         #     bw_adjust=1.0,      # Bandwidth adjustment
#         # )
        
#         # # Remove axis labels from marginal plots
#         # g.ax_marg_x.set_xlabel("")
#         # g.ax_marg_x.set_ylabel("")
#         # g.ax_marg_y.set_xlabel("")
#         # g.ax_marg_y.set_ylabel("")
        
#         # Now create the legend at the figure level and place it outside
#         handles, labels = g.ax_joint.get_legend_handles_labels()
        
#         # Remove the automatically added legend if it exists
#         if g.ax_joint.get_legend():
#             g.ax_joint.get_legend().remove()
        
#         # Add legend to the figure, not the joint axis
#         g.fig.legend(handles=handles, labels=labels, title='V Gene Family', 
#                     title_fontsize=18, fontsize=14,
#                     loc='upper center', bbox_to_anchor=(0.5, -0.01), 
#                     ncol=6,
#                     frameon=False, borderaxespad=0.)
        
#         # Adjust the layout to make room for the legend
#         plt.subplots_adjust(bottom=0.13)  # Adjust bottom margin for legend
        
#         # Save the plot
#         g.savefig(f'{plot_save_dir}/{plot_save_prefix}_pal_custom_{method_name}.svg', format="svg", dpi=450)
        
#         # Show the plot
#         plt.show()



def plot_dimensionality_reduction(result, labels, method_name, plot_title_target=None, plot_save_prefix=None, model_name=None):
    # Convert result to a DataFrame
    palette = sns.color_palette(cc.glasbey, n_colors=25)

    with sns.axes_style("white"):
        result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
        result_df['Label'] = labels  # Add labels for coloring

        # Create a JointGrid with proper dimensions
        g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)
        
        # Plot the joint and marginal distributions
        sns.scatterplot(data=result_df, x="Component 1", y="Component 2", 
                    hue="Label", palette=palette, ax=g.ax_joint)
        
        # Increase font size for axis labels
        g.ax_joint.set_xlabel("Component 1", fontsize=18)
        g.ax_joint.set_ylabel("Component 2", fontsize=18)
        
        # Remove x-axis ticks and numbers
        g.ax_joint.set_xticks([])
        g.ax_joint.set_xticklabels([])

        # Remove x-axis and y-axis ticks and numbers
        g.ax_joint.set_xticks([])
        g.ax_joint.set_xticklabels([])
        g.ax_joint.set_yticks([])
        g.ax_joint.set_yticklabels([])
        
        # Keep y-axis ticks and numbers
        #g.ax_joint.tick_params(axis='y', labelsize=12)
        
        # Remove the top and right spines
        g.ax_joint.spines['top'].set_visible(False)
        g.ax_joint.spines['right'].set_visible(False)
        
        # Now create the legend at the figure level and place it closer to the plot
        handles, labels = g.ax_joint.get_legend_handles_labels()
        
        # Remove the automatically added legend if it exists
        if g.ax_joint.get_legend():
            g.ax_joint.get_legend().remove()
        
        # Add legend to the figure, closer to the plot
        g.fig.legend(handles=handles, labels=labels, title='V Gene Family', 
                    title_fontsize=18, fontsize=16,
                    loc='upper center', bbox_to_anchor=(0.5, -0.03), 
                    ncol=6,
                    frameon=False, borderaxespad=0.)
        
        # Adjust the layout to make room for the legend
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin for legend
        
        # Save the plot
        g.savefig(f'{plot_save_dir}/{plot_save_prefix}_pal_custom_{method_name}.svg', format="svg", dpi=450, transparent=True)
        
        # Show the plot
        plt.show()


#plot_title_target = "Loci"
#plot_title_target = "group"
#plot_save_prefix = "locus_light_light_input"
plot_save_prefix = "transparent_svg_dpi_450_FULL_final_legend_j_call_light_input_no_grid_fontsize_18"
#plot_save_prefix = "paired_digit_full_seq"
#plot_save_prefix = "BType_full_seq"
#plot_save_prefix = "group"


# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Plot PCA results using the function
plot_dimensionality_reduction(
    result=pca_result,
    labels=labels,
    method_name='PCA',
    #plot_title_target=plot_title_target,
    plot_save_prefix=plot_save_prefix,
    model_name=model_name
)


# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# Plot t-SNE results using the function
plot_dimensionality_reduction(
    result=tsne_result,
    labels=labels,
    method_name='t-SNE',
    #plot_title_target=plot_title_target,
    plot_save_prefix=plot_save_prefix,
    model_name=model_name
)


# Perform UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_reducer.fit_transform(embeddings)

# Plot UMAP results using the function
plot_dimensionality_reduction(
    result=umap_result,
    labels=labels,
    method_name='UMAP',
    #plot_title_target=plot_title_target,
    plot_save_prefix=plot_save_prefix,
    model_name=model_name
)

