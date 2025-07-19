from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig, AutoModel, BertTokenizer, RobertaForMaskedLM
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
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

# # BERT2BERT
# run_name = "PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-95615"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path 
# adapter_name = "heavy2light_adapter"

# gpt light unpaired epoch 41
run_name="gpt_unpaired_epoch_41"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-6058816"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# run_name = "heavyberta_small"
# small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
# #small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

# model_name = "bert_unpaired_heavy"


# tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
# model = RobertaForMaskedLM.from_pretrained(small_heavy_encoder)

# Initialize model and tokenizer
#model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

#model_name = "bert2gpt"
#model_name = "bert2bert"
model_name="gpt2_unpaired_light"

#sequence_heavy,locus_heavy,v_call_heavy,sequence_alignment_heavy,sequence_alignment_aa_heavy,germline_alignment_aa_heavy,
# cdr3_aa_heavy,sequence_light,locus_light,v_call_light,sequence_alignment_light,sequence_alignment_aa_light,germline_alignment_aa_light,
# cdr3_aa_light,sequence_alignment_heavy_sep_light,BType,Disease,Species,Subject,Author,Age,sequence_alignment_aa_light_1,generated_sequence_light,
# input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,calculated_similarity,general_v_gene_heavy,general_v_gene_light,v_gene_heavy_family,
# v_gene_light_family,alignment_germline,group
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/peak_analysis/bert2gpt/data/bert2gpt_df_merged_final_test_set.csv"

# paired and unpaired artificial dataset
#sequence_heavy,locus_heavy,v_call_heavy,sequence_alignment_heavy,sequence_alignment_aa_heavy,germline_alignment_aa_heavy,
#cdr3_aa_heavy,sequence_light,locus_light,v_call_light,sequence_alignment_light,sequence_alignment_aa_light,germline_alignment_aa_light,
#cdr3_aa_light,sequence_alignment_heavy_sep_light,BType,Disease,Species,Subject,Author,Age,sequence_alignment_aa_light_1,generated_sequence_light,
#input_heavy_sequence,BLOSUM_score,similarity,perplexity,calculated_blosum,calculated_similarity,general_v_gene_heavy,general_v_gene_light,v_gene_heavy_family,
#v_gene_light_family,alignment_germline,group,final_sequence_alignment_aa_light,paired_digit,full_sequence_heavy_light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/src/paired_artif_unpaired_dataset_bert2gpt_df_merged_final_test_set.csv"


# load test file as csv
test_df = pd.read_csv(test_file_path)

# Filter data to include only Memory-B-Cells or Naive-B-Cells
filtered_df = test_df[test_df["BType"].isin(["Memory-B-Cells", "Naive-B-Cells", "RV+B-Cells"])]

# Extract light sequences and labels for LDA
#light_sequences = filtered_df["sequence_alignment_aa_light_1"].tolist()
labels = filtered_df["BType"].tolist()  # Target labels

# test_df = load_data(test_file_path)
heavy_sequences = filtered_df["sequence_alignment_aa_light"].tolist()
#full_sequences = test_df["full_sequence_heavy_light"].tolist()
#labels = test_df['paired_digit'].tolist()

#heavy_sequences = test_df["input_heavy_sequence"].tolist()
#light_sequences = test_df["sequence_alignment_aa_light_1"].tolist()
#labels = test_df['BType'].tolist()
target="BType"
#target="paired_digit"
plot_title_target="BType"
#plot_title_target="paired_digit"


def get_last_layer_embeddings(model, tokenizer, sequences, device, model_name):
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
            if model_name == "bert2gpt":
                outputs = model.decoder(**inputs, output_hidden_states=True)  
            elif model_name == "gpt2_unpaired_light":
                outputs = model(**inputs, output_hidden_states=True)
            elif model_name == "bert2bert":
                outputs = model.decoder(**inputs, output_hidden_states=True)
            else:
                raise ValueError("Model name not recognized!")
            
            # for decoder only (gpt light model)
            #outputs = model(**inputs, output_hidden_states=True)

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
embeddings = get_last_layer_embeddings(model, tokenizer, heavy_sequences, device, model_name)
#embeddings = get_last_layer_embeddings(model, tokenizer, full_sequences, device, model_name)

# use LabelEncoder to encode target labels with value between 0 and n_classes-1. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Apply LDA (Number of components should be less than number of classes)
lda = LDA(n_components=2) # n_components = 2 for 2d visualization, j_call_fewer has 6 classes, therefore max n_components = 5
#lda = LDA(n_components=5) # n_components = 2 for 2d visualization, j_call_fewer has 6 classes, therefore max n_components = 5, if you choose n_components > 2 you cannot visualize the result in a plot


lda_result = lda.fit_transform(embeddings, encoded_labels)

# Save the LDA components to a DataFrame
lda_result_df = pd.DataFrame(lda_result, columns=['Component 1', 'Component 2'])


# Add the original labels to the DataFrame
lda_result_df[target] = labels

# Remove "RV+B-Cells" from the DataFrame used for plotting
lda_result_df = lda_result_df[lda_result_df[target] != "RV+B-Cells"]

# Create a larger JointGrid to plot the LDA components with marginal distributions
g = sns.JointGrid(data=lda_result_df, x="Component 1", y="Component 2", height=10)  # Increase height to make plot larger

#palette = sns.color_palette(cc.glasbey, n_colors=2)

#palette = 'Paired'
method_name = 'LDA'

# Define exact colors you want (using hex codes)
custom_colors = ["#1f77b4", "#ff7f0e"]  # Example: blue and orange

# For a simple approach, just use the custom colors directly
palette = custom_colors

# Get the unique labels (excluding "RV+B-Cells" since you removed it)
unique_labels = lda_result_df[target].unique()

# Create a dictionary mapping each label to a specific color
color_dict = {
    unique_labels[0]: "#1f77b4",  # First class gets blue
    unique_labels[1]: "#ff7f0e"   # Second class gets orange
}

# Then use this dictionary as your palette
g = g.plot(sns.scatterplot, sns.histplot, data=lda_result_df, hue=target, palette=color_dict)

# Plot the scatterplot of LDA components, explicitly passing the DataFrame and hue for coloring
#g = g.plot(sns.scatterplot, sns.histplot, data=lda_result_df, hue=target, palette=palette)

# Manually add a single legend, renaming it to 'J Gene Family'
handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(handles=handles, labels=labels, title='B-Cell Type', loc='lower right', frameon=False, fontsize=12, title_fontsize=14)


# Remove x and y axis ticks and numbers from joint plot
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
g.ax_joint.set_xticklabels([])
g.ax_joint.set_yticklabels([])

# Remove x and y axis ticks and numbers from marginal plots
g.ax_marg_x.set_xticks([])
g.ax_marg_x.set_xticklabels([])
g.ax_marg_y.set_yticks([])
g.ax_marg_y.set_yticklabels([])

# Increase font size for axis labels
g.ax_joint.set_xlabel('Component 1', fontsize=14)
g.ax_joint.set_ylabel('Component 2', fontsize=14)

# Adjust the layout to make room for the title and the legend
plt.subplots_adjust(top=0.9, right=0.85)

# Customize the plot title
#g.figure.suptitle(f'Differentiation of {plot_title_target} via LDA of Last Layer Embeddings', y=1.02)

plot_save_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/embedding_analysis/plots/{model_name}/{run_name}"
plot_save_prefix = "bigger_font_input_light_no_bord_no_title_BType_LDA_no_rv_b_cells"

# Create the directory if it does not exist
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)

# Save the plot
g.savefig(f'{plot_save_dir}/{plot_save_prefix}_pal_custom_colors_{method_name}.png', dpi=450, bbox_inches='tight')

# Show the plot
plt.show()

print(f"plot saved to {plot_save_dir}/{plot_save_prefix}_pal_custom_colors_{method_name}.png")

