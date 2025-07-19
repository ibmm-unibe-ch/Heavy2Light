import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import pandas as pd
from adapters import init

plotting_style="seaborn"

def get_mlm_last_layer_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
            # Ensure to request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Access the last hidden states from the hidden_states tuple
            last_hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states
            
            # Mean pooling across the sequence length dimension
            mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled_output)
    
    return np.vstack(embeddings)  # Stack into a numpy array



small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

model_name = "heavy_model"

# Load a pre-trained BERT model and tokenizer
#tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
#model = BertModel.from_pretrained(small_heavy_encoder)

tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
model = RobertaForMaskedLM.from_pretrained(small_heavy_encoder)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FULL test path with disease and species
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/spaces_full_test_data_extraction_species_diseases_no_dupl.txt"

# FULL test path only with BType memory and naive input format: Species,Disease,BType,Isotype_light,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl.csv"

# FULL test path input format: "Species,Disease,BType,Isotype_light,locus_heavy,locus_light,v_call_heavy,d_call_heavy,j_call_heavy,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light"
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt"

# FULL test file path with v d and j genes 
# v_call_fewer_heavy,d_call_fewer_heavy,j_call_fewer_heavy,v_call_fewer_heavy_star,v_call_heavy,d_call_heavy,j_call_heavy,sequence_alignment_heavy_sep_light
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/fewer_genes_relevant_cols_spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt"

# FULL test path with lambda and kappa subtypes (only subtypes without specific gene) file format heavy[SEP]light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/updated_test_file_subtypes.csv"

# small test path
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl_small.csv"

# FULL unpaired light test seqs
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl.txt"

# FULL unpaired light test seqs 80'000 rows
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl_80000.txt"

#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_gene_names_extracted_seqs_light_model_test_set_no_dupl_80000.txt"

# FULL unpaired heavy test seqs
# col names: v_call_fewer,j_call_fewer,d_call_fewer,v_call_fewer_star,v_call,d_call,j_call,sequence_alignment_aa
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_genes_extracted_seqs_heavy_model_test_set_80000.txt"

# load test file as csv
test_df_labels = pd.read_csv(test_file_path)



# if input file is of the format heavy[SEP]light use this function for the extrraction of heavy and light sequences
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    sequences = []
    for entry in data:
        split_entry = entry.split(' [SEP] ')
        if len(split_entry) == 2:
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")
    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

test_df_labels = load_data(test_file_path)

#target = "locus"
#target = "v_call"
#target = "d_call_fewer"
#target="BType"
#target="Age"
target = 'v_call_fewer'
#target = 'v_call_heavy'
#target = 'd_call_fewer_heavy'
#target = 'j_call_fewer_heavy'
#target = 'v_call_fewer_heavy'
#target = 'locus'
#target = 'subtype'



# test_df = load_data(test_file_path)
heavy_sequences = test_df_labels["heavy"].tolist()
# light_sequences = test_df["light"].tolist()
#labels = test_df_labels['v_family'].tolist()
#labels = test_df_labels['subtype'].tolist()
labels = test_df_labels[f'{target}'].tolist()

#plot_title_target = "B-Cell Types"
#plot_title_target = "Age"
plot_title_target = "V Gene Families"
#plot_title_target = "Kappa / Lambda Loci"
#plot_title_target = "J Gene Families"
#plot_title_target = "D Gene Families"
#plot_save_prefix = "heavy_v_gene_families"
#plot_save_prefix = "v_call_fewer_heavy_star_better_layout2"
#plot_save_prefix = "locus_unpaired_light_seqs_80000"
#plot_save_prefix = "jgenes_fewer_unpaired_light_seqs_80000"
#plot_save_prefix = "BTypes_unpaired_heavy_seqs_80000"
plot_save_prefix = f"{plotting_style}_fewer_vgenes_paired_heavy_seqs"
#plot_save_prefix = "Age_unpaired_light_seqs_80000"
#plot_save_prefix = "fewer_heavy_j_gene_families"
#plot_save_prefix = "heavy_d_gene_families"

#heavy_sequences = filtered_df['sequence_alignment_aa'].tolist()

embeddings = get_mlm_last_layer_embeddings(model, tokenizer, heavy_sequences, device)


def plot_dimensionality_reduction(result, labels, method_name, plot_title_target, plot_save_prefix, model_name):
    # Convert result to a DataFrame
    result_df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
    result_df['Label'] = labels  # Add labels for coloring

    # Create the plot using Seaborn
    g = sns.JointGrid(data=result_df, x="Component 1", y="Component 2", height=10)

    palette="icefire"
    
    # Plot scatter plot with hue (based on the label column)
    g = g.plot(sns.scatterplot, sns.histplot, data=result_df, hue="Label", palette=palette)

    # Manually add the legend with the title "Label"
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=labels, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Customize the plot title
    g.figure.suptitle(f'Differentiation of {plot_title_target} via {method_name} of Last Layer Embeddings', y=1.02)

    # Save the plot
    g.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/{method_name}/{plot_save_prefix}_pal_{palette}_mlm_heavy2light_FULL_data.png')

    # Show the plot
    plt.show()


# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Plot PCA results using the function
plot_dimensionality_reduction(
    result=pca_result,
    labels=labels,
    method_name='PCA',
    plot_title_target=plot_title_target,
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
    plot_title_target=plot_title_target,
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
    plot_title_target=plot_title_target,
    plot_save_prefix=plot_save_prefix,
    model_name=model_name
)

