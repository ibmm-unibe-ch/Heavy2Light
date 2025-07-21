import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, RobertaForMaskedLM, AutoModel, EncoderDecoderModel, AutoTokenizer
from adapters import init

# light and heavy MLM models
small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
#small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

#big_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-55740501"
# big_light_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-32519999"

# model_name = "light_model"

tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
model = RobertaForMaskedLM.from_pretrained(small_heavy_encoder)

# # light model unpaired gpt-2
# gpt_decoder = "/ibmm_data2/oas_database/paired_lea_tmp/light_model/gpt_model_light_unpaired/src/gpt_light_model_unpaired/model_outputs/full_new_tokenizer_gpt2_light_seqs_unp_lr_5e-4_wd_0.1_bs_32_epochs_500_/checkpoint-10196544"
# tokenizer = AutoTokenizer.from_pretrained(gpt_decoder)
# model = AutoModel.from_pretrained(gpt_decoder)

print(model)
base_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters in base model: {base_model_params}")


# protbert bfd NSP model

# model_name = "protbert_bfd_nsp"

# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/sweeps/model_checkpoints_FULL_data_7.july/model_checkpoint_nd526s39"

# model = AutoModel.from_pretrained(model_path)


# classification task with adapters

# Load the pre-trained model
# model_name = "IgBERT classification"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/trainer_small_test_lr_2e-05_batch_64_epochs_10_weight_decay_0.01_max_grad_norm_1.0_warmup_steps_1000"
# adapter_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/adapters_FULL_data_lr_1.0530326922846303e-06_batch_64_epochs_50_weight_decay_0.3028524952605797_warmup_steps_1000_max_grad_norm0.6196760588568332/class_adap"

# model = AutoModel.from_pretrained(model_path)

# print(model)

# encoder decoder model

# model_name = "encoder_decoder"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"

# model = EncoderDecoderModel.from_pretrained(model_path)
# print(model)

# igbert2igbert model

# # Load the pre-trained model
# model_name = "igbert2igbert"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0005_weight_decay_0.05"
# adapter_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0005_weight_decay_0.05/checkpoint-168020/seq2seq_adapter"

# model = EncoderDecoderModel.from_pretrained(model_path)

# # Initialize model
# model.to('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"model is on device: {model.device}")

# # Load the adapter
# init(model)
# model.load_adapter(adapter_path)
# model.set_active_adapters("seq2seq_adapter")    

# # Calculate trainable parameters for the base model (without adapters)
# base_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Trainable parameters in base model: {base_model_params}")

# # Print model structure for reference
# print(f"Model structure:\n {model}")

# igbert MLM model

# tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
# model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)

# print(model)
# base_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Trainable parameters in base model: {base_model_params}")
