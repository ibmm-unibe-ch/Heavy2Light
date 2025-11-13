#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=permutation_coherence_10000iter
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/permutation_coherence_10000iter_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/permutation_coherence_10000iter_%j.e


# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2


# Set paths
CSV_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/multiple_light_seqs_from_single_heavy/full_test_set_multiple_light_seqs/full_eval_generate_multiple_light_seqs_203276_cls_predictions_merged_genes.csv"
OUTPUT_DIR="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/coherence_plots_with_random"

# Change to output directory
cd $OUTPUT_DIR

# Run the analysis
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/src/run_randomization_control.py \
    --csv "$CSV_FILE" \
    --iterations 10000 \
    --min-seqs 4 \
    --threshold 80 \
    --suffix "_10000iter" \
    --seed 42 \
    --output randomization_results_10000iter.pkl

echo ""
echo "End time: $(date)"
echo "Job completed successfully"
