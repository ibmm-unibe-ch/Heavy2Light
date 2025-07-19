#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
##SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mmseqs_cdrh3
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/mmseqs_50_extended_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/mmseqs_50_extended_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate OAS_paired_env


if [ "$#" -ne 1 ]; then
    echo "DATA: $0 DB "
    exit 1
fi

DB="$1"
echo "DATA: $0 DB "
mmseqs createdb "$DB".fasta "$DB"

log_file="$DB""_mmseqs_linclust.log"
echo "Log saved to: $log_file"
db_rows=$(grep -o ">" "$DB".fasta | wc -l)
echo "Number of sequences in $DB.fasta: $db_rows" > "$log_file"
echo "" >> "$log_file"

echo "#####################################################################################################################################"  >> "$log_file"

##pident 50
min_seq_id=0.5
pident=50
#sensitivity=9.5
kmer=5
mmseqs linclust "$DB" "${DB}_${pident}_clu" tmp --min-seq-id "$min_seq_id" --kmer-per-seq "$kmer" -k 6 --spaced-kmer-mode 1 --spaced-kmer-pattern 11011101 --sub-mat VTML40.out --gap-open 16 --gap-extend 2
mmseqs createsubdb "${DB}_${pident}_clu" "$DB" "${DB}_${pident}_clu_rep"
mmseqs convert2fasta "${DB}_${pident}_clu_rep" "${DB}_${pident}_clu_rep.fasta"
mmseqs createtsv "$DB" "$DB" "${DB}_${pident}_clu" "${DB}_${pident}_clu.tsv"
mmseqs createseqfiledb "$DB" "${DB}_${pident}_clu" "${DB}_${pident}_clu_allseqs"
mmseqs result2flat "$DB" "$DB" "${DB}_${pident}_clu_allseqs" "${DB}_${pident}_clu_allseqs.fasta" --use-fasta-header

awk '/^>/ { printf("\n%s,", substr($0, 2)); next; } { printf("%s", $0);} END { printf("\n"); }' "${DB}_${pident}_clu_rep.fasta" > "${DB}_${pident}_clu_rep_idseq"
awk -F ',' '!seen[$2]++' "${DB}_${pident}_clu_rep_idseq" > "${DB}_${pident}_clu_rep_idseq_noduplicates"


clu_rows=$(wc -l < "${DB}_${pident}_clu.tsv")
rep_rows=$(wc -l < "${DB}_${pident}_clu_rep_idseq")
unique_sequences=$(cut -d ',' -f 2 "${DB}_${pident}_clu_rep_idseq" | sort | uniq | grep -c "")
rep_rows_noduplicates=$(wc -l < "${DB}_${pident}_clu_rep_idseq_noduplicates")
echo "${pident} CLUSTERING"
echo "Number of rows in ${DB}_${pident}_clu.tsv: $clu_rows" >> "$log_file"
echo "Number of rows in ${DB}_${pident}_clu_rep_idseq: $rep_rows" >> "$log_file"
echo "Number of unique centroids sequences in the second column: $unique_sequences" >> "$log_file"
echo "Number of rows in ${DB}_${pident}_clu_rep_idseq_noduplicates after removing duplicates: $rep_rows_noduplicates" >> "$log_file"
echo "" >> "$log_file"
rm -r tmp

echo "" >> "$log_file"
echo "#####################################################################################################################################"  >> "$log_file"
echo "#####################################################################################################################################"  >> "$log_file"
echo "#####################################################################################################################################"  >> "$log_file"
