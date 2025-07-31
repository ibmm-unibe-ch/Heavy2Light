#!/bin/bash

cd ../data

# Specify the pident values you want to use
chains=("heavy" "light")

for pident in "${pidents[@]}"
do
    for chain in "${chains[@]}"
    do
        echo "Running for chain: $chain with pident: $pident"
        
        # Run the Python script with the current pident and chain value
        python ../src/train_val_test_split_mmseqs.py \
            --tsv_mmseqs ./mmseqs/${chain}_cluster.tsv \
            --id_seq_file ${chain}_naive_memory_balanced.csv \
            --data_label ${chain}

        echo "id,aa_sequence,label" | cat - ${chain}_val.txt > ${chain}_val.csv && rm ${chain}_val.txt
        echo "id,aa_sequence,label" | cat - ${chain}_train.txt > ${chain}_train.csv && rm ${chain}_train.txt
        echo "id,aa_sequence,label" | cat - ${chain}_test.txt > ${chain}_test.csv && rm ${chain}_test.txt
    done
done
