#!/bin/bash


DATA_PATH="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/coherence_analysis_in_oas_db/data/full_extraction_for_coherence_paired_data_header.csv"
DATA_OUTPUT="../data"
CHAIN="heavy"


python create_data_splits.py --data_path "$DATA_PATH" \
                        --data_output "$DATA_OUTPUT" \
                        --chain "$CHAIN"
                        

CHAIN="light"
DATA_OUTPUT="../data"

python create_data_splits.py --data_path "$DATA_PATH" \
                        --data_output "$DATA_OUTPUT" \
                        --chain "$CHAIN"
                        
