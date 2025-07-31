#This code processes a TSV dataset containing cluster and sequence IDs. 
#It extract the clusters from the clustered x% pident data (*cluster.tsv), sorts them by size, and allocates #sequences to training, validation, and test sets based on specified percentages.
#The script logs information about the dataset, cluster count, and sequence count.
#It then populates train_sequences, val_sequences, and test_sequences lists while maintaining entire clusters in #each set. Bigger cluster are kept for the training set.

# Example of usage
#python train_val_test_split_mmseqs.py --tsv_mmseqs swiss_prot_mmseqs_30pident_clusters.tsv --id_seq_file swiss_prot.csv --data_label 30pident_swissprot

import random
from collections import defaultdict
import argparse
import logging

parser = argparse.ArgumentParser(description='')
parser.add_argument("--tsv_mmseqs", help="", type=str) #tsv file
parser.add_argument("--id_seq_file", help="", type=str)#ID,SEQ file
parser.add_argument("--data_label", help="", type=str)

args = parser.parse_args()
main_dataset = args.tsv_mmseqs
rep_file = args.id_seq_file
data = args.data_label

#rep_file = data + '_linclust_rep_seq.fasta' #ID,SEQ file
print(rep_file)

# Configure logging settings
logging.basicConfig(filename= 'train_val_test_split.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f'DATASET: {main_dataset}')

# Load your TSV and create a dictionary of clusters
cluster_data = defaultdict(list)
with open(main_dataset, 'r') as tsv_file:
    for line in tsv_file:
        cluster_id, sequence_id = line.strip().split('\t')
        cluster_data[cluster_id].append(sequence_id)

logging.info("Total clusters (# centroids): %d", len(cluster_data))

# Sort clusters by size in descending order
seed = 42  # Change this to your desired seed
random.seed(seed)
random.shuffle(cluster_data)
sorted_clusters = sorted(cluster_data.items(), key=lambda x: len(x[1]), reverse=True)


# Set the percentages for train, validation, and test
train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

# number of sequences for each set
line_count = 0
with open(main_dataset, 'r') as file:
    for line in file:
        line_count += 1
logging.info(f'Total sequences {line_count}')

n_train_sequences = line_count * train_percentage
n_test_sequences = line_count * test_percentage
n_val_sequences = line_count * val_percentage

## add the sequences to each set keeping entire clusters together but based on the %
train_sequences = []
val_sequences = []
test_sequences = []

for cluster_id, sequences in sorted_clusters:
    
    if len(train_sequences) <= n_train_sequences:
        train_sequences.extend(sequences)
    elif len(val_sequences) <= n_val_sequences:
        val_sequences.extend(sequences)
    else:
        test_sequences.extend(sequences)
        
       

logging.info('TRAIN:')
logging.info(f'Number of sequences {len(train_sequences)}')
logging.info('VAL:')
logging.info(f'Number of sequences {len(val_sequences)}')
logging.info('TEST:')
logging.info(f'Number of sequences {len(test_sequences)}')


# create train,test,val file
# Write data to files -- ID
def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            f.write(f'{d}\n')

train_ids = data+'_ids_train.txt'
val_ids = data+'_ids_val.txt'
test_ids = data+'_ids_test.txt'

train_file = data+'_train.txt'
val_file = data+'_val.txt'
test_file = data+'_test.txt'

write_to_file(train_ids , train_sequences)
write_to_file(val_ids, test_sequences)
write_to_file(test_ids, val_sequences)

# retrieve the sequences --> out_file: ID,SEQ
import subprocess
#rep_file = data + '_linclust_rep_seq.fasta' #ID,SEQ 
# TRAIN
command = f'grep -Fwf {train_ids} {rep_file} > {train_file}'
print(command)
output = subprocess.check_output(command, shell=True)
# TEST
command = f'grep -Fwf {test_ids} {rep_file} > {test_file}'
print(command)
output = subprocess.check_output(command, shell=True)
# VAL
#command = f'./train_val_test_fasta_retrieval.sh {val_file} {rep_file}'
command = f'grep -Fwf {val_ids} {rep_file} > {val_file}'
print(command)
output = subprocess.check_output(command, shell=True)



logging.info('################################################')

