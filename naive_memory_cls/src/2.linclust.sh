#!/usr/bin/bash

# mmseqs linclustering for each type within the groups and for all
mkdir -p '../data/mmseqs/'

cd '../data/mmseqs/' 
# linclust
mmseqs easy-linclust ../heavy_naive_memory_balanced.fasta ./heavy heavy
mmseqs easy-linclust ../light_naive_memory_balanced.fasta ./light light
