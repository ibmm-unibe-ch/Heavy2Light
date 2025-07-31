import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def load_and_prepare_naive_memory_data(data_path, chain):
    # Carica il dataset e filtra per Memory-B-Cells e Naive-B-Cells
    df = pd.read_csv(data_path)
    df_filtered = df[df["BType"].isin(["Memory-B-Cells", "Naive-B-Cells"])].copy()
    
    if chain == 'heavy':
        aa_sequence = 'sequence_alignment_aa_heavy'
    if chain == 'light':
        aa_sequence = 'sequence_alignment_aa_light'
    
    df_filtered = df_filtered[[aa_sequence, "BType"]]
    df_filtered = df_filtered.drop_duplicates(subset=[aa_sequence])
    
    memory_count = df_filtered[df_filtered["BType"] == "Memory-B-Cells"].shape[0]
    naive_count = df_filtered[df_filtered["BType"] == "Naive-B-Cells"].shape[0]
    print(f'##### CHAIN: {chain}')
    print(f"# datapoint Memory-B-Cells after removing duplicates: {memory_count}")
    print(f"# datapoint Naive-B-Cells after removing duplicates: {naive_count}")
    
    # Separa e campiona per creare un dataset bilanciato
    df_memory = df_filtered[df_filtered["BType"] == "Memory-B-Cells"]
    df_naive = df_filtered[df_filtered["BType"] == "Naive-B-Cells"]
    
    group_size = min(len(df_memory), len(df_naive))
    if len(df_memory) < group_size or len(df_naive) < group_size:
        raise ValueError(f"Non ci sono abbastanza campioni per creare un dataset bilanciato con {group_size} campioni per classe.")

    
    df_memory_sample = df_memory.sample(n=group_size, random_state=42)
    df_naive_sample = df_naive.sample(n=group_size, random_state=42)
    
    print(f"# datapoint Memory-B-Cells after balancing: {len(df_memory_sample)}")
    print(f"# datapoint Naive-B-Cells after balancing: {len(df_naive_sample)}")
    df_balanced = pd.concat([df_memory_sample, df_naive_sample]).sample(frac=1, random_state=42)
    
    # Mappa BType in etichette numeriche
    label_mapping = {"Naive-B-Cells": 0, "Memory-B-Cells": 1}
    df_balanced["label"] = df_balanced["BType"].map(label_mapping)
    df_balanced.rename(columns={aa_sequence: 'aa_sequence'}, inplace=True)

    
    return df_balanced

def main():
    parser = argparse.ArgumentParser(
        description="Crea split train, validation e test a partire dal dataset originale"
    )
    parser.add_argument("--data_path", type=str, required=True, help="Percorso del file CSV originale")
    parser.add_argument("--chain", type=str, required=True, help="Heavy or light chain")
    parser.add_argument("--data_output", type=str, default="train.csv", help="Percorso di output per il CSV di train")
    
    args = parser.parse_args()
    
    df_balanced = load_and_prepare_naive_memory_data(args.data_path, args.chain)
    
    ### RANDOM SPLIT
    # Dividi in train (80%) e validation+test (20%), quindi dividi validation e test a metà
    #train_df, val_test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
    #val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
    
    #print(f"Dimensione train: {len(train_df)}")
    #print(f"Dimensione validation: {len(val_df)}")
    #print(f"Dimensione test: {len(test_df)}")

    # Costruisci i percorsi dei file CSV per train, validation e test
    #train_path = os.path.join(args.data_output, f"{args.chain}_train.csv")
    #val_path = os.path.join(args.data_output, f"{args.chain}_val.csv")
    #test_path = os.path.join(args.data_output, f"{args.chain}_test.csv")

    # Salva i DataFrame nei rispettivi file CSV
    #train_df.to_csv(train_path, index=False)
    #val_df.to_csv(val_path, index=False)
    #test_df.to_csv(test_path, index=False)

    #print(f"Splits salvati in: {train_path}, {val_path} e {test_path}")
    
    ### PREPAR DF BALANCED FOR MMSEQS
    df_balanced = df_balanced[["aa_sequence","label"]]
    df_balanced["id_label"] = (
        df_balanced.groupby("label").cumcount().astype(str) + "_" + df_balanced["label"].astype(str)
    )
    # Riordina le colonne come richiesto
    df_balanced_ordered = df_balanced[["id_label", "aa_sequence", "label"]]
    # Salva in CSV
    df_balanced_ordered.to_csv(os.path.join(args.data_output,f"{args.chain}_naive_memory_balanced.csv"), index=False)
    # Scrive ogni riga come record FASTA
    #val_path =
    with open(os.path.join(args.data_output,f"{args.chain}_naive_memory_balanced.fasta"), "w") as fasta_file:
        for _, row in df_balanced.iterrows():
            fasta_file.write(f">{row['id_label']}\n{row['aa_sequence']}\n")


if __name__ == "__main__":
    main()
