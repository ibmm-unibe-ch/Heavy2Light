import tempfile
from pathlib import Path

import numpy as np

from chai_lab.chai1 import run_inference


fasta_path = "/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/data/all_seqs_generated_gpt2_sequences_10_no_ids.fasta"


# Generate structure
output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/chai_outputs"
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    # See example .aligned.pqt files in this directory
    msa_directory=Path(__file__).parent,
    # Exclusive with msa_directory; can be used for MMseqs2 server MSA generation
    use_msa_server=False,
)
cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
