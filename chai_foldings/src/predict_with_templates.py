import logging
from pathlib import Path

from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)

# See RCSB identifier 7WCU
example_fasta = """
>protein|101
TNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST
>protein|102
EVQLVESGGGLIQPGGSLRLSCAASEFIVSRNYMSWVRQAPGTGLEWVSVIYPGGSTFYADSVKGRFTISRDNSKNTLYLQMDSLRVEDTAVYYCARDYGDFYFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDK
>protein|103
EIVMTQSPVSLSVSPGERATLSCRASQGVASNLAWYQQKAGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISTLQSEDSAVYYCQQYNDRPRTFGQGTKLEIKRT
""".strip()

fasta_path = Path("/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/data/example.fasta")
fasta_path.write_text(example_fasta)

output_dir = Path("/ibmm_data2/oas_database/paired_lea_tmp/chai_foldings/chai_outputs")

candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    use_msa_server=True,
    use_templates_server=True,
    seed=1234,
    device="cuda:0",
)