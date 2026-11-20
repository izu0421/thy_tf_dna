#!/usr/bin/env python3
"""List the 6400 Stage-2 job names, in CSV row order.

Names only depend on the CSV (TF1/TF2/combo/class/background_idx), not on the
actual MSA content, so this can run before Stage 1 exists - it's what lets us
pre-generate the per-shard sbatch files locally, ready to ship to the HPC.
Naming must match scripts/03_build_inference_inputs.py exactly.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "tangermeme_hard_syntax_top_and_worst_motif_examples_n100_rho0.6.csv"


def main() -> None:
    names = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            triple = f"{row['TF1']}_{row['TF2']}_{row['motif_combination']}"
            names.append(f"{triple}_{row['class']}_bg{row['background_idx']}")

    (ROOT / "job_list_inference.txt").write_text("\n".join(names) + "\n")
    print(f"{len(names)} job names written to job_list_inference.txt")


if __name__ == "__main__":
    main()
