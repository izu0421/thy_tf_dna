#!/usr/bin/env python3
"""Build Stage-1 (MSA) input JSONs: one per unique (TF1, TF2, motif_combination).

Only 32 unique protein pairs are needed for the whole 6400-row CSV, since within
a motif_combination the DNA changes but the protein pair MSA does not. Each job
uses the background_idx==0 DNA for that triple as a placeholder - Stage 2 swaps
in the real per-row DNA later, reusing the *_data.json (which embeds the MSA).

Output: inputs_msa/<TF1>_<TF2>_<combo>.json  (one AF3 job per triple)
        job_list_msa.txt                     (one job name per line, for the array script)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "tangermeme_hard_syntax_top_and_worst_motif_examples_n100_rho0.6.csv"
SEQS = ROOT / "data" / "tf_sequences_run.json"
OUT_DIR = ROOT / "inputs_msa"

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def main() -> None:
    seqs = json.loads(SEQS.read_text())

    reps = {}  # (TF1, TF2, combo) -> representative row
    with CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row["TF1"], row["TF2"], row["motif_combination"])
            idx = int(row["background_idx"])
            if key not in reps or idx < int(reps[key]["background_idx"]):
                reps[key] = row

    OUT_DIR.mkdir(exist_ok=True)
    names = []
    for (tf1, tf2, combo), row in sorted(reps.items()):
        missing = [g for g in (tf1, tf2) if g not in seqs]
        if missing:
            raise SystemExit(f"No sequence cached for: {missing}")
        dna = row["sequence"].upper()
        name = f"{tf1}_{tf2}_{combo}"
        job = {
            "name": name,
            "sequences": [
                {"protein": {"id": ["A"], "sequence": seqs[tf1]}},
                {"protein": {"id": ["B"], "sequence": seqs[tf2]}},
                {"dna": {"id": ["C"], "sequence": dna}},
                {"dna": {"id": ["D"], "sequence": revcomp(dna)}},
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
        (OUT_DIR / f"{name}.json").write_text(json.dumps(job, indent=2))
        names.append(name)
        print(f"wrote {name}.json (A={tf1} {len(seqs[tf1])}aa, B={tf2} {len(seqs[tf2])}aa)")

    (ROOT / "job_list_msa.txt").write_text("\n".join(names) + "\n")
    print(f"\n{len(names)} MSA input JSONs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
