#!/usr/bin/env python3
"""Build Stage-2 (inference-only) input JSONs for all 6400 CSV rows.

Must run AFTER Stage 1 has produced outputs_msa/<TF1>_<TF2>_<combo>_data.json
for every triple (that file embeds the protein MSA). This script clones each
triple's data.json once per row and swaps in that row's real DNA (chain C) and
its reverse complement (chain D), so Stage 2 never re-runs the data pipeline.

Output: inputs_inference/<TF1>_<TF2>_<combo>_<class>_bg<idx>.json
        job_list_inference.txt
"""
import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "tangermeme_hard_syntax_top_and_worst_motif_examples_n100_rho0.6.csv"
MSA_OUT_DIR = ROOT / "outputs_msa"
OUT_DIR = ROOT / "inputs_inference"

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def set_dna(template: dict, dna: str) -> dict:
    job = copy.deepcopy(template)
    rc = revcomp(dna)
    for s in job["sequences"]:
        if "dna" in s:
            ids = s["dna"]["id"] if isinstance(s["dna"]["id"], list) else [s["dna"]["id"]]
            if "C" in ids:
                s["dna"]["sequence"] = dna
            elif "D" in ids:
                s["dna"]["sequence"] = rc
    return job


def main() -> None:
    templates = {}
    missing_templates = set()

    OUT_DIR.mkdir(exist_ok=True)
    names = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            tf1, tf2, combo = row["TF1"], row["TF2"], row["motif_combination"]
            triple = f"{tf1}_{tf2}_{combo}"
            if triple not in templates:
                p = MSA_OUT_DIR / f"{triple}_data.json"
                if not p.exists():
                    missing_templates.add(triple)
                    continue
                templates[triple] = json.loads(p.read_text())

            dna = row["sequence"].upper()
            if "N" in dna:
                raise SystemExit(f"Unexpected N in DNA for {triple} bg{row['background_idx']}: {dna}")

            name = f"{triple}_{row['class']}_bg{row['background_idx']}"
            job = set_dna(templates[triple], dna)
            job["name"] = name
            (OUT_DIR / f"{name}.json").write_text(json.dumps(job))
            names.append(name)

    if missing_templates:
        print(f"ERROR: missing Stage-1 outputs_msa/*_data.json for {len(missing_templates)} triples:", file=sys.stderr)
        for t in sorted(missing_templates):
            print(f"  {t}", file=sys.stderr)
        raise SystemExit("Run Stage 1 (02_msa.sbatch) to completion first.")

    (ROOT / "job_list_inference.txt").write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} inference-ready JSONs to {OUT_DIR}")


if __name__ == "__main__":
    main()
