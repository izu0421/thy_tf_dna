#!/usr/bin/env python3
"""Build inference-only AF3 inputs for the RUNX1-ETS1 positive control vs Shrivastava et al. 2014 (PDB 4L0Z).

2 protein forms x 3 DNAs, 5 seeds each:
  proteins: full     = RUNX1 1-453 + ETS1 1-441 (as in the n100_rho0.6 screen)
            construct = RUNX1 1-242 + ETS1 296-441 (the crystallised fragments)
  DNA:      tcra = 4L0Z TCRa enhancer strands exactly as crystallised (16 nt each, 1-nt 5' overhangs)
            high = RUNX1_ETS1_14_64_high_bg81 (representative high model in notebooks 4/6)
            low  = RUNX1_ETS1_14_64_low_bg30  (representative low model)

MSAs/templates are reused from the screen's Stage-1 output (no new MSA search). For the construct,
the full-length unpaired/paired MSAs are sliced to the construct columns (insertions inside the range
kept) and template query indices are shifted, so every condition sees identical evolutionary input.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MSA_JSON = ROOT.parent / "af3_screen_n100_rho0.6" / "outputs_msa" / "RUNX1_ETS1_14_64_data.json"
OUT_DIR = ROOT / "inputs_inference"
SEEDS = [1, 2, 3, 4, 5]

COMPLEMENT = str.maketrans("ACGT", "TGCA")
DNAS = {
    "tcra": ("GGAAGCCACATCCTCT", "CAGAGGATGTGGCTTC"),
    "high": ("ATCAGGAAGTTGTGGTTGT", None),
    "low": ("CCTGTGGTTCAGGAAGTGT", None),
}
CONSTRUCTS = {"full": {"A": (1, 453), "B": (1, 441)}, "construct": {"A": (1, 242), "B": (296, 441)}}


def slice_a3m(a3m, start, end):
    """Keep match columns start..end (1-based, inclusive) of every a3m row, plus insertions between them."""
    out = []
    for line in a3m.splitlines():
        if line.startswith(">") or not line:
            out.append(line)
            continue
        col, keep = 0, []
        for ch in line:
            if ch.isupper() or ch == "-":
                col += 1
                if start <= col <= end:
                    keep.append(ch)
            elif start <= col < end:  # lowercase insertion after column col
                keep.append(ch)
        out.append("".join(keep))
    return "\n".join(out) + "\n"


def slice_protein(entry, start, end):
    e = copy.deepcopy(entry)
    e["sequence"] = entry["sequence"][start - 1:end]
    e["unpairedMsa"] = slice_a3m(entry["unpairedMsa"], start, end)
    e["pairedMsa"] = slice_a3m(entry["pairedMsa"], start, end)
    q = e["unpairedMsa"].splitlines()[1]
    assert q == e["sequence"], "query row of sliced MSA does not match construct sequence"
    templates = []
    for t in entry["templates"] or []:
        pairs = [(qi - (start - 1), ti) for qi, ti in zip(t["queryIndices"], t["templateIndices"])
                 if start - 1 <= qi <= end - 1]
        if pairs:
            t2 = dict(t)
            t2["queryIndices"] = [p[0] for p in pairs]
            t2["templateIndices"] = [p[1] for p in pairs]
            templates.append(t2)
    e["templates"] = templates
    return e


def main():
    base = json.loads(MSA_JSON.read_text())
    OUT_DIR.mkdir(exist_ok=True)
    names = []
    for form, ranges in CONSTRUCTS.items():
        for dna_name, (top, bottom) in DNAS.items():
            bottom = bottom or top.translate(COMPLEMENT)[::-1]
            job = {"dialect": "alphafold3", "version": base["version"], "modelSeeds": SEEDS,
                   "bondedAtomPairs": None, "userCCD": None, "sequences": []}
            for s in base["sequences"]:
                if "protein" in s:
                    chain = s["protein"]["id"] if isinstance(s["protein"]["id"], str) else s["protein"]["id"][0]
                    start, end = ranges[chain]
                    job["sequences"].append({"protein": slice_protein(s["protein"], start, end)})
            job["sequences"].append({"dna": {"id": "C", "sequence": top}})
            job["sequences"].append({"dna": {"id": "D", "sequence": bottom}})
            name = f"RUNX1_ETS1_{form}_{dna_name}"
            job["name"] = name
            (OUT_DIR / f"{name}.json").write_text(json.dumps(job))
            names.append(name)
            prot = [x["protein"] for x in job["sequences"] if "protein" in x]
            print(f"{name}: A={len(prot[0]['sequence'])}aa ({len(prot[0]['templates'])} templates), "
                  f"B={len(prot[1]['sequence'])}aa, DNA {top}/{bottom}, seeds {SEEDS}")
    (ROOT / "job_list.txt").write_text("\n".join(names) + "\n")


if __name__ == "__main__":
    main()
