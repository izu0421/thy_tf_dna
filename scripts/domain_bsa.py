#!/usr/bin/env python3
"""TF1-TF2 buried surface area restricted to annotated structured domains, for every screen model.

Domains come from data/tf_domains_structured.csv (UniProt + Pfam, see scripts/fetch_tf_domains.py),
not from model confidence. Per model (chains A = TF1, B = TF2; DNA excluded):
  BSA_domain     0.5 * (SASA(A_dom) + SASA(B_dom) - SASA(A_dom + B_dom)), Shrake-Rupley as notebook 0
  domains_touch  any heavy atom of A_dom within 4 A of B_dom
  n_iface_res    domain residues (A + B) with a heavy atom within 4 A of the other TF's domains
  plddt_*        mean CA pLDDT inside / outside the annotated domains (QC only)

Writes results/cache/domain_bsa_checkpoint.csv; resumable.
Usage: python scripts/domain_bsa.py [n_workers]
"""
import copy
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "af3_screen_n100_rho0.6" / "outputs_inference"
CSV = ROOT / "data" / "tangermeme_hard_syntax_top_and_worst_motif_examples_n100_rho0.6.csv"
DOMAINS = pd.read_csv(ROOT / "data" / "tf_domains_structured.csv")
CHECKPOINT = ROOT / "results" / "cache" / "domain_bsa_checkpoint.csv"

DOM_RES = {tf: set().union(*(range(s, e + 1) for s, e in zip(g["start"], g["end"])))
           for tf, g in DOMAINS.groupby("TF")}


def yy_keep(model, keep):
    """Copy of model with only {chain_id: set(resnums)}."""
    m = copy.deepcopy(model)
    for ch in list(m):
        if ch.id not in keep:
            m.detach_child(ch.id)
            continue
        for res in list(ch):
            if res.id[1] not in keep[ch.id]:
                ch.detach_child(res.id)
    return m


def yy_sasa(model):
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(model, level="A")
    return sum(a.sasa for a in model.get_atoms())


def yy_one(row):
    name, tf1, tf2 = row
    model = MMCIFParser(QUIET=True).get_structure(name, OUT / f"{name}_model.cif")[0]
    dom = {"A": DOM_RES[tf1], "B": DOM_RES[tf2]}
    sasa_a = yy_sasa(yy_keep(model, {"A": dom["A"]}))
    sasa_b = yy_sasa(yy_keep(model, {"B": dom["B"]}))
    both = yy_keep(model, dom)
    sasa_ab = yy_sasa(copy.deepcopy(both))

    atoms_a = [a for a in both["A"].get_atoms() if a.element != "H"]
    atoms_b = [a for a in both["B"].get_atoms() if a.element != "H"]
    ns = NeighborSearch(atoms_b)
    iface_a, iface_b = set(), set()
    for a in atoms_a:
        close = ns.search(a.coord, 4.0)
        if close:
            iface_a.add(a.get_parent().id[1])
            iface_b.update(b.get_parent().id[1] for b in close)

    rec = {"name": name, "BSA_domain": max(0.5 * (sasa_a + sasa_b - sasa_ab), 0.0),
           "domains_touch": bool(iface_a), "n_iface_res": len(iface_a) + len(iface_b)}
    for ch in "AB":
        ca = {r.id[1]: r["CA"].get_bfactor() for r in model[ch] if "CA" in r}
        inside = [v for k, v in ca.items() if k in dom[ch]]
        outside = [v for k, v in ca.items() if k not in dom[ch]]
        rec[f"plddt_dom_{ch}"] = float(np.mean(inside)) if inside else np.nan
        rec[f"plddt_rest_{ch}"] = float(np.mean(outside)) if outside else np.nan
    return rec


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    dt = pd.read_csv(CSV, dtype={"motif_combination": str})
    dt["name"] = (dt["TF1"] + "_" + dt["TF2"] + "_" + dt["motif_combination"] + "_" + dt["class"]
                  + "_bg" + dt["background_idx"].astype(str))
    done = pd.read_csv(CHECKPOINT) if CHECKPOINT.exists() else pd.DataFrame(columns=["name"])
    todo = dt[~dt["name"].isin(done["name"])]
    print(f"{len(done)} done, {len(todo)} to do, {n_workers} workers", flush=True)

    records = done.to_dict("records")
    with Pool(n_workers) as pool:
        jobs = [tuple(r) for r in todo[["name", "TF1", "TF2"]].to_numpy()]
        for i, rec in enumerate(pool.imap_unordered(yy_one, jobs, chunksize=4)):
            records.append(rec)
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                pd.DataFrame(records).to_csv(CHECKPOINT, index=False)
                print(f"{i + 1}/{len(todo)}", flush=True)
    print("done", flush=True)
