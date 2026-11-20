"""Export the four representative models for the site's 3D viewers (docs/s/*.pdb) and print the
motif / footprint metadata that is pasted into scripts/site/page.html as META.

Shown regions only; low superposed on high via the Runt domain; coordinates rotated so the default
3D view matches the notebook figures.
Usage (from the project root): python scripts/site/export_structures.py . docs
"""
import sys, json, re, numpy as np, pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, Select, Superimposer
from scipy.spatial import cKDTree
P, OUTDIR = sys.argv[1], sys.argv[2]
OUT = f"{P}/af3_screen_n100_rho0.6/outputs_inference"
seqs = pd.read_csv(f"{P}/data/tangermeme_hard_syntax_top_and_worst_motif_examples_n100_rho0.6.csv", dtype={"motif_combination": str})
seqs["name"] = seqs.TF1 + "_" + seqs.TF2 + "_" + seqs.motif_combination + "_" + seqs["class"] + "_bg" + seqs.background_idx.astype(str)
seqs = seqs.set_index("name")
PAIRS = {
    "lef1": dict(high="RUNX1_LEF1_14_157_high_bg26", low="RUNX1_LEF1_14_157_low_bg45", A=[(50, 179)], B=[(299, 367)], cam="interface"),
    "ets1": dict(high="RUNX1_ETS1_14_64_high_bg16", low="RUNX1_ETS1_14_64_low_bg16", A=[(50, 212)], B=[(51, 136), (301, 441)], cam="dna"),
}
inr = lambda r, segs: any(s <= r <= e for s, e in segs)
rc = lambda x: x[::-1].translate(str.maketrans("ACGT", "TGCA"))
meta = {}
for key, cfg in PAIRS.items():
    M = {c: MMCIFParser(QUIET=True).get_structure(c, f"{OUT}/{cfg[c]}_model.cif")[0] for c in ["high", "low"]}
    ca = lambda m: [r["CA"] for r in m["A"] if 50 <= r.id[1] <= 179]
    sup = Superimposer(); sup.set_atoms(ca(M["high"]), ca(M["low"])); sup.apply(list(M["low"].get_atoms()))
    h = M["high"]
    heavy = lambda m, ch, segs: [(r.id[1], a.coord) for r in m[ch] if inr(r.id[1], segs) for a in r if a.element != "H"]
    def footprint(m):
        A, B = heavy(m, "A", cfg["A"]), heavy(m, "B", cfg["B"])
        t = cKDTree(np.array([c for _, c in B])); d, _ = t.query(np.array([c for _, c in A]))
        tA = cKDTree(np.array([c for _, c in A])); dB, _ = tA.query(np.array([c for _, c in B]))
        return sorted({r for (r, _), x in zip(A, d) if x <= 4.5}), sorted({r for (r, _), x in zip(B, dB) if x <= 4.5})
    fA, fB = footprint(h)
    dna = np.array([a.coord for c in "CD" for a in h[c].get_atoms()])
    if cfg["cam"] == "dna":
        pc = np.array([a.coord for c in "CD" for a in h[c].get_atoms() if a.get_id() == "P"])
        x = np.linalg.svd(pc - pc.mean(0))[2][0]
        u = np.array([c for _, c in heavy(h, "A", cfg["A"])]).mean(0) - pc.mean(0); u -= u.dot(x) * x; u /= np.linalg.norm(u)
        z = -np.cross(x, u); y = np.cross(z, x)
    else:
        a = np.array([c for r, c in heavy(h, "A", cfg["A"]) if r in fA]).mean(0); b = np.array([c for r, c in heavy(h, "B", cfg["B"]) if r in fB]).mean(0)
        z = (b - a) / np.linalg.norm(b - a); up = dna.mean(0) - a; y = up - up.dot(z) * z; y /= np.linalg.norm(y); x = np.cross(y, z)
        z, x = x, -z   # yaw 90 deg: side view of the interface
    R = np.vstack([x, y, z]).T.astype("f")           # new = (coord - centre) @ R
    centre = np.array([c for _, c in heavy(h, "A", cfg["A"])]).mean(0).astype("f")
    meta[key] = {}
    for cls in ["high", "low"]:
        m = M[cls]
        for a in m.get_atoms(): a.transform(np.eye(3, dtype="f"), -centre); a.transform(R, np.zeros(3, dtype="f"))
        class Sel(Select):
            def accept_residue(self, r):
                ch = r.get_parent().id
                return ch in "CD" or (ch == "A" and inr(r.id[1], cfg["A"])) or (ch == "B" and inr(r.id[1], cfg["B"]))
            def accept_atom(self, a): return a.element != "H"
        io = PDBIO(); io.set_structure(m); io.save(f"{OUTDIR}/s/{key}_{cls}.pdb", Sel())
        row = seqs.loc[cfg[cls]]; seq = row.sequence; motifs = []
        for tfk, mot in [("m1", row.motif1), ("m2", row.motif2)]:
            f, r_ = seq.find(mot), seq.find(rc(mot)); st = f if f >= 0 else r_
            motifs.append({"k": tfk, "start": st + 1, "len": len(mot), "fwd": f >= 0})
        fa, fb = footprint(m) if cls == "low" else (fA, fB)
        # footprint recomputed after transform is identical (rigid); for low use its own contacts
        meta[key][cls] = {"file": f"s/{key}_{cls}.pdb", "bg": int(cfg[cls].split("_bg")[1]), "seq": seq, "motifs": motifs, "foot": fa, "n": len(seq)}
        print(key, cls, "footprint", fa)
json.dump(meta, open(f"{OUTDIR}/meta.json", "w"))
print(json.dumps(meta))
