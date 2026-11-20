#!/usr/bin/env python3
"""Fetch curated domain annotations for every TF in the AF3 screen.

For each TF in data/tf_sequences_run.json:
  1. find the reviewed human UniProtKB entry (gene_exact match) whose canonical sequence is
     identical to the sequence AF3 was run on, so UniProt residue numbers == model residue numbers;
  2. keep UniProt features of type Domain, DNA binding and Zinc finger (curated structured units);
  3. add Pfam domains from InterPro as a cross-check.

Writes data/tf_domains.csv (one row per domain) and data/tf_domains_uniprot_raw.json (provenance).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEQS = json.load(open(ROOT / "data" / "tf_sequences_run.json"))
UNIPROT_TYPES = {"Domain", "DNA binding", "Zinc finger"}


def yy_get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def yy_uniprot_entry(gene, seq):
    q = urllib.parse.quote(f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true")
    hits = yy_get(f"https://rest.uniprot.org/uniprotkb/search?query={q}&format=json&size=10")["results"]
    for h in hits:
        if h["sequence"]["value"] == seq:
            return h
    raise ValueError(f"{gene}: no reviewed human entry with an identical canonical sequence "
                     f"(candidates: {[h['primaryAccession'] for h in hits]})")


rows, raw = [], {}
for gene, seq in sorted(SEQS.items()):
    entry = yy_uniprot_entry(gene, seq)
    acc = entry["primaryAccession"]
    raw[gene] = {"accession": acc, "entry_version": entry.get("entryAudit", {}).get("entryVersion"),
                 "sequence_version": entry.get("entryAudit", {}).get("sequenceVersion"), "features": []}
    for f in entry.get("features", []):
        if f["type"] in UNIPROT_TYPES:
            start, end = f["location"]["start"]["value"], f["location"]["end"]["value"]
            raw[gene]["features"].append(f)
            rows.append({"TF": gene, "accession": acc, "source": "UniProt", "type": f["type"],
                         "name": f.get("description", ""), "start": start, "end": end, "length": len(seq)})
    pf = yy_get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{acc}?page_size=100")
    for r in pf.get("results", []):
        for p in r["proteins"]:
            for loc in p["entry_protein_locations"]:
                for frag in loc["fragments"]:
                    rows.append({"TF": gene, "accession": acc, "source": "Pfam", "type": f"Pfam {r['metadata']['type']}",
                                 "name": f"{r['metadata']['accession']} {r['metadata']['name']}",
                                 "start": frag["start"], "end": frag["end"], "length": len(seq)})
    time.sleep(1)  # InterPro rate limit
    print(f"{gene:7s} {acc}  {sum(1 for x in rows if x['TF'] == gene and x['source'] == 'UniProt')} UniProt domains")

df = pd.DataFrame(rows).sort_values(["TF", "source", "start"]).reset_index(drop=True)
df.to_csv(ROOT / "data" / "tf_domains.csv", index=False)
json.dump(raw, open(ROOT / "data" / "tf_domains_uniprot_raw.json", "w"), indent=1)
print(f"\nwrote {len(df)} rows -> data/tf_domains.csv")


# ---------------------------------------------------------------------------------------------
# Structured-domain set used for domain-only BSA (one row per merged segment per TF).
# Rule: UniProt Domain / DNA binding / Zinc finger features
#       + Pfam entries of type domain, repeat or conserved_site
#       + Pfam families that are folded domains (allow-list below)
#       - regions annotated as intrinsically disordered regulatory segments (deny-list by name)
# Overlapping / adjacent (<= 5 residues apart) segments are merged.
PFAM_FOLDED_FAMILIES = {"PF02312",   # CBFB core-binding factor beta (whole folded protein)
                        "PF10401",   # IRF-3 IAD (IRF association domain)
                        "PF05920"}   # Homeobox KN (homeodomain)
DISORDERED = ("transactivation", "inhibition", "flanking", "CTNNB1", "transcription modulat", "pre-N-terminus")

keep = df[(df["source"] == "UniProt")
          | df["type"].isin(["Pfam domain", "Pfam repeat", "Pfam conserved_site"])
          | df["name"].str.split().str[0].isin(PFAM_FOLDED_FAMILIES)]
keep = keep[~keep["name"].str.contains("|".join(DISORDERED), case=False)]

segs = []
for tf, g in keep.sort_values("start").groupby("TF"):
    cur = None
    for _, r in g.iterrows():
        label = r["name"].split(" ", 1)[1] if r["source"] == "Pfam" else r["name"]
        if cur and r["start"] <= cur["end"] + 5:
            cur["end"] = max(cur["end"], r["end"])
            if label not in cur["labels"]:
                cur["labels"].append(label)
        else:
            cur and segs.append(cur)
            cur = {"TF": tf, "accession": r["accession"], "start": r["start"], "end": r["end"], "labels": [label]}
    segs.append(cur)
dom = pd.DataFrame(segs)
dom["domain"] = dom["labels"].apply(lambda x: " / ".join(x))
dom = dom.drop(columns="labels")[["TF", "accession", "domain", "start", "end"]]
dom.to_csv(ROOT / "data" / "tf_domains_structured.csv", index=False)
print(f"wrote {len(dom)} structured segments for {dom['TF'].nunique()} TFs -> data/tf_domains_structured.csv")
