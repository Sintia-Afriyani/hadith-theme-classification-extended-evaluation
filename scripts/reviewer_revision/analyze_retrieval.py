# -*- coding: utf-8 -*-
"""Evaluasi pencarian dari penilaian pakar. Pemakaian: python analyze_retrieval.py Pakar1.xlsx [Pakar2.xlsx]"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
files = sys.argv[1:]; key = pd.read_csv("pencarian_kunci_kolam.csv"); Q = pd.read_csv("kueri_daftar.csv"); SYS = ["BM25", "BM25 + tema lunak", "BM25 + saring tema keras"]
def read(f):
    w = pd.read_excel(f, sheet_name="Penilaian", header=2); w = w[["No", "Relevansi"]].dropna(subset=["No"])
    w["g"] = w.Relevansi.astype(str).str.extract(r"^\s*([012])")[0].astype(float); return w.set_index("No").g
J = pd.concat([read(f).rename(f"j{i+1}") for i, f in enumerate(files)], axis=1)
key = key.set_index("No").join(J); key["g"] = J.mean(axis=1).reindex(key.index)
if len(files) == 2:
    both = J.dropna(); print(f"butir dinilai ganda: {len(both)}")
    print("kappa berbobot linear (3 tingkat):", round(cohen_kappa_score(both.j1.astype(int), both.j2.astype(int), weights="linear"), 3),
          "| kappa biner (relevan>=1):", round(cohen_kappa_score((both.j1 >= 1).astype(int), (both.j2 >= 1).astype(int)), 3),
          "| persen sama persis:", round(float((both.j1 == both.j2).mean()), 3))
    dis = key.loc[both.index][(both.j1 >= 1) != (both.j2 >= 1)]; dis.to_csv("pencarian_perlu_adjudikasi.csv"); print("perbedaan keputusan relevan/tidak:", len(dis), "(lihat pencarian_perlu_adjudikasi.csv)")
key = key.dropna(subset=["g"]); rows = []
for q in Q.qid:
    for s in SYS:
        r = key[(key.qid == q) & key[s].notna()].sort_values(s).head(5)
        if len(r) == 0: continue
        gain = r.g.values; dcg = (gain / np.log2(np.arange(2, len(gain) + 2))).sum()
        allg = np.sort(key[key.qid == q].g.values)[::-1][:5]; idcg = (allg / np.log2(np.arange(2, len(allg) + 2))).sum()
        rel = (r.g >= 1)
        rows.append(dict(qid=q, jenis=Q.set_index("qid").loc[q, "jenis"], sistem=s, n_5teratas=len(r), P5_relevan_atau_sebagian=rel.mean(), P5_hanya_relevan=(r.g == 2).mean(),
                         nDCG5=dcg / idcg if idcg > 0 else np.nan, label_berbeda_pada_relevan=r[rel].label.nunique()))
E = pd.DataFrame(rows); E.to_csv("pencarian_hasil_per_kueri.csv", index=False)
pd.set_option("display.width", 200)
print(E.groupby("sistem")[["P5_relevan_atau_sebagian", "P5_hanya_relevan", "nDCG5", "label_berbeda_pada_relevan"]].mean().round(3))
print(E.groupby(["jenis", "sistem"])[["P5_relevan_atau_sebagian", "nDCG5", "label_berbeda_pada_relevan"]].mean().round(3))
rng = np.random.RandomState(0)
for s in SYS:
    v = E[E.sistem == s].P5_relevan_atau_sebagian.values; bs = [rng.choice(v, len(v)).mean() for _ in range(5000)]
    print(s, "P@5 rata-rata", round(v.mean(), 3), "CI95 bootstrap antar-kueri", [round(x, 3) for x in np.percentile(bs, [2.5, 97.5])])
