# -*- coding: utf-8 -*-
"""Sensitivitas ambang near-duplicate: jumlah kelompok & kinerja CV (ensemble default, 3 ulangan)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
d = load(); X, y = d.text.values, d.label.values; rows = []
for th in [None, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
    if th is None: g = np.arange(len(d))
    else: g = near_dup_groups(d.text, th)
    gs = pd.Series(g).value_counts(); mixed = int((d.groupby(g).label.nunique() > 1).sum())
    a, f = [], []
    for rep in range(3):
        sk = (StratifiedKFold(5, shuffle=True, random_state=SEED+rep).split(X, y) if th is None
              else StratifiedGroupKFold(5, shuffle=True, random_state=SEED+rep).split(X, y, g))
        for tr, te in sk:
            p = pipe(ens(1.0, 100, 0.1)).fit(X[tr], y[tr]).predict(X[te])
            a.append(accuracy_score(y[te], p)); f.append(f1_score(y[te], p, average="macro"))
    rows.append(dict(ambang=("tanpa pengelompokan (acak)" if th is None else th), n_kelompok=len(gs), kelompok_gt1=int((gs > 1).sum()),
                     hadis_dalam_kelompok_gt1=int(gs[gs > 1].sum()), kelompok_berlabel_campur=mixed,
                     akurasi=np.mean(a), akurasi_sd=np.std(a, ddof=1), macro_f1=np.mean(f), macro_f1_sd=np.std(f, ddof=1)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/sensitivitas_ambang_neardup.csv", index=False)
