# -*- coding: utf-8 -*-
"""LOCO per kelas: (A) latih semua kelas, (B) latih hanya kelas yang ada di koleksi uji, (C) pembanding dalam-koleksi (OOF nested CV)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
d = load(); X, y, g, coll = d.text.values, d.label.values, d.group.values, d.collection.values
oof = pd.read_csv(f"{OUT}/oof_all.csv"); T = "Ensemble (tuned, nested CV)"
oo = oof[oof.model == T].merge(d[["rec_id", "collection"]], on="rec_id"); oo["ok"] = oo.true == oo.pred
classes = sorted(set(y)); rows, perclass = [], []
ctab = pd.crosstab(d.collection, d.label); ctab.to_csv(f"{OUT}/dukungan_kelas_per_koleksi.csv")
for c in sorted(set(coll)):
    te = coll == c; tr = (~te) & (~pd.Series(g).isin(set(g[te])).values)
    present = sorted(set(y[te]))
    for mode, trm in [("A_semua_kelas_latih", tr), ("B_kelas_sama_dgn_uji", tr & np.isin(y, present))]:
        m = pipe(ens(1.0, 100, 0.1)).fit(X[trm], y[trm]); p = m.predict(X[te])
        rows.append(dict(koleksi_uji=c, mode=mode, n_uji=int(te.sum()), n_latih=int(trm.sum()), kelas_ada_di_uji=len(present),
                         salah=int((p != y[te]).sum()), akurasi=accuracy_score(y[te], p),
                         macro_f1=f1_score(y[te], p, labels=present, average="macro", zero_division=0)))
        pr, rc, f1, sup = precision_recall_fscore_support(y[te], p, labels=classes, zero_division=0)
        for k, cl in enumerate(classes):
            if sup[k] > 0 or (p == cl).any():
                perclass.append(dict(koleksi_uji=c, mode=mode, kelas=cl, support=int(sup[k]), diprediksi=int((p == cl).sum()),
                                     presisi=pr[k], recall=rc[k], f1=f1[k],
                                     salah_dari_kelas_ini=int(((y[te] == cl) & (p != cl)).sum())))
    s = oo[oo.collection == c].groupby("repeat").ok.mean()
    rows.append(dict(koleksi_uji=c, mode="C_dalam_koleksi_nestedCV_OOF", n_uji=int(te.sum()), n_latih=np.nan, kelas_ada_di_uji=len(present),
                     salah=float((1 - s).mean() * te.sum()), akurasi=s.mean(), macro_f1=np.nan))
    # tebakan kelas terbanyak di koleksi uji (batas bawah)
    maj = pd.Series(y[tr]).value_counts().index[0]
    rows.append(dict(koleksi_uji=c, mode="D_tebak_kelas_terbanyak_latih", n_uji=int(te.sum()), n_latih=int(tr.sum()), kelas_ada_di_uji=len(present),
                     salah=int((y[te] != maj).sum()), akurasi=float((y[te] == maj).mean()), macro_f1=np.nan))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/loco_ringkasan.csv", index=False)
pd.DataFrame(perclass).to_csv(f"{OUT}/loco_per_kelas.csv", index=False)
print(ctab.to_string()); print(R.round(3).to_string(index=False))
print(pd.DataFrame(perclass).round(3).to_string(index=False))
