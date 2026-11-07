# -*- coding: utf-8 -*-
"""Dampak undersampling Bersuci: latih seimbang vs berbobot-kelas vs tanpa bobot; uji pada prevalensi alami dan seimbang."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support
OUT = sys.argv[1]
pool = pd.read_csv(f"{OUT}/kolam_penuh_tanpa_enam.csv")
pool["group"] = near_dup_groups(pool["text"], NEARDUP_TH)
X, y, g = pool.text.values, pool.label.values, pool.group.values
classes = ["Bersuci", "Fenomena Alam", "Makanan", "Pengobatan"]

def model(weighted):
    cw = "balanced" if weighted else None
    return Pipeline([("tfidf", tfidf()), ("clf", VotingClassifier([
        ("svm", SVC(C=1.0, kernel="linear", probability=True, class_weight=cw, random_state=SEED)),
        ("rf", RandomForestClassifier(100, class_weight=cw, random_state=SEED, n_jobs=1)),
        ("nb", MultinomialNB(alpha=0.1, fit_prior=not weighted))], voting="soft"))])

def under(idx, rng):
    lab = y[idx]; cap = max((lab == c).sum() for c in classes if c != "Bersuci")
    b = idx[lab == "Bersuci"]; keep = rng.choice(b, min(cap, len(b)), replace=False)
    return np.concatenate([idx[lab != "Bersuci"], keep])

rows, pcrows = [], []
for rep in range(3):
    rng = np.random.RandomState(SEED + rep)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(X, y, g)):
        te_bal = under(te, rng)       # uji "seimbang" gaya paper
        for name, trn, w in [("A. Undersampling Bersuci (seperti paper)", under(tr, rng), False),
                             ("B. Data penuh, berbobot kelas", tr, True),
                             ("C. Data penuh, tanpa bobot", tr, False)]:
            m = model(w).fit(X[trn], y[trn])
            for tname, tidx in [("uji prevalensi alami", te), ("uji seimbang (Bersuci dikurangi)", te_bal)]:
                p = m.predict(X[tidx]); yt = y[tidx]
                rows.append(dict(strategi=name, uji=tname, repeat=rep, fold=fold, n_uji=len(tidx), akurasi=accuracy_score(yt, p),
                                 macro_f1=f1_score(yt, p, average="macro"), balanced_acc=balanced_accuracy_score(yt, p)))
                pr, rc, f1, sp = precision_recall_fscore_support(yt, p, labels=classes, zero_division=0)
                for k, c in enumerate(classes):
                    pcrows.append(dict(strategi=name, uji=tname, repeat=rep, fold=fold, kelas=c, presisi=pr[k], recall=rc[k], f1=f1[k], support=int(sp[k]), diprediksi=int((p == c).sum())))
    print("ulangan", rep + 1, "selesai", flush=True)
R = pd.DataFrame(rows); PC = pd.DataFrame(pcrows)
S = R.groupby(["strategi", "uji"])[["akurasi", "macro_f1", "balanced_acc"]].agg(["mean", "std"]).round(4)
S.to_csv(f"{OUT}/undersampling_ringkasan.csv"); print(S.to_string())
# presisi/recall per kelas dihitung dari total TP/FP/FN seluruh lipatan agar stabil
def agg(d):
    tp = (d.recall * d.support).sum(); sup = d.support.sum(); pred = d.diprediksi.sum()
    return pd.Series(dict(support=sup, diprediksi=pred, presisi=tp / pred if pred else np.nan, recall=tp / sup))
PCs = PC.groupby(["strategi", "uji", "kelas"]).apply(agg).round(3); PCs.to_csv(f"{OUT}/undersampling_per_kelas.csv"); print(PCs.to_string())
# tambahan: latih pada 1.323 (seimbang), uji pada hadis Bersuci yang dibuang (tanpa kelompok yang tumpang tindih dgn data latih)
fin = pool.in_final.values; tr = np.where(fin)[0]; dropped = np.where(~fin)[0]
ok = ~pd.Series(g[dropped]).isin(set(g[tr])).values; dropped = dropped[ok]
m = model(False).fit(X[tr], y[tr]); p = m.predict(X[dropped])
h = pd.Series(p).value_counts().to_dict()
print("\nBersuci yang dibuang & tidak tumpang tindih kelompok:", len(dropped), "| recall Bersuci:", round(float((p == "Bersuci").mean()), 4), h)
pd.DataFrame([dict(n_uji=len(dropped), recall_bersuci=float((p == "Bersuci").mean()), **{f"diprediksi_{k}": v for k, v in h.items()})]).to_csv(f"{OUT}/holdout_bersuci_dibuang.csv", index=False)
