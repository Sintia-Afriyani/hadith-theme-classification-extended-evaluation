# -*- coding: utf-8 -*-
"""Diagnostik pada 1.323 hadis (v1.1): kata penciri chi-kuadrat, ablasi, kontrol label diacak, baseline koleksi-saja."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
OUT = sys.argv[1]; d = load(); X, y, g = d.text.values, d.label.values, d.group.values
def cv_fixed(ya, stop=None, reps=3):
    a, f = [], []
    for rep in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(X, ya, g):
            p = pipe(ens(1.0, 100, 0.1), stop).fit(X[tr], ya[tr]).predict(X[te]); a.append(accuracy_score(ya[te], p)); f.append(f1_score(ya[te], p, average="macro"))
    return np.mean(a), np.std(a, ddof=1), np.mean(f), np.std(f, ddof=1)
cvz = CountVectorizer(stop_words=STOP_MAIN, min_df=3, binary=True); Z = cvz.fit_transform(X); vocab = np.array(cvz.get_feature_names_out())
lb = LabelBinarizer().fit(y); Y = lb.transform(y); top = {}
for ci, cls in enumerate(lb.classes_):
    sc, _ = chi2(Z, Y[:, ci]); top[cls] = list(vocab[np.argsort(-np.nan_to_num(sc))[:50]])
pd.DataFrame(top).to_csv(f"{OUT}/kata_penciri_top50_v11.csv", index=False)
abl = []
for k in [0, 10, 25, 50]:
    rm = set(w for c in top for w in top[c][:k]); a, asd, f, fsd = cv_fixed(y, sorted(STOP_MAIN_SET | rm))
    abl.append(dict(kata_dibuang_per_kelas=k, total_kata_dibuang=len(rm), akurasi=a, akurasi_sd=asd, macro_f1=f, macro_f1_sd=fsd)); print(abl[-1], flush=True)
pd.DataFrame(abl).to_csv(f"{OUT}/ablasi_kata_v11.csv", index=False)
sh = []
for s in range(5):
    a, asd, f, fsd = cv_fixed(np.random.RandomState(SEED + s).permutation(y), reps=1); sh.append((a, f))
sh = np.array(sh); print("label diacak (5 permutasi): akurasi", sh[:, 0].mean().round(3), "±", sh[:, 0].std(ddof=1).round(3), "| macro-F1", sh[:, 1].mean().round(3))
pd.DataFrame([dict(kontrol="label diacak, 5 permutasi", akurasi=sh[:, 0].mean(), akurasi_sd=sh[:, 0].std(ddof=1), macro_f1=sh[:, 1].mean())]).to_csv(f"{OUT}/kontrol_label_acak_v11.csv", index=False)
maj = d.groupby("collection")["label"].agg(lambda s: s.value_counts().index[0]); print("baseline koleksi-saja:", round(float((d.collection.map(maj) == d.label).mean()), 3), "| kelas terbanyak global:", round(float(pd.Series(y).value_counts(normalize=True).iloc[0]), 3))
