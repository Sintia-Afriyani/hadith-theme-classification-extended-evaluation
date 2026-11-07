# -*- coding: utf-8 -*-
"""Tugas pencarian kecil: BM25, BM25 + keselarasan tema (lunak), BM25 dengan penyaring tema keras. Kolam penilaian untuk pakar."""
import sys, os
sys.path.insert(0, "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad/m5")
from common import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold
SP = "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad"
d = load(); X, y = d.text.values, d.label.values; classes = sorted(set(y))
# urutan: 4 pertama dinilai ganda oleh kedua pakar
Q = [
 ("Q1", "performing ablution (wudu) with water", "Tata cara wudu dengan air", "Bersuci", "tunggal"),
 ("Q2", "honey as a cure for illness", "Madu sebagai obat penyakit", "Pengobatan+Makanan", "lintas"),
 ("Q3", "prayer during a solar eclipse", "Salat ketika gerhana matahari", "Fenomena Alam", "tunggal"),
 ("Q4", "washing hands before and after eating", "Mencuci tangan sebelum dan sesudah makan", "Bersuci+Makanan", "lintas"),
 ("Q5", "cupping as treatment", "Bekam sebagai pengobatan", "Pengobatan", "tunggal"),
 ("Q6", "eating dates and drinking milk", "Makan kurma dan minum susu", "Makanan", "tunggal"),
 ("Q7", "reciting ruqyah and using water to treat the sick", "Ruqyah dan air untuk mengobati orang sakit", "Pengobatan+Bersuci", "lintas"),
 ("Q8", "menstruating woman eating or drinking", "Perempuan haid yang makan atau minum", "Bersuci+Makanan", "lintas"),
]
# OOF probabilitas dokumen (lipatan ulangan 0), dan model penuh untuk kueri
fa = pd.read_csv(f"{SP}/m5/out/fold_assignments.csv")["rep0"].values
P_doc = np.zeros((len(d), 4))
for f in range(5):
    tr, te = np.where(fa != f)[0], np.where(fa == f)[0]
    P_doc[te] = pipe(ens(1.0, 100, 0.1)).fit(X[tr], y[tr]).predict_proba(X[te])
full = pipe(ens(1.0, 100, 0.1)).fit(X, y); assert list(full.classes_) == classes
cv = CountVectorizer(stop_words=STOP_MAIN, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"); M = cv.fit_transform(X).tocsc()
N = M.shape[0]; dl = np.asarray(M.sum(1)).ravel(); avg = dl.mean(); df_ = np.asarray((M > 0).sum(0)).ravel()
idf = np.log(1 + (N - df_ + .5) / (df_ + .5)); k1, b = 1.5, .75
def bm25(q):
    v = cv.transform([q]); score = np.zeros(N)
    for j in v.indices:
        col = M[:, j].toarray().ravel(); score += idf[j] * col * (k1 + 1) / (col + k1 * (1 - b + b * dl / avg))
    return score
rows, TOPK = [], 5
for qid, qen, qid_, theme, typ in Q:
    s = bm25(qen); pq = full.predict_proba([qen])[0]; A = P_doc @ pq
    sn = s / s.max() if s.max() > 0 else s
    hyb = sn * A
    top1 = int(np.argmax(pq)); cand = np.where(P_doc.argmax(1) == top1)[0]
    hard = np.full(N, -1.0); hard[cand] = s[cand]
    systems = {"BM25": s, "BM25 + tema lunak": hyb, "BM25 + saring tema keras": hard}
    print(f"\n{qid} [{typ}] {qen} | tema kueri diprediksi: " + ", ".join(f"{c}={p:.2f}" for c, p in zip(classes, pq)))
    for sysn, sc in systems.items():
        top = np.argsort(-sc, kind="stable")[:TOPK]
        print("  ", sysn.ljust(26), [d.rec_id[i] + ":" + y[i][:3] for i in top])
        for r, i in enumerate(top, 1): rows.append(dict(qid=qid, sistem=sysn, rank=r, rec_id=d.rec_id[i], label=y[i], prediksi=classes[int(P_doc[i].argmax())], skor=float(sc[i])))
R = pd.DataFrame(rows); R.to_csv("kueri_hasil_per_sistem.csv", index=False)
pd.DataFrame(Q, columns=["qid", "kueri_en", "kueri_id", "tema_kueri", "jenis"]).to_csv("kueri_daftar.csv", index=False)
print("\nKomposisi label di 5 teratas per sistem (jumlah hadis):")
print(R.groupby(["sistem", "label"]).size().unstack(fill_value=0))
for q in Q:
    pass
pool = R.groupby(["qid", "rec_id"]).agg(label=("label", "first")).reset_index()
print("\nukuran kolam per kueri:", pool.groupby("qid").size().to_dict(), "| total", len(pool))
for a, b_ in [("BM25", "BM25 + tema lunak"), ("BM25", "BM25 + saring tema keras")]:
    ov = [len(set(R[(R.qid == q[0]) & (R.sistem == a)].rec_id) & set(R[(R.qid == q[0]) & (R.sistem == b_)].rec_id)) for q in Q]
    print(f"irisan top-5 {a} vs {b_}:", ov)
