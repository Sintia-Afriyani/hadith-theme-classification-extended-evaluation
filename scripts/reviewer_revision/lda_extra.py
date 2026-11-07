# -*- coding: utf-8 -*-
"""LDA pada 1.323 hadis (v1.1): K terpilih, istilah teratas, riwayat representatif, sebelum/sesudah boilerplate,
koleksi tertahan, pembanding (NMF, KMeans-LSA), kestabilan antar-benih, dan penyebab kesesuaian lemah."""
import sys, os, itertools, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, homogeneity_completeness_v_measure
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from common import SOURCE_STOP, REPO
RAW, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SEED = 42
d = pd.read_csv(f"{REPO}/dataset_rilis_tanpa_enam.csv").sort_values("url", kind="stable").reset_index(drop=True)
d.insert(0, "rec_id", [f"H{i+1:04d}" for i in range(len(d))])
raw = pd.read_csv(RAW, low_memory=False)[["url", "text_en", "english_full"]]
raw["raw_text"] = raw["text_en"].where(raw["text_en"].astype(str).str.strip().ne("") & raw["text_en"].notna(), raw["english_full"]).astype(str)
d = d.merge(raw[["url", "raw_text"]], on="url", how="left"); assert d.raw_text.notna().all() and len(d) == 1323
y = d.label.values; coll = d.collection.values
HADITH_STOP = {"allah", "prophet", "messenger", "narrated", "said", "saws", "pbuh", "ibn", "abu", "bin", "muhammad", "peace", "upon",
               "blessings", "sallallahu", "alayhi", "wasallam", "hadith", "reported", "told", "asked", "went", "came", "say", "says", "man",
               "men", "people", "did", "used", "saw", "heard", "let", "got", "took", "gave", "make", "made", "thing", "things", "will", "would",
               "shall", "one", "two", "also", "now", "come", "going", "go"}
STOP_AFTER = list(ENGLISH_STOP_WORDS | HADITH_STOP | SOURCE_STOP); STOP_BEFORE = list(ENGLISH_STOP_WORDS)
def tokens(texts, stop): return [CountVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True).build_analyzer()(t) for t in texts]
def cramers(a, b):
    ct = pd.crosstab(a, b); chi = stats.chi2_contingency(ct)[0]; return float(np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1))))
def metrics(lbl, cl):
    h, c, v = homogeneity_completeness_v_measure(lbl, cl)
    return dict(ARI=ARI(lbl, cl), NMI=NMI(lbl, cl), homogeneity=h, completeness=c, V_measure=v, Cramers_V=cramers(lbl, cl))
def fit_lda(toks, K, seed, passes=20, iters=200, no_below=5):
    dic = Dictionary(toks); dic.filter_extremes(no_below=no_below, no_above=0.5)
    corp = [dic.doc2bow(t) for t in toks]
    m = LdaModel(corpus=corp, id2word=dic, num_topics=K, random_state=seed, passes=passes, iterations=iters, alpha="auto", eta="auto")
    return m, dic, corp
def theta_of(m, corp, K):
    th = np.zeros((len(corp), K))
    for i, b in enumerate(corp):
        for k, p in m.get_document_topics(b, minimum_probability=0.0): th[i, k] = p
    return th
R = []
toks = tokens(d.text, STOP_AFTER); toksB = tokens(d.raw_text, STOP_BEFORE)
for K in [5]:
    m, dic_, corp_ = fit_lda(toks, K, SEED); th = theta_of(m, corp_, K); top = th.argmax(1)
    R.append(dict(analisis="A. LDA utama (boilerplate dibuang)", K=K, **metrics(y, top), NMI_dengan_koleksi=NMI(coll, top), prop_theta_maks_lt_05=float((th.max(1) < 0.5).mean())))
    pd.DataFrame([dict(topik=k, kata_teratas=", ".join(w for w, _ in m.show_topic(k, topn=12))) for k in range(K)]).to_csv(f"{OUT}/lda_topik_K{K}_v11.csv", index=False)
    pd.crosstab(y, top).to_csv(f"{OUT}/lda_silang_label_topik_K{K}_v11.csv"); pd.crosstab(coll, top).to_csv(f"{OUT}/lda_silang_koleksi_topik_K{K}_v11.csv")
    rep = [dict(topik=k, theta=round(float(th[i, k]), 3), label=y[i], collection=coll[i], rec_id=d.rec_id[i], url=d.url[i], teks=d.text[i][:400]) for k in range(K) for i in np.argsort(-th[:, k])[:5]]
    pd.DataFrame(rep).to_csv(f"{OUT}/lda_hadis_representatif_K{K}_v11.csv", index=False)
    mB, dB, cB = fit_lda(toksB, K, SEED); thB = theta_of(mB, cB, K); topB = thB.argmax(1)
    R.append(dict(analisis="C. Sebelum (teks mentah, stopword Inggris saja)", K=K, **metrics(y, topB), NMI_dengan_koleksi=NMI(coll, topB), prop_theta_maks_lt_05=float((thB.max(1) < 0.5).mean()), ARI_sebelum_vs_sesudah=ARI(topB, top)))
    pd.DataFrame([dict(topik=k, kata_teratas=", ".join(w for w, _ in mB.show_topic(k, topn=12))) for k in range(K)]).to_csv(f"{OUT}/lda_topik_SEBELUM_boilerplate_K{K}.csv", index=False)
    pd.crosstab(y, topB).to_csv(f"{OUT}/lda_silang_label_topik_SEBELUM_K{K}.csv")
    nm = NMF(K, init="nndsvda", random_state=SEED, max_iter=600)
    tv = TfidfVectorizer(stop_words=STOP_AFTER, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", min_df=5, max_df=0.5, sublinear_tf=True); Xt = tv.fit_transform(d.text); W = nm.fit_transform(Xt)
    R.append(dict(analisis="E. NMF (TF-IDF)", K=K, **metrics(y, W.argmax(1)), NMI_dengan_koleksi=NMI(coll, W.argmax(1))))
    # tertahan K=5
    for c in sorted(set(coll)):
        te = coll == c; tr = ~te
        m2, d2, c2 = fit_lda([toks[i] for i in np.where(tr)[0]], K, SEED); top_tr = theta_of(m2, c2, K).argmax(1)
        mp = pd.Series(y[tr]).groupby(top_tr).agg(lambda s: s.value_counts().index[0]).to_dict()
        top_te = theta_of(m2, [d2.doc2bow(toks[i]) for i in np.where(te)[0]], K).argmax(1)
        R.append(dict(analisis=f"D. Koleksi tertahan: {c}", K=K, n=int(te.sum()), akurasi_topik_ke_label=float((np.array([mp.get(t, "?") for t in top_te]) == y[te]).mean()), ARI=ARI(y[te], top_te), NMI=NMI(y[te], top_te)))
cv = CountVectorizer(stop_words=STOP_AFTER, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", min_df=5, max_df=0.5); Xc = cv.fit_transform(d.text); vocab = np.array(cv.get_feature_names_out())
topf = {c: set(vocab[np.argsort(-np.asarray(Xc[y == c].sum(0)).ravel())[:50]]) for c in sorted(set(y))}
jac = pd.DataFrame({a: {b: len(topf[a] & topf[b]) / len(topf[a] | topf[b]) for b in topf} for a in topf}); jac.to_csv(f"{OUT}/jaccard_50_kata_tersering_antarkelas.csv"); print(jac.round(2).to_string())
for a, b in [("Makanan", "Pengobatan"), ("Bersuci", "Pengobatan"), ("Makanan", "Bersuci")]: print(a, b, sorted(topf[a] & topf[b]))
pd.DataFrame(R).to_csv(f"{OUT}/lda_ringkasan_K5_v11.csv", index=False); print(pd.DataFrame(R).round(3).to_string())
