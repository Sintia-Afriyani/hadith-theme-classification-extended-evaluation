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
# ---- A. pilih K dengan koherensi c_v (protokol asli), 1.323 hadis, boilerplate dibuang
toks = tokens(d.text, STOP_AFTER)
dic = Dictionary(toks); dic.filter_extremes(no_below=5, no_above=0.5); corp = [dic.doc2bow(t) for t in toks]
tf = [[w for w in t if w in dic.token2id] for t in toks]
rows = []
for K in range(2, 11):
    cv, pp = [], []
    for sd in [1, 2, 3]:
        m = LdaModel(corpus=corp, id2word=dic, num_topics=K, random_state=sd, passes=10, iterations=100, alpha="auto", eta="auto")
        cv.append(CoherenceModel(model=m, texts=tf, dictionary=dic, coherence="c_v", processes=1).get_coherence()); pp.append(m.log_perplexity(corp))
    rows.append(dict(K=K, c_v_mean=np.mean(cv), c_v_sd=np.std(cv), log_perplexity_per_word=np.mean(pp))); print(rows[-1], flush=True)
coh = pd.DataFrame(rows); coh.to_csv(f"{OUT}/lda_koherensi_v11.csv", index=False)
K_BEST = int(coh.loc[coh.c_v_mean.idxmax(), "K"]); print("K terpilih:", K_BEST, flush=True)
# ---- B. model utama: istilah teratas, representatif, tabel silang, metrik
main = {}
for K in sorted({K_BEST, 4}):
    m, dic_, corp_ = fit_lda(toks, K, SEED); th = theta_of(m, corp_, K); top = th.argmax(1); main[K] = (m, th, top)
    met = metrics(y, top); R.append(dict(analisis="A. LDA utama (boilerplate dibuang)", K=K, **met, NMI_dengan_koleksi=NMI(coll, top)))
    pd.DataFrame([dict(topik=k, kata_teratas=", ".join(w for w, _ in m.show_topic(k, topn=12))) for k in range(K)]).to_csv(f"{OUT}/lda_topik_K{K}_v11.csv", index=False)
    pd.crosstab(y, top).to_csv(f"{OUT}/lda_silang_label_topik_K{K}_v11.csv")
    pd.crosstab(coll, top).to_csv(f"{OUT}/lda_silang_koleksi_topik_K{K}_v11.csv")
    pd.DataFrame(th, columns=[f"topik_{k}" for k in range(K)]).groupby(y).mean().to_csv(f"{OUT}/lda_P_topik_per_kelas_K{K}_v11.csv")
    rep = [dict(topik=k, theta=round(float(th[i, k]), 3), label=y[i], collection=coll[i], rec_id=d.rec_id[i], url=d.url[i], teks=d.text[i][:400])
           for k in range(K) for i in np.argsort(-th[:, k])[:5]]
    pd.DataFrame(rep).to_csv(f"{OUT}/lda_hadis_representatif_K{K}_v11.csv", index=False)
    print(f"K={K}", {k: round(v, 3) for k, v in met.items()}, "| max theta < 0.5:", round(float((th.max(1) < 0.5).mean()), 3), flush=True)
    R[-1]["prop_theta_maks_lt_05"] = float((th.max(1) < 0.5).mean())
# ---- C. sebelum vs sesudah membuang boilerplate (teks mentah + stopword Inggris saja, vs teks bersih + stopword hadis/sumber)
toksB = tokens(d.raw_text, STOP_BEFORE)
for K in sorted({K_BEST, 4}):
    mB, dB, cB = fit_lda(toksB, K, SEED); thB = theta_of(mB, cB, K); topB = thB.argmax(1)
    R.append(dict(analisis="C. Sebelum (teks mentah, stopword Inggris saja)", K=K, **metrics(y, topB), NMI_dengan_koleksi=NMI(coll, topB),
                  prop_theta_maks_lt_05=float((thB.max(1) < 0.5).mean()), ARI_sebelum_vs_sesudah=ARI(topB, main[K][2])))
    pd.DataFrame([dict(topik=k, kata_teratas=", ".join(w for w, _ in mB.show_topic(k, topn=12))) for k in range(K)]).to_csv(f"{OUT}/lda_topik_SEBELUM_boilerplate_K{K}.csv", index=False)
    pd.crosstab(y, topB).to_csv(f"{OUT}/lda_silang_label_topik_SEBELUM_K{K}.csv")
    print("sebelum K=%d" % K, {k: round(v, 3) for k, v in R[-1].items() if isinstance(v, float)}, flush=True)
# ---- D. koleksi tertahan: latih pada 5 koleksi, infer topik pada koleksi tertahan, petakan topik ke label mayoritas data latih
K = K_BEST
for c in sorted(set(coll)):
    te = coll == c; tr = ~te
    m, dic_, corp_ = fit_lda([toks[i] for i in np.where(tr)[0]], K, SEED)
    th_tr = theta_of(m, corp_, K); top_tr = th_tr.argmax(1)
    mp = pd.Series(y[tr]).groupby(top_tr).agg(lambda s: s.value_counts().index[0]).to_dict()
    corp_te = [dic_.doc2bow(toks[i]) for i in np.where(te)[0]]; th_te = theta_of(m, corp_te, K); top_te = th_te.argmax(1)
    pred = np.array([mp.get(t, "?") for t in top_te]); acc = float((pred == y[te]).mean())
    met = {k: v for k, v in metrics(y[te], top_te).items() if k in ("ARI", "NMI", "V_measure")}
    R.append(dict(analisis=f"D. Koleksi tertahan: {c}", K=K, n=int(te.sum()), akurasi_topik_ke_label=acc, **met))
    print("tertahan", c, round(acc, 3), {k: round(v, 3) for k, v in met.items()}, flush=True)
# ---- E. pembanding: NMF (TF-IDF) dan KMeans pada LSA
tv = TfidfVectorizer(stop_words=STOP_AFTER, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", min_df=5, max_df=0.5, sublinear_tf=True); Xt = tv.fit_transform(d.text)
terms = np.array(tv.get_feature_names_out())
for K in sorted({K_BEST, 4}):
    nm = NMF(K, init="nndsvda", random_state=SEED, max_iter=600).fit(Xt); W = nm.transform(Xt); top = W.argmax(1)
    R.append(dict(analisis="E. NMF (TF-IDF)", K=K, **metrics(y, top), NMI_dengan_koleksi=NMI(coll, top)))
    pd.DataFrame([dict(topik=k, kata_teratas=", ".join(terms[np.argsort(-nm.components_[k])[:12]])) for k in range(K)]).to_csv(f"{OUT}/nmf_topik_K{K}.csv", index=False)
    Z = TruncatedSVD(50, random_state=SEED).fit_transform(Xt); Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    km = KMeans(K, n_init=20, random_state=SEED).fit_predict(Z)
    R.append(dict(analisis="E. KMeans pada LSA-50 (cosine)", K=K, **metrics(y, km), NMI_dengan_koleksi=NMI(coll, km)))
# ---- F. kestabilan antar-benih (LDA utama, K terpilih)
runs = []
for sd in range(1, 6):
    m, dic_, corp_ = fit_lda(toks, K_BEST, sd); runs.append(theta_of(m, corp_, K_BEST).argmax(1))
aris = [ARI(a, b) for a, b in itertools.combinations(runs, 2)]; vs = [ARI(y, r) for r in runs]
R.append(dict(analisis="F. Kestabilan 5 benih", K=K_BEST, ARI=float(np.mean(vs)), ARI_sd=float(np.std(vs, ddof=1)), ARI_antar_benih_mean=float(np.mean(aris)), ARI_antar_benih_min=float(np.min(aris))))
# ---- G. sebab kesesuaian lemah: tumpang tindih kosakata antarkelas dan komposisi topik per kelas
cv = CountVectorizer(stop_words=STOP_AFTER, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", min_df=5, max_df=0.5); Xc = cv.fit_transform(d.text); vocab = np.array(cv.get_feature_names_out())
top50 = {}
for c in sorted(set(y)):
    cnt = np.asarray(Xc[y == c].sum(0)).ravel(); rest = np.asarray(Xc[y != c].sum(0)).ravel()
    sc = (cnt / cnt.sum()) * np.log((cnt / cnt.sum() + 1e-9) / (rest / rest.sum() + 1e-9)); top50[c] = set(vocab[np.argsort(-sc)[:50]])
jac = pd.DataFrame({a: {b: len(top50[a] & top50[b]) / len(top50[a] | top50[b]) for b in top50} for a in top50}); jac.to_csv(f"{OUT}/jaccard_kata_penciri_antarkelas.csv"); print(jac.round(2).to_string())
pd.DataFrame({c: sorted(top50[c]) for c in top50 if True}.items(), columns=["kelas", "kata_penciri_top50"]).assign(kata_penciri_top50=lambda x: x.kata_penciri_top50.map(", ".join)).to_csv(f"{OUT}/kata_penciri_per_kelas_top50.csv", index=False)
pd.DataFrame(R).to_csv(f"{OUT}/lda_ringkasan_v11.csv", index=False)
print(pd.DataFrame(R).round(3).to_string(), flush=True); print("SELESAI", flush=True)
