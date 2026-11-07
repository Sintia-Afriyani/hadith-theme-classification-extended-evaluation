# -*- coding: utf-8 -*-
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
cv = pd.read_csv(f"{OUT}/cv_per_fold.csv"); oof = pd.read_csv(f"{OUT}/oof_all.csv")
ids = pd.read_csv(f"{OUT}/dataset_ids_v1.1.csv")
R = cv["repeat"].nunique(); order = list(cv.model.unique()); classes = sorted(ids.label.unique())
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# 1. ringkasan: rata-rata dan SD antar-fold (R*5 fold), plus SD antar-ulangan
rows = []
for m in order:
    d = cv[cv.model == m]; rep = d.groupby("repeat")[["accuracy", "macro_f1", "balanced_acc"]].mean()
    h = lambda v: stats.t.ppf(.975, len(v)-1) * v.std(ddof=1) / np.sqrt(len(v))
    rows.append(dict(model=m, n_fold=len(d), akurasi=d.accuracy.mean(), akurasi_sd_fold=d.accuracy.std(ddof=1),
                     macro_f1=d.macro_f1.mean(), macro_f1_sd_fold=d.macro_f1.std(ddof=1),
                     macro_f1_sd_antar_ulangan=rep.macro_f1.std(ddof=1), bal_acc=d.balanced_acc.mean(),
                     waktu_latih_detik_per_fold=d.fit_s.mean(),
                     akurasi_ci95_lo=d.accuracy.mean()-h(rep.accuracy), akurasi_ci95_hi=d.accuracy.mean()+h(rep.accuracy)))
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/ringkasan_model.csv", index=False)
P("== RINGKASAN (R=%d ulangan x 5 fold) ==" % R); P(S.round(4).to_string(index=False))

# 2. jumlah kesalahan per model per ulangan (satu angka per ulangan, dari seluruh 1.323 prediksi out-of-fold)
oof["salah"] = oof.true != oof.pred
E = oof.groupby(["model", "repeat"]).salah.sum().unstack()
E.to_csv(f"{OUT}/jumlah_kesalahan_per_ulangan.csv")
P("\n== KESALAHAN per model (baris) per ulangan (kolom) =="); P(E.to_string())
P("rata-rata kesalahan ensemble tuned: %.1f (SD %.1f), rentang %d-%d dari %d" % (
    E.loc['Ensemble (tuned, nested CV)'].mean(), E.loc['Ensemble (tuned, nested CV)'].std(ddof=1),
    E.loc['Ensemble (tuned, nested CV)'].min(), E.loc['Ensemble (tuned, nested CV)'].max(), len(ids)))

# 3. matriks konfusi + metrik per kelas (ensemble tuned): ulangan 0 dan rata-rata ulangan
T = "Ensemble (tuned, nested CV)"
for rep in [0]:
    o = oof[(oof.model == T) & (oof["repeat"] == rep)]
    cm = pd.DataFrame(confusion_matrix(o.true, o.pred, labels=classes), index=classes, columns=classes)
    cm.to_csv(f"{OUT}/confusion_matrix_rep{rep}.csv")
    P(f"\n== MATRIKS KONFUSI ulangan {rep} (kesalahan = {int(cm.values.sum()-np.trace(cm.values))}, akurasi {np.trace(cm.values)/cm.values.sum():.4f}) =="); P(cm.to_string())
    pg = o[o.salah].groupby(["true", "pred"]).size().sort_values(ascending=False).rename("n").reset_index()
    pg.to_csv(f"{OUT}/pasangan_galat_rep{rep}.csv", index=False)
pcs = []
for rep in range(R):
    o = oof[(oof.model == T) & (oof["repeat"] == rep)]
    p, r, f, s = precision_recall_fscore_support(o.true, o.pred, labels=classes, zero_division=0)
    pcs.append(pd.DataFrame(dict(kelas=classes, presisi=p, recall=r, f1=f, n=s, repeat=rep)))
pc = pd.concat(pcs).groupby("kelas")[["presisi", "recall", "f1", "n"]].agg(["mean", "std"]); pc.to_csv(f"{OUT}/metrik_per_kelas_rata2.csv")
P("\n== METRIK PER KELAS (rata-rata & SD antar-ulangan) =="); P(pc.round(3).to_string())

# 4. uji statistik berpasangan: ensemble tuned vs komponen (50 fold; uji-t terkoreksi Nadeau-Bengio, Wilcoxon, McNemar pada ulangan 0)
P("\n== UJI BERPASANGAN macro-F1: %s vs pembanding ==" % T)
base = cv[cv.model == T].sort_values(["repeat", "fold"]).macro_f1.values
res = []
for m in order:
    if m == T: continue
    other = cv[cv.model == m].sort_values(["repeat", "fold"]).macro_f1.values
    dif = base - other; n = len(dif)
    var = dif.var(ddof=1); tc = dif.mean() / np.sqrt((1/n + 0.25) * var) if var > 0 else np.nan   # test/train = 1/4
    pnb = 2 * (1 - stats.t.cdf(abs(tc), n - 1)) if var > 0 else np.nan
    try: pw = stats.wilcoxon(dif).pvalue
    except Exception: pw = np.nan
    a = oof[(oof.model == T) & (oof["repeat"] == 0)].sort_values("rec_id"); b = oof[(oof.model == m) & (oof["repeat"] == 0)].sort_values("rec_id")
    ca, cb = (a.true.values == a.pred.values), (b.true.values == b.pred.values)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pm = stats.binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else np.nan
    res.append(dict(pembanding=m, selisih_macroF1=dif.mean(), p_Nadeau_Bengio=pnb, p_Wilcoxon=pw,
                    benar_hanya_ensemble=n01, benar_hanya_pembanding=n10, p_McNemar_exact=pm))
res = pd.DataFrame(res); res.to_csv(f"{OUT}/uji_berpasangan.csv", index=False); P(res.round(4).to_string(index=False))
open(f"{OUT}/analisis_cv_log.txt", "w", encoding="utf-8").write("\n".join(log))
