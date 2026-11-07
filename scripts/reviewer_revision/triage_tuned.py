# -*- coding: utf-8 -*-
"""Triase berbasis kepercayaan pada run Tabel 4 (nested CV, ulangan 0, ensemble tuned, 123 kesalahan)."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad/m5")
from common import *
from sklearn.metrics import roc_auc_score
SP = "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad"
d = load(); X, y = d.text.values, d.label.values; classes = sorted(set(y))
fa = pd.read_csv(f"{SP}/m5/out/fold_assignments.csv")["rep0"].values
bp = pd.read_csv(f"{SP}/m5/out/cv_best_params_per_fold.csv"); bp = bp[bp["repeat"] == 0].set_index("fold")
P = np.zeros((len(d), 4))
for f in range(5):
    tr, te = np.where(fa != f)[0], np.where(fa == f)[0]; r = bp.loc[f]
    P[te] = pipe(ens(float(r["svm__C"]), int(r["rf__n_estimators"]), float(r["nb__alpha"]))).fit(X[tr], y[tr]).predict_proba(X[te])
pred = np.array(classes)[P.argmax(1)]; conf = P.max(1); ok = pred == y
pd.DataFrame({"rec_id": d.rec_id, "true": y, "pred": pred, "conf": conf, "ok": ok}).to_csv("oof_probabilitas_rep0_tuned.csv", index=False)
o = np.argsort(-conf)
pd.DataFrame([dict(cakupan=c, n=int(round(c*len(d))), akurasi=ok[o[:int(round(c*len(d)))]].mean(), galat=int((~ok[o[:int(round(c*len(d)))]]).sum())) for c in [1.0, .9, .8, .7, .6, .5]]).to_csv("risiko_cakupan_tuned.csv", index=False)
print("akurasi", ok.mean(), "galat", int((~ok).sum()), "AUROC", roc_auc_score(ok, conf))
