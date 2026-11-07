# -*- coding: utf-8 -*-
"""Nested CV pada 1.323 hadis: simpan lipatan, prediksi out-of-fold semua model & semua ulangan, waktu latih."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"; os.makedirs(OUT, exist_ok=True)
d = load(); X, y, g = d.text.values, d.label.values, d.group.values
d[["rec_id", "url", "collection", "label", "group"]].to_csv(f"{OUT}/dataset_ids_v1.1.csv", index=False)
folds = pd.DataFrame({"rec_id": d.rec_id, "url": d.url})
oof, rows, params, t0 = [], [], [], time.time()
for rep in range(R):
    fa = np.zeros(len(d), int)
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED + rep)
    for fold, (tr, te) in enumerate(outer.split(X, y, g)):
        fa[te] = fold
        models = {}
        for name, mk in BASELINES.items():
            t = time.time(); m = mk().fit(X[tr], y[tr]); ft = time.time() - t
            t = time.time(); p = m.predict(X[te]); pt = time.time() - t
            models[name] = (p, ft, pt)
        inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED)
        t = time.time()
        gs = GridSearchCV(pipe(ens()), GRID, cv=inner, scoring="f1_macro", n_jobs=2, refit=True).fit(X[tr], y[tr], groups=g[tr])
        ft = time.time() - t
        models["Ensemble (tuned, nested CV)"] = (gs.predict(X[te]), ft, 0.0)
        params.append(dict(repeat=rep, fold=fold, **{k.replace("clf__", ""): v for k, v in gs.best_params_.items()}))
        for name, (p, ft, pt) in models.items():
            rows.append(dict(model=name, repeat=rep, fold=fold, accuracy=accuracy_score(y[te], p),
                             macro_f1=f1_score(y[te], p, average="macro"),
                             balanced_acc=balanced_accuracy_score(y[te], p), fit_s=ft, pred_s=pt, n_test=len(te)))
            oof.append(pd.DataFrame({"rec_id": d.rec_id.values[te], "repeat": rep, "fold": fold, "model": name,
                                     "true": y[te], "pred": p}))
    folds[f"rep{rep}"] = fa
    pd.DataFrame(rows).to_csv(f"{OUT}/cv_per_fold.csv", index=False)
    pd.concat(oof).to_csv(f"{OUT}/oof_all.csv", index=False)
    pd.DataFrame(params).to_csv(f"{OUT}/cv_best_params_per_fold.csv", index=False)
    folds.to_csv(f"{OUT}/fold_assignments.csv", index=False)
    print(f"ulangan {rep+1}/{R} selesai ({(time.time()-t0)/60:.1f} menit)", flush=True)
print("SELESAI", flush=True)
