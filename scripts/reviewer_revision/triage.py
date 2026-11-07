# -*- coding: utf-8 -*-
"""Kepercayaan model sebagai alat triase: kurva risiko-cakupan, kalibrasi, dan hubungan dengan kecocokan label menurut pakar."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad/m5")
from common import *
from sklearn.metrics import roc_auc_score
SP = "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad"
d = load(); X, y = d.text.values, d.label.values; classes = sorted(set(y))
fa = pd.read_csv(f"{SP}/m5/out/fold_assignments.csv")["rep0"].values
P = np.zeros((len(d), 4))
for f in range(5):
    tr, te = np.where(fa != f)[0], np.where(fa == f)[0]; P[te] = pipe(ens(1.0, 100, 0.1)).fit(X[tr], y[tr]).predict_proba(X[te])
pred = np.array(classes)[P.argmax(1)]; conf = P.max(1); ok = pred == y
pd.DataFrame({"rec_id": d.rec_id, "true": y, "pred": pred, "conf": conf, "ok": ok}).to_csv("oof_probabilitas_rep0.csv", index=False)
print("akurasi OOF (rep0, ensemble default):", round(ok.mean(), 4), "| galat:", int((~ok).sum()))
o = np.argsort(-conf); rows = []
for cov in [1.0, .9, .8, .7, .6, .5]:
    k = int(round(cov * len(d))); sel = o[:k]; rows.append(dict(cakupan=cov, n=k, akurasi=ok[sel].mean(), galat=int((~ok[sel]).sum()), ambang_kepercayaan=conf[sel].min()))
RC = pd.DataFrame(rows); RC.to_csv("risiko_cakupan.csv", index=False); print(RC.round(3).to_string(index=False))
low = o[int(0.8 * len(d)):]; print("20% paling tidak yakin memuat", int((~ok[low]).sum()), "dari", int((~ok).sum()), "galat (", round(float((~ok[low]).sum() / (~ok).sum()), 3), ")")
bins = np.linspace(0, 1, 11); ece = 0
for a, b in zip(bins[:-1], bins[1:]):
    m = (conf > a) & (conf <= b)
    if m.any(): ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
brier = np.mean([(P[i] - (np.array(classes) == y[i])) ** 2 for i in range(len(d))]) * 4
print("ECE:", round(ece, 3), "| Brier (jumlah kuadrat, rata2 hadis):", round(float(brier), 3), "| AUROC konf memisahkan benar/salah:", round(roc_auc_score(ok, conf), 3))
# hubungan dengan kecocokan label menurut pakar (kepercayaan run B, 1.329 hadis)
m = pd.read_csv(f"{SP}/lab/telaah237_gabung.csv"); k = pd.read_csv(f"{SP}/repo/hasil_langkah3/kunci_telaah.csv")
s = pd.read_csv("/home/claude/sintia-afriyani/hadith-theme-classification-uad/Lembar-Telaah-Fix.csv").iloc[:, [4, 5, 8]]; s.columns = ["url", "P1", "P4"]
oo = pd.read_csv(f"{SP}/repo/hasil_langkah3/oof_prediksi.csv"); print(oo.columns.tolist())
conf_col = "conf" if "conf" in oo.columns else None
if conf_col:
    t = s.merge(oo[["url", "conf", "ok"]], on="url"); t["ya"] = t.P1.eq("ya")
    print("n telaah", len(t), "| konf rata2 per P1:", t.groupby("P1").conf.mean().round(3).to_dict())
    print("AUROC konfiden -> label cocok penuh (semua 237):", round(roc_auc_score(t.ya, t.conf), 3))
    for nm, g in t.groupby("ok"): print("  dalam kelompok", "benar" if nm else "salah", "n=", len(g), "AUROC:", round(roc_auc_score(g.ya, g.conf), 3))
    t["tm"] = t.P4.eq("tema_campur"); print("konf rata2 tema campur vs lain:", t.groupby("tm").conf.mean().round(3).to_dict())
    t.to_csv("konfiden_vs_kecocokan_pakar.csv", index=False)
