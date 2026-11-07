# -*- coding: utf-8 -*-
"""Sensitivitas validitas label dari lembar telaah 237 hadis (kesalahan = sensus; benar = acak sederhana)."""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad"
m = pd.read_csv("telaah237_gabung.csv")
six = ["abudawud:10", "abudawud:1193", "abudawud:1183", "tirmidhi:1859", "ibnmajah:3318", "ibnmajah:3358"]
pop = pd.read_csv(f"{SP}/repo/hasil_langkah3/oof_prediksi.csv")
def prep(version):
    p, s = pop.copy(), m.copy()
    if version == "v1.1":
        ex = lambda u: any(u.endswith("/" + x) for x in six)
        p = p[~p.url.map(ex)]; s = s[~s.url.map(ex)]
    s["tm"] = s.P4.eq("tema_campur")
    return p, s
rng = np.random.RandomState(0); out = []
def est(p, s, cond, rs=None):
    """proporsi hadis populasi yang memenuhi cond, sampel benar diambil ulang bila rs diberikan"""
    Ne, Nc = (~p.ok).sum(), p.ok.sum(); e, c = s[~s.ok], s[s.ok]
    if rs is not None: c = c.iloc[rs.randint(0, len(c), len(c))]
    return (Ne * cond(e).mean() + Nc * cond(c).mean()) / (Ne + Nc), Ne, Nc, e, c
def ci(p, s, cond, B=5000):
    rs = np.random.RandomState(1); v = [est(p, s, cond, rs)[0] for _ in range(B)]; return np.percentile(v, [2.5, 97.5])
for ver in ["v1.0", "v1.1"]:
    p, s = prep(ver); N = len(p)
    for name, cond in [("Isi cocok penuh (ya)", lambda d: d.P1.eq("ya")), ("Cocok sebagian", lambda d: d.P1.eq("sebagian")),
                       ("Tidak cocok", lambda d: d.P1.eq("tidak")), ("Ya atau sebagian", lambda d: d.P1.ne("tidak")),
                       ("Bertema campur (P4)", lambda d: d.tm), ("Kerangka fikih (P4)", lambda d: d.P4.eq("kerangka_fikih"))]:
        v = est(p, s, cond)[0]; lo, hi = ci(p, s, cond); out.append(dict(versi=ver, N=N, kuantitas=name, estimasi_korpus=v, ci_lo=lo, ci_hi=hi))
    # akurasi terhadap label kitab bila hadis dikeluarkan berdasarkan validitas label
    for name, keep in [("Semua hadis (acuan)", lambda d: pd.Series(True, index=d.index)), ("Tanpa hadis 'tidak cocok'", lambda d: d.P1.ne("tidak")),
                       ("Hanya 'ya' (label cocok penuh)", lambda d: d.P1.eq("ya")), ("Tanpa hadis bertema campur", lambda d: ~d.tm)]:
        def acc(rs=None):
            Ne, Nc = (~p.ok).sum(), p.ok.sum(); e, c = s[~s.ok], s[s.ok]
            if rs is not None: c = c.iloc[rs.randint(0, len(c), len(c))]
            ke, kc = Ne * keep(e).mean(), Nc * keep(c).mean(); return kc / (kc + ke), (ke + kc)
        a, n_ = acc(); rs = np.random.RandomState(2); b = [acc(rs)[0] for _ in range(5000)]
        out.append(dict(versi=ver, N=N, kuantitas="AKURASI: " + name, estimasi_korpus=a, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_hadis_setara=n_))
R = pd.DataFrame(out); R.to_csv("sensitivitas_label_ringkasan.csv", index=False); pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
# per kelas (v1.1)
p, s = prep("v1.1"); rows = []
for c in ["Bersuci", "Makanan", "Pengobatan", "Fenomena Alam"]:
    pc, sc = p[p.label == c], s[s.label == c]; Ne, Nc = (~pc.ok).sum(), pc.ok.sum(); e, k = sc[~sc.ok], sc[sc.ok]
    f = lambda q: (Ne * q(e).mean() + Nc * q(k).mean()) / (Ne + Nc)
    rows.append(dict(kelas=c, N_populasi=len(pc), galat_model=Ne, n_telaah=len(sc), n_ya=int((sc.P1 == "ya").sum()), n_sebagian=int((sc.P1 == "sebagian").sum()), n_tidak=int((sc.P1 == "tidak").sum()),
                     est_ya=f(lambda d: d.P1.eq("ya")), est_sebagian=f(lambda d: d.P1.eq("sebagian")), est_tidak=f(lambda d: d.P1.eq("tidak")),
                     n_kerangka_fikih=int((sc.P4 == "kerangka_fikih").sum()), n_tema_campur=int((sc.P4 == "tema_campur").sum())))
C = pd.DataFrame(rows); C.to_csv("sensitivitas_label_per_kelas.csv", index=False); print(C.round(3).to_string(index=False))
# galat vs benar: sebaran P1 dan P4
print(pd.crosstab(s.ok.map({True: "prediksi benar", False: "prediksi salah"}), s.P1, normalize="index").round(3))
print("share galat dengan P1 != ya:", round(float(s[~s.ok].P1.ne("ya").mean()), 3), "| share benar dengan P1 != ya:", round(float(s[s.ok].P1.ne("ya").mean()), 3))
print("P3 butuh konteks (ya):", round(float(s.P3.eq("ya").mean()), 3))
