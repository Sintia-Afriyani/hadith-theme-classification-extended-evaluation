# -*- coding: utf-8 -*-
"""Rekonstruksi alur data dari data mentah: jumlah kelas tiap tahap, kolam penuh (sebelum undersampling)."""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
REPO = "/home/claude/sintia-afriyani/hadith-theme-classification-uad"
src = open(f"{REPO}/langkah2_pipeline.py", encoding="utf-8").read().split("\n")
ns = {"re": re, "ENGLISH_STOP_WORDS": ENGLISH_STOP_WORDS}
exec("\n".join(src[127:170]), ns)          # fix_mojibake .. clean_hadith_text
clean, fixm = ns["clean_hadith_text"], ns["fix_mojibake"]
PRIMARY = ["Sahih al-Bukhari","Sahih Muslim","Sunan Abi Dawud","Jami` at-Tirmidhi","Sunan an-Nasa'i","Sunan Ibn Majah"]
stages = {}
def snap(name, d): stages[name] = d["label"].value_counts().reindex(["Bersuci","Makanan","Pengobatan","Fenomena Alam"]).fillna(0).astype(int).to_dict() | {"Total": len(d)}
df = pd.read_csv(sys.argv[1], low_memory=False)
snap("1. Data mentah (6 koleksi, 30 kitab)", df)
for c in ["book_en", "chapter_en", "text_en", "english_full"]: df[c] = df[c].map(fixm)
df["text"] = df["text_en"].where(df["text_en"].astype(str).str.strip().ne("") & df["text_en"].notna(), df["english_full"])
df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["text"].str.lower().ne("nan")]; df["text"] = df["text"].map(clean)
df["n_words"] = df["text"].str.split().str.len(); df = df[df["n_words"] >= 5].copy()
snap("2. Setelah pembersihan dan buang teks < 5 kata", df)
df["_key"] = df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
ck = df.groupby("_key")["label"].nunique(); ck = set(ck[ck > 1].index)
df = df[~df["_key"].isin(ck)].copy(); snap("3. Setelah buang teks identik berlabel beda", df)
df["_prio"] = df["collection"].map({c: i for i, c in enumerate(PRIMARY)})
df = df.sort_values(["_prio"], kind="stable").drop_duplicates("_key", keep="first").copy()
snap("4. Setelah deduplikasi (KOLAM PENUH sebelum undersampling)", df)
pool = df.copy()
idx = df.index[df["label"] == "Bersuci"]
keep = pd.Series(idx).sample(400, random_state=42).values
df = df[(df["label"] != "Bersuci") | (df.index.isin(keep))].copy().reset_index(drop=True)
snap("5. Setelah Bersuci dibatasi 400 (dataset 1.329)", df)
old = pd.read_csv(f"{REPO}/dataset_rilis_tanpa_enam.csv")
six = {"https://sunnah.com/abudawud:10","https://sunnah.com/abudawud:1193","https://sunnah.com/abudawud:1183","https://sunnah.com/tirmidhi:1859","https://sunnah.com/ibnmajah:3318","https://sunnah.com/ibnmajah:3358"}
print("cocok dengan dataset 1.329 di repo:", set(df.url) == set(old.url) | six, "| n", len(df), "| selisih:", len(set(df.url) ^ (set(old.url) | six)))
df2 = df[~df.url.isin(six)]; snap("6. Setelah 6 hadis maudu'/munkar dikeluarkan (1.323)", df2)
S = pd.DataFrame(stages).T; print(S.to_string()); S.to_csv(sys.argv[2] + "/alur_jumlah_kelas.csv")
pool = pool[~pool.url.isin(six)].reset_index(drop=True)
pool["in_final"] = pool.url.isin(set(df2.url))
pool[["collection","hadith_no_in_book","book_en","chapter_en","label","text","url","in_final"]].to_csv(sys.argv[2] + "/kolam_penuh_tanpa_enam.csv", index=False)
print("kolam penuh tanpa enam:", len(pool), pool.label.value_counts().to_dict(), "| dalam dataset final:", int(pool.in_final.sum()))
print(pd.crosstab(pool.collection, pool.label).to_string())
print("Bersuci per koleksi, dikeluarkan vs dipakai:"); print(pd.crosstab(pool[pool.label=="Bersuci"].collection, pool[pool.label=="Bersuci"].in_final).to_string())
