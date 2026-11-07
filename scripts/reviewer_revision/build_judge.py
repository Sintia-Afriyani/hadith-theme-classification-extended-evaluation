# -*- coding: utf-8 -*-
import math, sys, pandas as pd, numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-claude/5d45981c-16eb-56e7-ae5d-3281f3e68879/scratchpad/m5")
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from common import load
d = load()[["rec_id", "text", "label"]]
R = pd.read_csv("kueri_hasil_per_sistem.csv"); Q = pd.read_csv("kueri_daftar.csv")
pool = R.groupby(["qid", "rec_id"]).agg(label=("label", "first")).reset_index().merge(d[["rec_id", "text"]], on="rec_id").merge(Q, on="qid")
pool["_o"] = pool.groupby("qid").rec_id.transform(lambda s: np.random.RandomState(11).permutation(len(s)))
pool = pool.sort_values(["qid", "_o"]).reset_index(drop=True); pool["No"] = np.arange(1, len(pool) + 1)
key = R.pivot_table(index=["qid", "rec_id"], columns="sistem", values="rank", aggfunc="min").reset_index().merge(pool[["No", "qid", "rec_id", "label"]], on=["qid", "rec_id"])
key.to_csv("pencarian_kunci_kolam.csv", index=False)
N = len(pool); print("butir:", N, pool.groupby("qid").size().to_dict())
F = lambda **k: Font(name="Arial", size=k.pop("size", 10), **k)
YEL = PatternFill("solid", fgColor="FFF2CC"); HDR = PatternFill("solid", fgColor="D9E1F2")
thin = Side(style="thin", color="BFBFBF"); BOR = Border(left=thin, right=thin, top=thin, bottom=thin)
OPT = ["2 - Relevan", "1 - Sebagian relevan", "0 - Tidak relevan"]
GUIDE = [("Tujuan", "Menilai apakah setiap hadis berguna bagi sebuah kebutuhan informasi (kueri pencarian). Hasilnya dipakai untuk mengevaluasi sistem pencarian hadis pada penelitian klasifikasi tema hadis."),
 ("Cara menilai", "Untuk tiap baris, bayangkan Anda peneliti hadis yang mencari riwayat untuk kueri pada kolom \"Kueri\". Baca teks hadis, lalu pilih salah satu pada kolom \"Relevansi\"."),
 ("2 - Relevan", "Hadis membahas langsung hal yang dicari kueri, dan akan Anda ambil sebagai bahan."),
 ("1 - Sebagian relevan", "Hadis berkaitan tetapi tidak langsung: hanya menyinggung sebagian kueri, atau konteksnya berdekatan."),
 ("0 - Tidak relevan", "Hadis tidak membantu kueri ini, walaupun mungkin membahas topik lain yang bagus."),
 ("Aturan penting", "Nilailah berdasarkan ISI hadis. Urutan baris sudah diacak dan sistem pencarian tidak ditampilkan. Kerjakan mandiri tanpa berdiskusi dengan penilai lain. Satu hadis bisa relevan untuk kueri bertema ganda (misalnya makanan dan bersuci sekaligus): nilai berdasarkan kueri."),
 ("Urutan kerja", "Kerjakan dari atas. Jika waktu terbatas, selesaikan Q1 sampai Q4 lebih dulu (baris paling atas); Q5 sampai Q8 menyusul bila sempat."),
 ("Pengembalian", "Simpan berkas, lalu kirim kembali ke penulis.")]
for k in (1, 2):
    wb = Workbook(); ws = wb.active; ws.title = "Panduan"; ws["A1"] = f"Panduan penilaian relevansi pencarian hadis (Penilai {k})"; ws["A1"].font = F(bold=True, size=13)
    for i, (a, t) in enumerate(GUIDE):
        r = 3 + i; ws.cell(r, 1, a).font = F(bold=True); c = ws.cell(r, 2, t); c.font = F(); c.alignment = Alignment(wrap_text=True, vertical="top"); ws.cell(r, 1).alignment = Alignment(vertical="top")
        ws.row_dimensions[r].height = 15 * max(2, math.ceil(len(t) / 95))
    ws.column_dimensions["A"].width = 22; ws.column_dimensions["B"].width = 100
    w = wb.create_sheet("Penilaian"); w["A1"] = f"Penilaian relevansi, Penilai {k} ({N} butir, {pool.qid.nunique()} kueri)"; w["A1"].font = F(bold=True, size=13)
    for j, h in enumerate(["No", "Kueri", "Kueri (Indonesia)", "ID", "Teks hadis", "Relevansi", "Catatan"], 1):
        c = w.cell(3, j, h); c.font = F(bold=True); c.fill = HDR; c.border = BOR; c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in pool.iterrows():
        r = 4 + i
        for j, v in enumerate([int(row.No), f"{row.qid}: {row.kueri_en}", row.kueri_id, row.rec_id, row.text], 1):
            c = w.cell(r, j, v); c.font = F(); c.border = BOR; c.alignment = Alignment(wrap_text=True, vertical="top")
        for j in (6, 7): c = w.cell(r, j); c.fill = YEL; c.border = BOR; c.font = F(); c.alignment = Alignment(wrap_text=True, vertical="top")
        w.row_dimensions[r].height = min(409, 14 * max(3, math.ceil(len(row.text) / 100)))
    for col, wd in zip("ABCDEFG", [5, 28, 26, 8, 95, 20, 30]): w.column_dimensions[col].width = wd
    dv = DataValidation(type="list", formula1='"' + ",".join(OPT) + '"', allow_blank=True); w.add_data_validation(dv); dv.add(f"F4:F{3+N}")
    w.freeze_panes = "F4"; wb.save(f"Penilaian_Pencarian_Pakar_{k}.xlsx")
