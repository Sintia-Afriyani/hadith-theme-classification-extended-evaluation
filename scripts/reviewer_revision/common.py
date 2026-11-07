# -*- coding: utf-8 -*-
"""Bagian bersama untuk analisis pada 1.323 hadis (v1.1)."""
import re, warnings, os, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "v1.1"
))
SEED, OUTER, INNER, NEARDUP_TH = 42, 5, 3, 0.80
SOURCE_STOP = {"hasan","sahih","gharib","daif","eisa","isa","esa","hadith","pbuh","saws","saw","dawud","tradition",
               "traditions","version","versions","authority","chain","chains","narrator","narrators","transmitted",
               "alayhi","sallam","allaahu"}
STOP_MAIN_SET = set(ENGLISH_STOP_WORDS) | SOURCE_STOP
STOP_MAIN = sorted(STOP_MAIN_SET)

def load(th=NEARDUP_TH):
    d = pd.read_csv(f"{REPO}/dataset_rilis_tanpa_enam.csv")
    d = d.sort_values("url", kind="stable").reset_index(drop=True)
    d.insert(0, "rec_id", [f"H{i+1:04d}" for i in range(len(d))])
    d["group"] = near_dup_groups(d["text"], th)
    return d

def near_dup_groups(texts, th):
    Xg = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(texts)
    S = (Xg @ Xg.T).tocsr(); S.data[S.data < th] = 0; S.eliminate_zeros()
    return connected_components(S, directed=False)[1]

def tfidf(stop=None):
    return TfidfVectorizer(lowercase=True, stop_words=(STOP_MAIN if stop is None else stop),
                           ngram_range=(1, 2), max_df=0.9, min_df=3)
def svm(C=1.0): return SVC(C=C, kernel="linear", probability=True, random_state=SEED)
def rf(n=100):  return RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=SEED, n_jobs=1)
def nb(a=1.0):  return MultinomialNB(alpha=a)
def ens(C=1.0, n=100, a=1.0): return VotingClassifier([("svm", svm(C)), ("rf", rf(n)), ("nb", nb(a))], voting="soft")
def pipe(clf, stop=None): return Pipeline([("tfidf", tfidf(stop)), ("clf", clf)])

BASELINES = {
    "Majority class (Dummy)":  lambda: pipe(DummyClassifier(strategy="most_frequent")),
    "Naive Bayes (default)":   lambda: pipe(nb(1.0)),
    "SVM (default)":           lambda: pipe(svm(1.0)),
    "Random Forest (default)": lambda: pipe(rf(100)),
    "Ensemble (default)":      lambda: pipe(ens(1.0, 100, 1.0)),
}
GRID = {"clf__svm__C": [0.1, 1, 10], "clf__rf__n_estimators": [100, 200], "clf__nb__alpha": [0.1, 1]}
