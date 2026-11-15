"""Job ROI & Growth model.

Run from the Datasets folder:
    .venv/bin/python job_roi_model/01_extract_indeed.py   # slims the 2 GB Indeed CSV
    .venv/bin/python job_roi_model/02_model.py

Three predictive models feed one occupation-level score:
  1. Salary model     - Ridge regression on posting text -> log annual salary
  2. Burnout model    - TF-IDF + logistic regression on review text -> P(work-life rating <= 2)
  3. Promotion model  - TF-IDF + logistic regression on review text -> P(career-opportunity rating >= 4)
Each is compared with simple baselines on a held-out set split by COMPANY
(so the model cannot memorise one employer's postings/reviews).
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score, f1_score,
                             accuracy_score, average_precision_score, brier_score_loss)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from occupations import map_title

ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
RNG = 42
metrics = {}

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
                "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())

# ---------------------------------------------------------------- Indeed
jobs = pd.read_parquet(OUT / "indeed_slim.parquet").drop_duplicates("jobid")
jobs["occupation"] = jobs.job_title.map(map_title)
jobs["state"] = jobs.location.str.extract(r",\s*([A-Z]{2})\b")[0].where(lambda s: s.isin(US_STATES), "Other/Remote")
jobs["date"] = pd.to_datetime(jobs.date_posted_parsed, utc=True, errors="coerce")

PERIOD = {"hour": 2080, "hora": 2080, "day": 260, "día": 260, "week": 52, "semana": 52,
          "month": 12, "mes": 12, "year": 1, "año": 1}

def parse_salary(s):
    if not isinstance(s, str):
        return np.nan, None
    nums = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,]+(?:\.\d+)?)", s)]
    unit = next((u for u in PERIOD if u in s.lower()), None)
    if not nums or unit is None:
        return np.nan, None
    return float(np.mean(nums[:2])) * PERIOD[unit], unit

parsed = jobs.salary_formatted.map(parse_salary)
jobs["salary_annual"] = [p[0] for p in parsed]
jobs["pay_unit"] = [p[1] for p in parsed]
jobs.loc[~jobs.salary_annual.between(15_000, 1_000_000), "salary_annual"] = np.nan
jobs["hourly"] = jobs.pay_unit.isin(["hour", "hora"])
print("postings:", len(jobs), "with salary:", jobs.salary_annual.notna().sum(),
      "mapped to occupation:", jobs.occupation.notna().mean().round(3))

# education requirement extracted from description (for financial ROI)
def edu_years(text):
    t = text.lower() if isinstance(text, str) else ""
    if re.search(r"\b(ph\.?d|doctorate|doctoral|m\.?d\.|pharm\.?d|\bdpt\b|doctor of|juris doctor|\bj\.?d\b|\bdds\b|\bod\b)", t): return 8
    if re.search(r"master'?s|\bm\.?s\.?\b|\bmba\b|\bmsw\b|\bmsn\b|graduate degree", t): return 6
    if re.search(r"bachelor'?s|\bb\.?s\.?\b|\bb\.?a\.?\b|\bbsn\b|4.year degree|college degree|undergraduate degree", t): return 4
    if re.search(r"associate'?s degree|\badn\b|2.year degree|certificate program|vocational", t): return 2
    return 0
jobs["edu_years"] = jobs.description_text.map(edu_years)

# ---------------------------------------------------------------- Model 1: salary
sal = jobs[jobs.salary_annual.notna()].copy()
# remove dollar amounts / pay phrases from the description to avoid target leakage
sal["desc_clean"] = (sal.description_text.fillna("")
                     .str.replace(r"\$\s?[\d,]+(\.\d+)?[kK]?", " ", regex=True)
                     .str.replace(r"\b\d[\d,]*(\.\d+)?\s?(per|an|a|/)\s?(hour|hr|year|yr|annum)\b", " ", regex=True))
sal["y"] = np.log(sal.salary_annual)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
tr, te = next(gss.split(sal, groups=sal.company_name))
s_tr, s_te = sal.iloc[tr], sal.iloc[te]

v_title = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
v_desc = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60_000, sublinear_tf=True, stop_words="english")
ohe = OneHotEncoder(handle_unknown="ignore")
def cat_frame(d):
    return pd.DataFrame({"jt": d.job_type.fillna("unknown"), "st": d.state, "occ": d.occupation.fillna("Unmapped"),
                         "edu": d.edu_years.astype(str)})
def build(d, fit=False):
    parts = [v_title.fit_transform(d.job_title) if fit else v_title.transform(d.job_title),
             v_desc.fit_transform(d.desc_clean) if fit else v_desc.transform(d.desc_clean),
             ohe.fit_transform(cat_frame(d)) if fit else ohe.transform(cat_frame(d))]
    return sparse.hstack(parts).tocsr()
X_tr, X_te = build(s_tr, fit=True), build(s_te)
ridge = Ridge(alpha=1.0).fit(X_tr, s_tr.y)
pred = np.exp(ridge.predict(X_te))
actual = s_te.salary_annual.values

base_global = np.full(len(s_te), np.exp(s_tr.y.median()))
occ_med = s_tr.groupby(s_tr.occupation.fillna("Unmapped")).salary_annual.median()
base_occ = s_te.occupation.fillna("Unmapped").map(occ_med).fillna(base_global[0]).values

def reg_report(p):
    return {"MAE_$": round(float(mean_absolute_error(actual, p))),
            "MedAPE_%": round(float(np.median(np.abs(p - actual) / actual) * 100), 1),
            "R2_log": round(float(r2_score(np.log(actual), np.log(p))), 3)}
metrics["salary"] = {"n_train": len(s_tr), "n_test": len(s_te), "split": "grouped by company (80/20)",
                     "baseline_global_median": reg_report(base_global),
                     "baseline_occupation_median": reg_report(base_occ),
                     "ridge_text_model": reg_report(pred)}
# fairness: error by wage tier and pay type
s_te = s_te.assign(pred=pred, ape=np.abs(pred - actual) / actual)
s_te["wage_tier"] = pd.qcut(s_te.salary_annual, 3, labels=["low", "middle", "high"])
metrics["salary"]["by_wage_tier_MedAPE_%"] = (s_te.groupby("wage_tier", observed=True).ape.median() * 100).round(1).to_dict()
metrics["salary"]["by_pay_type_MedAPE_%"] = (s_te.groupby(np.where(s_te.hourly, "hourly", "salaried")).ape.median() * 100).round(1).to_dict()
print("salary:", json.dumps(metrics["salary"], indent=1))

# predicted salary for every posting (fills the ~14% without a posted salary)
jobs["desc_clean"] = jobs.description_text.fillna("").str.replace(r"\$\s?[\d,]+(\.\d+)?[kK]?", " ", regex=True)
jobs["salary_pred"] = np.exp(ridge.predict(build(jobs)))
jobs["salary_best"] = jobs.salary_annual.fillna(jobs.salary_pred)

# ---------------------------------------------------------------- Glassdoor
RCOLS = ["rating_overall", "rating_work_life", "rating_culture_values", "rating_diversity_inclusion",
         "rating_career_opportunities", "rating_compensation_benefits", "rating_senior_leadership"]
rev = pd.read_csv(ROOT / "glassdoor_reviews.csv", low_memory=False,
                  usecols=RCOLS + ["review_id", "company_name", "employee_job_title", "employee_location",
                                   "employee_status", "employee_type", "summary", "review_pros", "review_cons",
                                   "rating_date", "flags_recommend_frend"])
rev = rev.drop_duplicates("review_id")
for c in RCOLS[1:]:
    rev[c] = rev[c].replace(0, np.nan)  # 0 = sub-rating not given
rev["state"] = rev.employee_location.str.extract(r",\s*([A-Z]{2})$")[0]
metrics["glassdoor_rows"] = {"all": len(rev), "us": int(rev.state.isin(US_STATES).sum())}
rev = rev[rev.state.isin(US_STATES)].copy()
rev["occupation"] = rev.employee_job_title.map(map_title)
rev["text"] = (rev.summary.fillna("") + " . PROS: " + rev.review_pros.fillna("") + " . CONS: " + rev.review_cons.fillna(""))
rev["cons_l"] = rev.review_cons.fillna("").str.lower()
rev["pros_l"] = rev.review_pros.fillna("").str.lower()

# lexicon features (transparent, used as a baseline and as a descriptive signal)
BURN = r"burn ?out|burnt out|overwork|over-work|exhaust|long hours|mandatory overtime|forced overtime|understaff|short.?staff|no work.?life|stress|toxic|micromanag|no breaks|draining|overload"
PROMO = r"promot|advance|career growth|growth opportunit|move up|moving up|career progression|upward mobility|ladder"
rev["burn_lex"] = rev.cons_l.str.contains(BURN)
rev["promo_pos"] = rev.pros_l.str.contains(PROMO)
rev["promo_neg"] = rev.cons_l.str.contains(PROMO)

def text_classifier(label_col, positive, name, lexicon_flag=None):
    d = rev[rev[label_col].notna()].copy()
    d["y"] = positive(d[label_col]).astype(int)
    g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
    a, b = next(g.split(d, groups=d.company_name))
    dtr, dte = d.iloc[a], d.iloc[b]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=150_000, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, solver="liblinear")  # unweighted -> probabilities stay calibrated for rate estimates
    Xtr_t, Xte_t = vec.fit_transform(dtr.text), vec.transform(dte.text)
    clf.fit(Xtr_t, dtr.y)
    p = clf.predict_proba(Xte_t)[:, 1]
    star = OneHotEncoder(handle_unknown="ignore").fit(dtr[["rating_overall"]])
    clf2 = LogisticRegression(C=4.0, solver="liblinear").fit(sparse.hstack([Xtr_t, star.transform(dtr[["rating_overall"]])]).tocsr(), dtr.y)
    p2 = clf2.predict_proba(sparse.hstack([Xte_t, star.transform(dte[["rating_overall"]])]).tocsr())[:, 1]
    def rep(prob, hard=None):
        hard = (prob >= 0.5).astype(int) if hard is None else hard
        r = {"accuracy": accuracy_score(dte.y, hard), "F1": f1_score(dte.y, hard)}
        if prob is not None and len(np.unique(prob)) > 1:
            r.update({"ROC_AUC": roc_auc_score(dte.y, prob), "PR_AUC": average_precision_score(dte.y, prob),
                      "Brier": brier_score_loss(dte.y, prob)})
        return {k: round(float(v), 3) for k, v in r.items()}
    res = {"n_train": len(dtr), "n_test": len(dte), "positive_rate": round(float(dte.y.mean()), 3),
           "baseline_majority_class": rep(np.full(len(dte), float(dtr.y.mean() >= .5)), np.full(len(dte), int(dtr.y.mean() >= .5)))}
    if lexicon_flag is not None:
        lex = lexicon_flag(dte).astype(int).values
        res["baseline_keyword_rule"] = rep(lex.astype(float), lex)
    # strong baseline: overall star rating only (no text)
    lr1 = LogisticRegression().fit(dtr[["rating_overall"]], dtr.y)
    res["baseline_overall_star_only"] = rep(lr1.predict_proba(dte[["rating_overall"]])[:, 1])
    res["tfidf_logreg_text_model"] = rep(p)
    res["text_plus_stars_model"] = rep(p2)
    # subgroup performance (bias check)
    dte = dte.assign(p=p)
    sub = {}
    for grp, gd in dte.groupby("employee_status"):
        if len(gd) >= 200 and gd.y.nunique() == 2:
            sub[grp] = {"n": len(gd), "ROC_AUC": round(float(roc_auc_score(gd.y, gd.p)), 3),
                        "actual_rate": round(float(gd.y.mean()), 3), "mean_pred": round(float(gd.p.mean()), 3)}
    res["by_employee_status"] = sub
    terms = np.array(vec.get_feature_names_out()); coef = clf.coef_[0]
    res["top_positive_terms"] = terms[np.argsort(coef)[-20:]][::-1].tolist()
    res["top_negative_terms"] = terms[np.argsort(coef)[:20]].tolist()
    metrics[name] = res
    print(name, json.dumps({k: v for k, v in res.items() if "terms" not in k}, indent=1))
    rev[name + "_prob"] = clf.predict_proba(vec.transform(rev.text))[:, 1]

text_classifier("rating_work_life", lambda s: s <= 2, "burnout", lambda d: d.burn_lex)
text_classifier("rating_career_opportunities", lambda s: s >= 4, "promotion",
                lambda d: d.promo_pos & ~d.promo_neg)

# ---------------------------------------------------------------- occupation aggregation
MIN_REV, MIN_JOBS = 40, 40
EXCLUDE = {"Intern"}  # a career stage, not an occupation
rv = rev[rev.occupation.notna() & ~rev.occupation.isin(EXCLUDE)]
jb = jobs[jobs.occupation.notna() & ~jobs.occupation.isin(EXCLUDE)]

def shrunk_mean(col, k=20):
    """Empirical-Bayes shrinkage toward the global mean so small groups aren't extreme."""
    gm = rv[col].mean()
    a = rv.groupby("occupation")[col].agg(["sum", "count"])
    return (a["sum"] + k * gm) / (a["count"] + k)

occ = pd.DataFrame({
    "n_reviews": rv.groupby("occupation").size(),
    "culture_diversity": (shrunk_mean("rating_culture_values") + shrunk_mean("rating_diversity_inclusion")) / 2,
    "career_opportunities": shrunk_mean("rating_career_opportunities"),
    "work_life": shrunk_mean("rating_work_life"),
    "promotion_likelihood": rv.groupby("occupation").promotion_prob.mean(),
    "promo_mention_net": rv.groupby("occupation").apply(lambda d: d.promo_pos.mean() - d.promo_neg.mean()),
    "burnout_rate": rv.groupby("occupation").burnout_prob.mean(),
    "burnout_keyword_rate": rv.groupby("occupation").burn_lex.mean(),
    "overall_rating": shrunk_mean("rating_overall"),
}).join(pd.DataFrame({
    "n_postings": jb.groupby("occupation").size(),
    "n_postings_with_salary": jb.groupby("occupation").salary_annual.count(),
    "median_salary_posted": jb.groupby("occupation").salary_annual.median(),
    "median_salary_model": jb.groupby("occupation").salary_best.median(),
    "share_hourly": jb.groupby("occupation").hourly.mean(),
    "typical_edu_years": jb.groupby("occupation").edu_years.median(),
}), how="inner")
occ = occ[(occ.n_reviews >= MIN_REV) & (occ.n_postings >= MIN_JOBS)].copy()

# demand growth: change in the occupation's SHARE of all postings (cancels scraper volume swings)
early = jb.date < "2026-02-01"
tot_e, tot_l = early.sum(), (~early).sum()
ce = jb[early].groupby("occupation").size().reindex(occ.index).fillna(0)
cl = jb[~early].groupby("occupation").size().reindex(occ.index).fillna(0)
occ["demand_growth_pct"] = (np.exp(np.log((cl + 5) / tot_l) - np.log((ce + 5) / tot_e)) - 1) * 100
occ["log_postings"] = np.log(occ.n_postings)

# financial ROI (superseded by 02b_roi.py, which overwrites these columns): 20-year NPV of earnings premium over a no-degree baseline, net of education cost
TUITION, DISC, HORIZON = 27_000, 0.03, 20
# baseline = median pay across occupations whose typical posting asks for no degree
no_deg_occ = occ.index[occ.typical_edu_years == 0]
BASE = float(jb[jb.occupation.isin(no_deg_occ)].salary_best.median())
def npv_premium(salary, edu):
    edu = int(edu)
    cost = sum((TUITION + BASE) / (1 + DISC) ** t for t in range(edu))
    gain = sum((salary - BASE) / (1 + DISC) ** t for t in range(edu, HORIZON))
    return gain - cost, cost
r = [npv_premium(s, e) for s, e in zip(occ.median_salary_model, occ.typical_edu_years)]
occ["npv_20yr_premium"] = [x[0] for x in r]
occ["education_cost"] = [x[1] for x in r]
occ["roi_multiple"] = np.where(occ.education_cost > 0, occ.npv_20yr_premium / occ.education_cost, np.nan)
metrics["roi_assumptions"] = {"tuition_per_year": TUITION, "discount_rate": DISC, "horizon_years": HORIZON,
                              "no_degree_baseline_salary": round(BASE), "no_degree_occupations": list(no_deg_occ)}

# composite score: percentile rank of each component, higher = better (burnout inverted)
COMPONENTS = {"median_salary_model": +1, "culture_diversity": +1, "career_opportunities": +1,
              "log_postings": +1, "promotion_likelihood": +1, "work_life": +1, "burnout_rate": -1}
def score(frame, weights=None):
    weights = weights or {c: 1 for c in COMPONENTS}
    tot = sum(weights.values())
    return sum(weights[c] * (frame[c] * sgn).rank(pct=True) for c, sgn in COMPONENTS.items()) / tot * 100
occ["growth_score"] = score(occ)

# bootstrap stability of the ranking (resample reviews and postings within occupation)
rng = np.random.default_rng(RNG)
boots = []
rv_o = {o: d for o, d in rv[rv.occupation.isin(occ.index)].groupby("occupation")}
jb_o = {o: d.salary_best.values for o, d in jb[jb.occupation.isin(occ.index)].groupby("occupation")}
for b in range(200):
    f = occ.copy()
    for o, d in rv_o.items():
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        f.loc[o, "culture_diversity"] = np.nanmean([s.rating_culture_values.mean(), s.rating_diversity_inclusion.mean()])
        f.loc[o, "career_opportunities"] = s.rating_career_opportunities.mean()
        f.loc[o, "work_life"] = s.rating_work_life.mean()
        f.loc[o, "promotion_likelihood"] = s.promotion_prob.mean()
        f.loc[o, "burnout_rate"] = s.burnout_prob.mean()
        f.loc[o, "median_salary_model"] = np.median(rng.choice(jb_o[o], len(jb_o[o])))
    boots.append(score(f))
B = pd.concat(boots, axis=1)
occ["score_ci_low"], occ["score_ci_high"] = B.quantile(0.05, axis=1), B.quantile(0.95, axis=1)
occ["rank"] = occ.growth_score.rank(ascending=False).astype(int)
occ = occ.sort_values("growth_score", ascending=False)
occ.round(4).to_csv(OUT / "occupation_scores.csv")
metrics["n_occupations_scored"] = len(occ)
metrics["coverage"] = {"postings_mapped": round(float(jobs.occupation.notna().mean()), 3),
                       "us_reviews_mapped": round(float(rev.occupation.notna().mean()), 3)}

# correlations among components (redundancy check)
metrics["component_corr"] = occ[list(COMPONENTS)].corr(method="spearman").round(2).to_dict()
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2, default=str)
print(occ[["growth_score", "score_ci_low", "score_ci_high", "median_salary_model", "work_life", "burnout_rate",
           "promotion_likelihood", "n_reviews", "n_postings", "npv_20yr_premium"]].round(2).to_string())
