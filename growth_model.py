"""Growth Score Model: one model that turns a job posting into a 0-100 growth score.

    # build the model once (about 5-10 min); writes model/growth_model.json
    .venv/bin/python job_roi_model/growth_model.py build

    # score your own postings: fill in job_roi_model/my_postings_template.csv, then
    .venv/bin/python job_roi_model/growth_model.py score job_roi_model/my_postings_template.csv

How a posting is scored
  1. Salary        - the salary you enter, or predicted from the title + description (ridge regression).
  2. Occupation    - mapped from the job title (or chosen by you).
  3. Review-based  - culture+diversity, career opportunities, work-life balance, promotion likelihood
                     and burnout come from Glassdoor reviews of that company, blended with the
                     occupation average (more reviews -> more weight on the company). You can type in
                     your own ratings instead.
  4. Job openings  - how many postings exist for that occupation.
  5. Each value is compared with 44 reference occupations (percentile, burnout reversed) and the
     weighted average x 100 is the growth score.
The whole model (vocabulary, weights, reference tables) lives in one JSON file that this script
and the web calculator both read, so both give the same answer.
"""
import json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from occupations import FAMILIES, map_title

MODEL_PATH = HERE / "model" / "growth_model.json"
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
             "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
COMPONENTS = [  # key, label, direction
    ("salary", "Salary", 1), ("culture_diversity", "Culture + diversity", 1),
    ("career_opportunities", "Career opportunities", 1), ("job_openings", "Job openings", 1),
    ("promotion_likelihood", "Promotion likelihood", 1), ("work_life", "Work-life balance", 1),
    ("burnout_rate", "Burnout rate", -1)]
SHRINK_K = 20          # reviews needed before a company counts as much as its occupation average
MIN_COMPANY_REVIEWS = 10
TOKEN = re.compile(r"\b\w\w+\b", re.U)  # same as scikit-learn's default token_pattern
MONEY = re.compile(r"\$\s?[\d,]+(\.\d+)?[kK]?")


def norm_company(name):
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("&", " and "))
    s = re.sub(r"\b(inc|llc|l l c|corp|corporation|co|company|ltd|limited|plc|lp|llp|the|group|holdings)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def state_of(location):
    m = re.search(r",\s*([A-Z]{2})\b", location or "")
    return m.group(1) if m and m.group(1) in US_STATES else "Other/Remote"


# ------------------------------------------------------------------ scoring (pure python, mirrors the JS)
class GrowthScoreModel:
    def __init__(self, path=MODEL_PATH):
        self.m = json.load(open(path))
        self.fam = [(n, re.compile(p)) for n, p in self.m["occupation_patterns"]]
        self.stop = set(self.m["stop_words"])

    def _tfidf(self, text, vocab, ngram, stop):
        toks = TOKEN.findall((text or "").lower())
        if stop:
            toks = [t for t in toks if t not in self.stop]
        grams = list(toks)
        if ngram == 2:
            grams += [a + " " + b for a, b in zip(toks, toks[1:])]
        counts = {}
        for g in grams:
            if g in vocab:
                counts[g] = counts.get(g, 0) + 1
        w = {g: (1 + math.log(c)) * vocab[g][0] for g, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return sum(v / norm * vocab[g][1] for g, v in w.items())

    def occupation(self, title):
        t = (title or "").lower()
        return next((n for n, rx in self.fam if rx.search(t)), None)

    def predict_salary(self, title, description="", job_type="", state="Other/Remote", occupation=None):
        s = self.m["salary_model"]
        y = s["intercept"]
        y += self._tfidf(title, s["title_vocab"], 2, False)
        y += self._tfidf(MONEY.sub(" ", description or ""), s["desc_vocab"], 1, True)
        y += s["cat"]["job_type"].get(job_type or "unknown", 0)
        y += s["cat"]["state"].get(state, 0)
        y += s["cat"]["occupation"].get(occupation or "Unmapped", 0)
        return float(math.exp(y))

    def score(self, posting, weights=None):
        """posting: dict with job_title, description, company, location or state, job_type, optional salary
        and optional overrides: culture_diversity, career_opportunities, work_life (1-5 stars)."""
        m = self.m
        occ = posting.get("occupation") or self.occupation(posting.get("job_title"))
        state = posting.get("state") or state_of(posting.get("location", ""))
        ref = m["occupations"].get(occ) or m["overall"]
        comp = m["companies"].get(norm_company(posting.get("company", "")))
        pred = self.predict_salary(posting.get("job_title"), posting.get("description", ""),
                                   posting.get("job_type"), state, occ)
        given = posting.get("salary")
        given = float(given) if given not in (None, "", "nan") and not pd.isna(given) else None
        vals = {"salary": given or pred, "job_openings": ref["job_openings"]}
        src = {"salary": "you entered it" if given else "predicted from posting text",
               "job_openings": f"postings for {occ}" if occ else "all-occupation median"}
        for k in ["culture_diversity", "career_opportunities", "promotion_likelihood", "work_life", "burnout_rate"]:
            if posting.get(k) not in (None, "") and not pd.isna(posting.get(k)):
                vals[k], src[k] = float(posting[k]), "you entered it"
            elif comp:
                w = comp["n"] / (comp["n"] + SHRINK_K)
                vals[k] = w * comp[k] + (1 - w) * ref[k]
                src[k] = f"{comp['name']} reviews ({comp['n']}) blended with occupation"
            else:
                vals[k], src[k] = ref[k], (f"{occ} average" if occ else "all-occupation average")
        weights = weights or {k: 1 for k, _, _ in COMPONENTS}
        pct = {}
        for k, _, sgn in COMPONENTS:
            r = np.array(m["reference"][k]) * sgn
            v = vals[k] * sgn
            pct[k] = float(((r < v).sum() + 0.5 * (r == v).sum()) / len(r))
        tot = sum(weights.values()) or 1
        score = 100 * sum(weights[k] * pct[k] for k in pct) / tot
        return {"growth_score": round(score, 1), "occupation": occ, "predicted_salary": round(pred),
                "company_matched": comp["name"] if comp else None,
                "values": vals, "percentiles": pct, "sources": src}


# ------------------------------------------------------------------ building the model
def build():
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
    sys.path.insert(0, str(HERE))
    src = (HERE / "02_model.py").read_text()
    exec(re.search(r"(PERIOD = .*?return float\(np.mean\(nums\[:2\]\)\) \* PERIOD\[unit\], unit\n)", src, re.S).group(1), globals())
    OUT = HERE / "outputs"; (HERE / "model").mkdir(exist_ok=True)
    metrics = {}

    # ---- salary model (compact enough to run in a browser)
    jobs = pd.read_parquet(OUT / "indeed_slim.parquet").drop_duplicates("jobid")
    jobs["occupation"] = jobs.job_title.map(map_title)
    jobs["state"] = jobs.location.map(state_of)
    jobs["salary"] = [parse_salary(s)[0] for s in jobs.salary_formatted]
    jobs.loc[~jobs.salary.between(15_000, 1_000_000), "salary"] = np.nan
    top_types = jobs.job_type.value_counts().index[:12]
    jobs["jt"] = jobs.job_type.where(jobs.job_type.isin(top_types), "unknown").fillna("unknown")
    jobs["desc"] = jobs.description_text.fillna("").map(lambda t: MONEY.sub(" ", t))
    sal = jobs[jobs.salary.notna()].copy()
    vt = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, token_pattern=r"\b\w\w+\b")
    vd = TfidfVectorizer(min_df=5, max_features=8000, sublinear_tf=True, stop_words="english", token_pattern=r"\b\w\w+\b")
    cats = ["jt", "state", "occupation"]
    def X(d, fit):
        parts = [vt.fit_transform(d.job_title) if fit else vt.transform(d.job_title),
                 vd.fit_transform(d.desc) if fit else vd.transform(d.desc)]
        for c in cats:
            vals = d[c].fillna("Unmapped")
            parts.append(sparse.csr_matrix(pd.get_dummies(vals).reindex(columns=levels[c], fill_value=0).values.astype(float)))
        return sparse.hstack(parts).tocsr()
    levels = {c: sorted(jobs[c].fillna("Unmapped").unique()) for c in cats}
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(sal, groups=sal.company_name))
    r = Ridge(alpha=1.0).fit(X(sal.iloc[tr], True), np.log(sal.salary.iloc[tr]))
    p = np.exp(r.predict(X(sal.iloc[te], False))); a = sal.salary.iloc[te].values
    metrics["salary_model"] = {"MedAPE_%": round(float(np.median(abs(p - a) / a) * 100), 1),
                               "MAE_$": round(float(mean_absolute_error(a, p))),
                               "R2_log": round(float(r2_score(np.log(a), np.log(p))), 3)}
    test_postings = sal.iloc[te].sample(400, random_state=1)
    print("salary (held-out companies):", metrics["salary_model"])
    r = Ridge(alpha=1.0).fit(X(sal, True), np.log(sal.salary))  # refit on everything for the final model
    coef = r.coef_; nt, nd = len(vt.vocabulary_), len(vd.vocabulary_)
    rnd = lambda x: float(f"{x:.5g}")
    sm = {"intercept": rnd(r.intercept_),
          "title_vocab": {t: [rnd(vt.idf_[i]), rnd(coef[i])] for t, i in vt.vocabulary_.items()},
          "desc_vocab": {t: [rnd(vd.idf_[i]), rnd(coef[nt + i])] for t, i in vd.vocabulary_.items()},
          "cat": {}}
    off = nt + nd
    for c, name in zip(cats, ["job_type", "state", "occupation"]):
        sm["cat"][name] = {lv: rnd(coef[off + j]) for j, lv in enumerate(levels[c])}
        off += len(levels[c])
    jobs["salary_best"] = jobs.salary.fillna(pd.Series(np.exp(r.predict(X(jobs, False))), index=jobs.index))

    # ---- review text models (burnout, promotion)
    RC = ["rating_work_life", "rating_culture_values", "rating_diversity_inclusion", "rating_career_opportunities"]
    rev = pd.read_csv(HERE.parent / "glassdoor_reviews.csv", low_memory=False,
                      usecols=RC + ["review_id", "company_name", "employee_job_title", "employee_location",
                                    "summary", "review_pros", "review_cons"]).drop_duplicates("review_id")
    rev[RC] = rev[RC].replace(0, np.nan)
    rev = rev[rev.employee_location.str.extract(r",\s*([A-Z]{2})$")[0].isin(US_STATES)].copy()
    rev["occupation"] = rev.employee_job_title.map(map_title)
    rev["text"] = rev.summary.fillna("") + " . PROS: " + rev.review_pros.fillna("") + " . CONS: " + rev.review_cons.fillna("")
    vr = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=150_000, sublinear_tf=True)
    XR = vr.fit_transform(rev.text)
    for name, col, fn in [("burnout_rate", "rating_work_life", lambda s: s <= 2),
                          ("promotion_likelihood", "rating_career_opportunities", lambda s: s >= 4)]:
        mask = rev[col].notna().values
        y = fn(rev.loc[mask, col]).astype(int).values
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(y, groups=rev.company_name[mask]))
        idx = np.where(mask)[0]
        clf = LogisticRegression(C=4.0, solver="liblinear").fit(XR[idx[tr]], y[tr])
        metrics[name + "_model_ROC_AUC"] = round(float(roc_auc_score(y[te], clf.predict_proba(XR[idx[te]])[:, 1])), 3)
        print(name, "AUC", metrics[name + "_model_ROC_AUC"])
        rev[name] = LogisticRegression(C=4.0, solver="liblinear").fit(XR[idx], y).predict_proba(XR)[:, 1]
    rev["culture_diversity"] = rev[["rating_culture_values", "rating_diversity_inclusion"]].mean(axis=1)
    rev = rev.rename(columns={"rating_career_opportunities": "career_opportunities", "rating_work_life": "work_life"})
    RK = ["culture_diversity", "career_opportunities", "promotion_likelihood", "work_life", "burnout_rate"]

    # ---- reference occupation table (same filters as the report: >=40 reviews and >=40 postings)
    gm = rev[RK].mean()
    def shrunk(d):
        return {k: float((d[k].sum() + SHRINK_K * gm[k]) / (d[k].count() + SHRINK_K)) for k in RK}
    occs = {}
    for o, d in rev[rev.occupation.notna() & (rev.occupation != "Intern")].groupby("occupation"):
        jp = jobs[jobs.occupation == o]
        if len(d) >= 40 and len(jp) >= 40:
            occs[o] = {**shrunk(d), "salary": float(jp.salary_best.median()), "job_openings": float(np.log(len(jp))),
                       "n_reviews": len(d), "n_postings": len(jp)}
    ref = {k: [v[k] for v in occs.values()] for k, _, _ in COMPONENTS}
    overall = {k: float(np.median(ref[k])) for k in ref}

    # ---- company table
    rev["ckey"] = rev.company_name.map(norm_company)
    comps = {}
    for c, d in rev.groupby("ckey"):
        if c and len(d) >= MIN_COMPANY_REVIEWS:
            comps[c] = {"name": d.company_name.mode().iloc[0], "n": len(d),
                        **{k: rnd(d[k].mean()) if d[k].notna().any() else rnd(gm[k]) for k in RK}}
    model = {"version": 1, "built_from": "glassdoor_reviews.csv + indeed_job_listings.csv",
             "occupation_patterns": FAMILIES, "stop_words": sorted(ENGLISH_STOP_WORDS),
             "salary_model": sm, "occupations": occs, "overall": overall, "reference": ref,
             "companies": comps, "job_types": list(top_types) + ["unknown"], "states": US_STATES + ["Other/Remote"],
             "components": COMPONENTS, "shrink_k": SHRINK_K, "metrics": metrics}
    json.dump(model, open(MODEL_PATH, "w"), separators=(",", ":"))
    print("wrote", MODEL_PATH, round(MODEL_PATH.stat().st_size / 1e6, 2), "MB;", len(occs), "occupations,", len(comps), "companies")

    # ---- checks on held-out postings: JSON scorer == sklearn, and how much predicted salary moves the score
    gsm = GrowthScoreModel(MODEL_PATH)
    rows = []
    for _, t in test_postings.iterrows():
        post = {"job_title": t.job_title, "description": t.description_text, "company": t.company_name,
                "state": t.state, "job_type": t.jt}
        with_real = gsm.score({**post, "salary": t.salary})
        with_pred = gsm.score(post)
        rows.append((with_real["growth_score"], with_pred["growth_score"], with_pred["company_matched"] is not None))
    rows = np.array(rows, dtype=float)
    sk = np.exp(r.predict(X(test_postings, False)))
    js = np.array([gsm.predict_salary(t.job_title, t.description_text, t.jt, t.state, t.occupation if isinstance(t.occupation, str) else None) for _, t in test_postings.iterrows()])
    metrics["json_vs_sklearn_max_rel_diff"] = float(np.max(np.abs(js - sk) / sk))
    metrics["score_check"] = {
        "n_held_out_postings": len(rows),
        "mean_abs_score_change_using_predicted_salary": round(float(np.abs(rows[:, 0] - rows[:, 1]).mean()), 2),
        "share_within_5_points": round(float((np.abs(rows[:, 0] - rows[:, 1]) <= 5).mean()), 3),
        "share_company_matched": round(float(rows[:, 2].mean()), 3)}
    model["metrics"] = metrics
    json.dump(model, open(MODEL_PATH, "w"), separators=(",", ":"))
    json.dump(metrics, open(OUT / "growth_model_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=1))


def score_csv(path):
    gsm = GrowthScoreModel()
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        r = gsm.score({k: (None if pd.isna(v) else v) for k, v in row.items()})
        out.append({"growth_score": r["growth_score"], "occupation": r["occupation"],
                    "company_matched": r["company_matched"], "salary_used": round(r["values"]["salary"]),
                    **{f"pct_{k}": round(100 * v) for k, v in r["percentiles"].items()}})
    res = pd.concat([df, pd.DataFrame(out)], axis=1).sort_values("growth_score", ascending=False)
    dest = Path(path).with_name(Path(path).stem + "_scored.csv")
    res.to_csv(dest, index=False)
    print(res[["job_title", "company", "growth_score", "occupation", "company_matched", "salary_used"]].to_string(index=False))
    print("wrote", dest)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    build() if cmd == "build" else score_csv(sys.argv[2])
