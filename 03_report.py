"""Builds report.html from outputs/metrics.json and outputs/occupation_scores.csv."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
m = json.load(open(HERE / "outputs" / "metrics.json"))
occ = pd.read_csv(HERE / "outputs" / "occupation_scores.csv", index_col="occupation")
cols = ["occupation", "growth_score", "score_ci_low", "score_ci_high", "median_salary_model", "culture_diversity",
        "career_opportunities", "n_postings", "log_postings", "promotion_likelihood", "work_life", "burnout_rate",
        "burnout_keyword_rate", "n_reviews", "demand_growth_pct", "typical_edu_years", "npv_20yr_premium", "education_cost",
        "roi_multiple", "share_hourly"]
from scipy.stats import spearmanr
nd = occ.typical_edu_years == 0
hp = occ[(occ.median_salary_model > 150_000) & (occ.growth_score < 35)]
bn = occ[nd].nlargest(3, "growth_score")
f = {"no_degree_share": float(occ[nd].n_postings.sum() / occ.n_postings.sum()),
     "score_no_degree": float(occ[nd].growth_score.mean()), "score_degree": float(occ[~nd].growth_score.mean()),
     "rho_hourly": float(spearmanr(occ.share_hourly, occ.growth_score)[0]),
     "rho_pay_burnout": float(spearmanr(occ.median_salary_model, occ.burnout_rate)[0]),
     "rho_pay_promo": float(spearmanr(occ.median_salary_model, occ.promotion_likelihood)[0]),
     "high_pay_low_growth": [{"occupation": o, "salary": float(r.median_salary_model), "score": float(r.growth_score)} for o, r in hp.iterrows()],
     "best_no_degree": [{"occupation": o, "score": float(r.growth_score)} for o, r in bn.iterrows()]}
data = {"occ": json.loads(occ.reset_index()[cols].round(4).to_json(orient="records")), "m": m, "f": f}
tpl = (HERE / "report_template.html").read_text()
(HERE / "report.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data)))
print("wrote", HERE / "report.html")
