"""Financial ROI per occupation (runs after 02_model.py, overwrites the ROI columns).

Typical education = reference table of entry-level education per occupation family (EDU below).
Baseline pay = median posted pay across occupations classed as 'no degree'.
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from occupations import map_title
OUT = HERE / "outputs"
exec(re.search(r"(def edu_years.*?return 0\n)", (HERE / "02_model.py").read_text(), re.S).group(1))
exec(re.search(r"(PERIOD = .*?return float\(np.mean\(nums\[:2\]\)\) \* PERIOD\[unit\], unit\n)", (HERE / "02_model.py").read_text(), re.S).group(1))

j = pd.read_parquet(OUT / "indeed_slim.parquet", columns=["jobid", "job_title", "description_text", "salary_formatted"]).drop_duplicates("jobid")
j["occupation"] = j.job_title.map(map_title)
j["edu"] = j.description_text.map(edu_years)
j["sal"] = [parse_salary(s)[0] for s in j.salary_formatted]
j.loc[~j.sal.between(15_000, 1_000_000), "sal"] = np.nan
occ = pd.read_csv(OUT / "occupation_scores.csv", index_col="occupation")
# Typical entry-level education, years past high school (BLS "typical entry-level education" style).
# Posting-text extraction was tested and rejected: descriptions are truncated and requirements are often unstated.
EDU = {"Physician / Surgeon": 8, "Pharmacist": 8, "Physical / Occupational Therapist": 7, "Legal": 4,
       "Nurse Practitioner / PA": 6, "Data Scientist / ML": 6, "Scientist / Lab": 4, "Registered Nurse": 4,
       "Software Engineer": 4, "Engineer (non-software)": 4, "Data / BI Analyst": 4, "IT / Systems / Security": 2,
       "Product Manager": 4, "Designer (UX/Graphic)": 4, "Accountant / Auditor": 4, "Financial Analyst / Banking": 4,
       "Consultant": 4, "HR / Recruiter": 4, "Marketing / Communications": 4, "Account Executive / B2B Sales": 4,
       "Project / Program Manager": 4, "Business / Operations Analyst": 4, "Operations / General Manager": 4,
       "Teacher / Educator": 4, "Behavior Technician / Counselor": 2, "Dental": 2, "Medical Assistant / CNA": 1,
       "Pharmacy Technician": 1, "Real Estate / Property": 0, "Manager (general)": 0, "Administrative / Office": 0,
       "Customer Service": 0, "Retail Store Manager": 0, "Cashier / Retail Associate": 0, "Barista": 0,
       "Cook / Kitchen": 0, "Server / Bartender / Host": 0, "Housekeeping / Janitorial": 0, "Driver / Delivery": 0,
       "Warehouse / Logistics": 0, "Maintenance / Skilled Trades": 1, "Manufacturing / Production": 0,
       "Security Guard": 0, "Bank Teller / Loan Officer": 0}
missing = set(occ.index) - set(EDU); assert not missing, missing
occ["typical_edu_years"] = occ.index.map(EDU).astype(int)
occ["edu_stated_share"] = j[j.occupation.isin(occ.index)].groupby("occupation").edu.apply(lambda s: (s > 0).mean()).reindex(occ.index)
no_deg = occ.index[occ.typical_edu_years == 0]
BASE = float(j[j.occupation.isin(no_deg)].sal.median())
TUITION, DISC, HORIZON = 27_000, 0.03, 20
def npv(salary, edu):
    cost = sum((TUITION + BASE) / (1 + DISC) ** t for t in range(edu))
    gain = sum((salary - BASE) / (1 + DISC) ** t for t in range(edu, HORIZON))
    return gain - cost, cost
r = [npv(s, int(e)) for s, e in zip(occ.median_salary_model, occ.typical_edu_years)]
occ["npv_20yr_premium"] = [x[0] for x in r]; occ["education_cost"] = [x[1] for x in r]
occ["roi_multiple"] = np.where(occ.education_cost > 0, occ.npv_20yr_premium / occ.education_cost, np.nan)
occ.round(4).to_csv(OUT / "occupation_scores.csv")
m = json.load(open(OUT / "metrics.json"))
m["roi_assumptions"] = {"tuition_per_year": TUITION, "discount_rate": DISC, "horizon_years": HORIZON,
                        "no_degree_baseline_salary": round(BASE), "no_degree_occupations": list(no_deg),
                        "edu_rule": "reference table of typical entry-level education (years past high school)", "edu_table": EDU}
json.dump(m, open(OUT / "metrics.json", "w"), indent=2)
print("baseline", round(BASE))
print(occ[["typical_edu_years", "edu_stated_share", "median_salary_model", "npv_20yr_premium", "roi_multiple"]].round(2).sort_values("npv_20yr_premium").to_string())
