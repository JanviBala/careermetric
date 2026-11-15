"""Scores the same postings in Python and in Node (scorer.js) and compares results."""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent))
from growth_model import GrowthScoreModel, MODEL_PATH, state_of
d = pd.read_parquet(H.parent / "outputs" / "indeed_slim.parquet").sample(300, random_state=7)
posts = [{"job_title": r.job_title, "description": r.description_text, "company": r.company_name,
          "state": state_of(r.location), "job_type": r.job_type if isinstance(r.job_type, str) else "unknown"} for r in d.itertuples()]
g = GrowthScoreModel()
py = [g.score(p) for p in posts]
js_code = f"""
const S=require({json.dumps(str(H/'scorer.js'))}); const m=require({json.dumps(str(MODEL_PATH))});
const G=S.create(m); const posts=JSON.parse(require('fs').readFileSync(0,'utf8'));
console.log(JSON.stringify(posts.map(p=>G.score(p))));"""
js = json.loads(subprocess.run(["node", "-e", js_code], input=json.dumps(posts), capture_output=True, text=True, check=True).stdout)
ds = [abs(a["growth_score"] - b["growth_score"]) for a, b in zip(py, js)]
dsal = [abs(a["predicted_salary"] - b["predicted_salary"]) / a["predicted_salary"] for a, b in zip(py, js)]
occ = sum(a["occupation"] == b["occupation"] for a, b in zip(py, js)) / len(py)
co = sum(a["company_matched"] == b["company_matched"] for a, b in zip(py, js)) / len(py)
print(f"n={len(py)}  max score diff={max(ds):.3f}  max salary rel diff={max(dsal):.5f}  occupation agree={occ:.3f}  company agree={co:.3f}")
