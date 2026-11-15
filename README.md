# CareerMetric: Is your job worth it? 

*By Dorthy Victor, Emily Kay and Janvi Balakrishnan · Indeed jobs × Glassdoor reviews*

Which jobs give workers growth in their career?

**Cause:** economic mobility for workers without a college degree.
**Question:** using data on what employees say about their jobs (Glassdoor dataset)  and what employers offer (Indeed dataset), can we predict how much growth a job offers, and do the most accessible jobs offer the least amount of career growth?

**Findings:** yes. Occupations that typically need no degree make up about a third of job postings but average a growth score of ~34/100, versus ~60 for degree jobs. They pay less, show more burnout and offer fewer promotions.
The share of hourly postings is the strongest warning sign (Spearman ρ ≈ −0.74 with the growth score).

**Solution: Growth Score**
On a scale of 0 to 100 evaluate the potential career growth a job offers using the average of 7 different metrics:
- Salary 
- Culture and Diversity ratings 
- Career Opportunities provided 
- Job Openings 
- How likely are you to get promoted 
- Work-life balance score 
- Burnout rate (lower = better)

**Where to start**
* **[data_analysis.ipynb](data_analysis.ipynb)**: Analyzes both datasets, builds 3 different models to show that entry level jobs offer the least growth.
* **[Growth Score Calculator](https://janvibala.github.io/careermetric/calculator/)**: the CareerMetric growth score calculator (open it in a browser).


### Headline model results

| Model | Simple baseline | Strong baseline | Our model (text) | Text + stars |
|---|---|---|---|---|
| Salary (posting text → yearly pay), median % error | global median: 55% | occupation median: 24% | **16%** (R² 0.84) | n/a |
| Burnout (review text → work-life ≤ 2), ROC-AUC | keyword rule: 0.59 | overall stars only: 0.86 | **0.85** | **0.89** |
| Promotion (review text → career rating ≥ 4), ROC-AUC | keyword rule: 0.52 | overall stars only: 0.89 | **0.84** | **0.90** |

The text models far outperform keyword rules. The reviewer's overall star rating is a strong rival; combining text and
stars beats every baseline. The growth score uses the text-only models so the ratings it already includes aren't
counted twice (see notebook §3).

Exact numbers for your run are in `outputs/metrics.json`.

## Data sets utilized: 
Put these two files in the **parent** folder of `job_roi_model/`:

```
Datasets/
├── glassdoor_reviews.csv          (~139 MB, 175k reviews)
├── indeed_job_listings.csv        (~2.1 GB, 141k US postings)
└── job_roi_model/                 (this folder)
```

### Install
```bash
cd Datasets
python3 -m venv .venv
.venv/bin/pip install -r job_roi_model/requirements.txt
```
### Run
**Option A: my notebook (recommended, ~5 min)**
1. Open `data_analysis.ipynb` in VS Code or Jupyter.
2. Choose the `.venv` Python as the kernel.
3. Click **Run All**. It reads the two CSVs, builds all three models, and prints every result and chart.

**Option B: the full pipeline (~20 min)**, which also rebuilds the calculator. Run from the `Datasets` folder:
```bash
.venv/bin/python job_roi_model/01_extract_indeed.py     # shrink the 2 GB Indeed file
.venv/bin/python job_roi_model/02_model.py              # 3 models + occupation scores
.venv/bin/python job_roi_model/02b_roi.py               # 20-year financial ROI
.venv/bin/python job_roi_model/growth_model.py build    # package the calculator model
.venv/bin/python job_roi_model/calculator/build_page.py # rebuild the calculator web page
```
`analysis.ipynb` shows the same pipeline as one long notebook with extra charts.

### Score your own job postings
Fill in `my_postings_template.csv` (replace the example rows; only `job_title` is required), then:
```bash
.venv/bin/python job_roi_model/growth_model.py score job_roi_model/my_postings_template.csv
```
Results go to `my_postings_template_scored.csv`. The web calculator (`calculator/index.html`) uses the same model file
and can fill in its fields from pasted posting text.

---

## Files

| File | Overview |
|---|---|
| `data_analysis.ipynb` | Step-by-step analysis: data, joins, models, baselines, story and bias checks |
| `analysis.ipynb` | The full story: data, joins, models, evaluation, charts, bias, conclusions |
| `occupations.py` | Maps free-text job titles to 44 occupation families (the join key) |
| `01_extract_indeed.py` | Streams the 2 GB Indeed file and keeps 12 columns |
| `02_model.py` | Cleans both datasets, trains the salary/burnout/promotion models, evaluates them against baselines, builds occupation scores with bootstrap intervals |
| `02b_roi.py` | 20-year NPV earnings premium over a no-degree baseline |
| `growth_model.py` | The combined model: builds `model/growth_model.json` and scores any posting (`score` command) |
| `calculator/` | CareerMetric web calculator: `page_template.html` (design + logic), `scorer.js` (mirrors the Python scorer exactly), `parity_test.py` (checks they agree), `build_page.py` (bundles everything into `index.html`) |
| `calculator/brand/` | The title slide, plus `extract_brand.py`, which cuts out the logo and wordmark as transparent PNGs |
| `03_report.py`, `report_template.html` | Builds the interactive results report |
| `outputs/` | `metrics.json`, `occupation_scores.csv`, `growth_model_metrics.json`, `figures/` |

## Method notes
* Glassdoor sub-ratings of 0 mean "not rated" and are treated as missing; non-US reviews are dropped.
* Pay text is parsed and annualised (hour × 2,080, week × 52, month × 12, day × 260); values outside $15k–$1M are dropped.
* Dollar amounts are removed from descriptions before training the salary model, to prevent leakage.
* Occupation-level ratings are shrunk toward the overall mean (k = 20); occupations need ≥ 40 reviews and ≥ 40 postings.
* Growth score = mean percentile of 7 components (burnout reversed) × 100; 200 bootstrap resamples give 90% intervals.
* Random seeds are fixed (`random_state=42`), so reruns give the same numbers on the same data and package versions.

## Limitations and Bias
* The burnout model is least accurate for part-time workers (ROC-AUC 0.81 vs 0.85), the group most likely to hold no-degree jobs, and it over-predicts their burnout by 2–4 points.
* Hourly workers write fewer reviews, so their scores rest on fewer voices.
* The salary model is not worse for low-wage jobs (14% error for the lowest third).
* The score should help people widen their options, never screen people out.
