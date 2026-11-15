// Growth Score scorer - a line-for-line port of GrowthScoreModel.score() in growth_model.py.
// Works in the browser (window.GrowthScorer) and in Node (module.exports) for the parity test.
(function (root) {
  const TOKEN = /[\p{L}\p{N}\p{Mn}_]{2,}/gu;          // = Python's \b\w\w+\b
  const MONEY = /\$\s?[\d,]+(\.\d+)?[kK]?/g;
  const STATE_RX = /,\s*([A-Z]{2})\b/;
  const RK = ["culture_diversity", "career_opportunities", "promotion_likelihood", "work_life", "burnout_rate"];

  function normCompany(name) {
    if (!name) return "";
    let s = name.toLowerCase().replace(/&/g, " and ").replace(/[^a-z0-9 ]/g, " ");
    s = s.replace(/\b(inc|llc|l l c|corp|corporation|co|company|ltd|limited|plc|lp|llp|the|group|holdings)\b/g, " ");
    return s.replace(/\s+/g, " ").trim();
  }

  function create(model) {
    const fam = model.occupation_patterns.map(([n, p]) => [n, new RegExp(p)]);
    const stop = new Set(model.stop_words);
    const sm = model.salary_model;

    function tfidf(text, vocab, ngram, useStop) {
      let toks = (text || "").toLowerCase().match(TOKEN) || [];
      if (useStop) toks = toks.filter(t => !stop.has(t));
      const grams = toks.slice();
      if (ngram === 2) for (let i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
      const counts = new Map();
      for (const g of grams) if (vocab[g]) counts.set(g, (counts.get(g) || 0) + 1);
      let norm = 0; const w = [];
      for (const [g, c] of counts) { const v = (1 + Math.log(c)) * vocab[g][0]; w.push([g, v]); norm += v * v; }
      norm = Math.sqrt(norm) || 1;
      return w.reduce((a, [g, v]) => a + v / norm * vocab[g][1], 0);
    }
    function occupation(title) {
      const t = (title || "").toLowerCase();
      const hit = fam.find(([, rx]) => rx.test(t));
      return hit ? hit[0] : null;
    }
    function stateOf(loc) {
      const m = STATE_RX.exec(loc || "");
      return m && model.states.includes(m[1]) ? m[1] : "Other/Remote";
    }
    function predictSalary(title, description, jobType, state, occ) {
      let y = sm.intercept;
      y += tfidf(title, sm.title_vocab, 2, false);
      y += tfidf((description || "").replace(MONEY, " "), sm.desc_vocab, 1, true);
      y += sm.cat.job_type[jobType || "unknown"] || 0;
      y += sm.cat.state[state] || 0;
      y += sm.cat.occupation[occ || "Unmapped"] || 0;
      return Math.exp(y);
    }
    const blank = v => v === undefined || v === null || v === "" || Number.isNaN(v);

    function score(p, weights) {
      const occ = p.occupation || occupation(p.job_title);
      const state = p.state || stateOf(p.location);
      const ref = model.occupations[occ] || model.overall;
      const comp = model.companies[normCompany(p.company)];
      const pred = predictSalary(p.job_title, p.description, p.job_type, state, occ);
      const given = blank(p.salary) ? null : +p.salary;
      const values = { salary: given || pred, job_openings: ref.job_openings };
      const sources = { salary: given ? "you entered it" : "predicted from posting text",
                        job_openings: occ ? `postings for ${occ}` : "all-occupation median" };
      for (const k of RK) {
        if (!blank(p[k])) { values[k] = +p[k]; sources[k] = "you entered it"; }
        else if (comp) {
          const w = comp.n / (comp.n + model.shrink_k);
          values[k] = w * comp[k] + (1 - w) * ref[k];
          sources[k] = `${comp.name} reviews (${comp.n}) blended with occupation`;
        } else { values[k] = ref[k]; sources[k] = occ ? `${occ} average` : "all-occupation average"; }
      }
      weights = weights || Object.fromEntries(model.components.map(c => [c[0], 1]));
      const pct = {};
      for (const [k, , sgn] of model.components) {
        const v = values[k] * sgn; let below = 0, eq = 0;
        for (const r0 of model.reference[k]) { const r = r0 * sgn; if (r < v) below++; else if (r === v) eq++; }
        pct[k] = (below + 0.5 * eq) / model.reference[k].length;
      }
      const tot = Object.values(weights).reduce((a, b) => a + b, 0) || 1;
      const s = 100 * Object.keys(pct).reduce((a, k) => a + weights[k] * pct[k], 0) / tot;
      return { growth_score: Math.round(s * 10) / 10, occupation: occ, state, predicted_salary: Math.round(pred),
               company_matched: comp ? comp.name : null, values, percentiles: pct, sources };
    }
    return { score, occupation, predictSalary, stateOf, normCompany };
  }
  const api = { create, normCompany };
  if (typeof module !== "undefined") module.exports = api; else root.GrowthScorer = api;
})(typeof window !== "undefined" ? window : globalThis);
