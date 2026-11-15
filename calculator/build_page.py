"""Inlines scorer.js and model/growth_model.json into index.html (one self-contained page)."""
from pathlib import Path
H = Path(__file__).resolve().parent
page = (H / "page_template.html").read_text()
page = page.replace("/*__SCORER__*/", (H / "scorer.js").read_text())
page = page.replace("/*__MODEL__*/null", (H.parent / "model" / "growth_model.json").read_text())
import base64
for key, f in [("/*__LOGO__*/", "logo.png"), ("/*__WORDMARK__*/", "wordmark.png")]:
    page = page.replace(key, "data:image/png;base64," + base64.b64encode((H / "brand" / f).read_bytes()).decode())
(H / "index.html").write_text(page)
print("wrote", H / "index.html", round(len(page) / 1e6, 2), "MB")
