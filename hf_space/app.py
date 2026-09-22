from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from gradio.themes.base import Base

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_price_pipeline.joblib"
METADATA_PATH = ROOT / "model_metadata.json"

GITHUB_URL = "https://github.com/Parmodk2310/PRJ-CAR-PRICE-PREDICTION"
PORTFOLIO_URL = "https://parmodk2310.vercel.app"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Missing models/car_price_pipeline.joblib. "
        "Run `python -m car_price.export_space` from the repo first."
    )

if not METADATA_PATH.exists():
    raise RuntimeError(
        "Missing model_metadata.json. Run `python -m car_price.export_space` from the repo first."
    )

pipeline = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

evaluation = metadata.get("evaluation", {})
training_rows = int(metadata.get("training_rows", 0))
model_name = metadata.get("model_name", "Random Forest")
model_version = metadata.get("model_version", "v0.2")
feature_count = int(metadata.get("feature_count", 8))


def choice_pairs(column: str):
    return [(item["label"], item["value"]) for item in metadata["categorical"][column]]


def numeric(column: str):
    return metadata["numerical"][column]


def metric_value(name: str, fallback: str) -> str:
    value = evaluation.get(name)
    if value is None:
        return fallback
    if name == "r2":
        return f"{float(value):.4f}"
    return f"£{float(value):,.0f}"


def render_waiting() -> str:
    return """
    <div class="result-shell waiting">
      <div class="result-chip">Prediction output</div>
      <div class="result-icon">◇</div>
      <div class="result-title">Awaiting vehicle details</div>
      <div class="result-subtitle">
        Complete the profile and click <strong>Estimate vehicle price</strong>.
      </div>
      <div class="result-meta-row">
        <div class="result-meta-box">
          <span>Artifact</span>
          <strong>Persisted pipeline</strong>
        </div>
        <div class="result-meta-box">
          <span>Inference</span>
          <strong>Realtime</strong>
        </div>
      </div>
    </div>
    """


def render_result(price: float, vehicle: str, year: int, mileage: float) -> str:
    return f"""
    <div class="result-shell success">
      <div class="result-chip">Model estimate</div>
      <div class="result-icon">✓</div>
      <div class="result-price">£{price:,.0f}</div>
      <div class="result-title">{html.escape(vehicle.strip())} · {int(year)}</div>
      <div class="result-subtitle">
        Estimated from {float(mileage):,.0f} miles and the supplied structured vehicle attributes.
      </div>
      <div class="result-meta-row">
        <div class="result-meta-box">
          <span>Model</span>
          <strong>{html.escape(model_name)}</strong>
        </div>
        <div class="result-meta-box">
          <span>Version</span>
          <strong>{html.escape(model_version)}</strong>
        </div>
      </div>
    </div>
    """


def estimate_price(model, year, transmission, mileage, fuel_type, tax, mpg, engine_size):
    if not model or not transmission or not fuel_type:
        raise gr.Error("Select model, transmission, and fuel type before estimating.")

    row = pd.DataFrame(
        [
            {
                "model": model,
                "year": int(year),
                "transmission": transmission,
                "mileage": float(mileage),
                "fuelType": fuel_type,
                "tax": float(tax),
                "mpg": float(mpg),
                "engineSize": float(engine_size),
            }
        ]
    )

    prediction = float(pipeline.predict(row)[0])

    summary = pd.DataFrame(
        [
            ["Model", str(model).strip()],
            ["Year", str(int(year))],
            ["Transmission", str(transmission)],
            ["Fuel type", str(fuel_type)],
            ["Mileage", f"{float(mileage):,.0f} mi"],
            ["Engine size", f"{float(engine_size):.1f} L"],
            ["MPG", f"{float(mpg):.1f}"],
            ["Road tax", f"£{float(tax):,.0f}"],
        ],
        columns=["Attribute", "Value"],
    )

    return render_result(prediction, str(model), int(year), float(mileage)), summary


year_cfg = numeric("year")
mileage_cfg = numeric("mileage")
tax_cfg = numeric("tax")
mpg_cfg = numeric("mpg")
engine_cfg = numeric("engineSize")

model_choices = choice_pairs("model")
transmission_choices = choice_pairs("transmission")
fuel_choices = choice_pairs("fuelType")

CUSTOM_CSS = """
:root {
  --bg: #07101f;
  --panel: #0d1728;
  --panel-soft: rgba(13, 23, 40, .82);
  --border: #20304a;
  --muted: #8ea0bc;
  --text: #eef4ff;
  --blue: #4f8ff7;
  --blue-strong: #2c6df3;
  --green: #1fd29a;
}

html, body {
  background: var(--bg) !important;
}

.gradio-container {
  max-width: 1560px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  background:
    radial-gradient(circle at 82% 0%, rgba(44, 109, 243, .11), transparent 28%),
    radial-gradient(circle at 10% 0%, rgba(79, 143, 247, .07), transparent 22%),
    var(--bg) !important;
  color: var(--text) !important;
  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif !important;
}

footer {
  display: none !important;
}

#app-shell {
  padding: 22px 24px 28px;
}

.hero {
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(14, 24, 42, .98), rgba(8, 16, 30, .98));
  border-radius: 18px;
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: 0 20px 48px rgba(0, 0, 0, .20);
}

.hero-top {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.hero-copy {
  flex: 1;
}

.hero-eyebrow {
  color: #78a7ff;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.hero h1 {
  margin: 0;
  font-size: 30px;
  line-height: 1.12;
  font-weight: 780;
  color: var(--text);
}

.hero p {
  margin: 10px 0 0;
  max-width: 860px;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.55;
}

.hero-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 10px;
}

.badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.badge {
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, .03);
  color: #d6e4fb;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  padding: 8px 12px;
}

.badge.live {
  background: rgba(31, 210, 154, .10);
  border-color: rgba(31, 210, 154, .38);
  color: #a7f3d0;
}

.hero-links {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.hero-link {
  text-decoration: none !important;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, .03);
  color: #dce7fb !important;
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 700;
}

.value-strip {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.value-box {
  border: 1px solid var(--border);
  background: rgba(8, 16, 30, .52);
  border-radius: 12px;
  padding: 12px 14px;
}

.value-box span {
  display: block;
  color: #7f93b1;
  font-size: 10px;
  letter-spacing: .10em;
  text-transform: uppercase;
  font-weight: 800;
  margin-bottom: 5px;
}

.value-box strong {
  color: var(--text);
  font-size: 13px;
  font-weight: 760;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.metric-card {
  border: 1px solid var(--border);
  background: rgba(12, 21, 38, .92);
  border-radius: 16px;
  padding: 16px;
}

.metric-label {
  color: #8090ab;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .10em;
  text-transform: uppercase;
}

.metric-value {
  color: var(--text);
  font-size: 24px;
  font-weight: 800;
  margin-top: 7px;
}

.metric-sub {
  color: #687a96;
  font-size: 11px;
  margin-top: 4px;
}

.panel {
  border: 1px solid var(--border);
  background: var(--panel-soft);
  border-radius: 16px;
  padding: 16px !important;
  height: 100%;
  box-sizing: border-box;
}

.section-title {
  color: var(--blue);
  font-size: 11px;
  font-weight: 850;
  letter-spacing: .14em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin: 0 0 12px;
}

.gradio-container label span {
  color: #99abc8 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container [role="listbox"] {
  border-color: #263754 !important;
}

#estimate-button {
  margin-top: 8px;
  border: none !important;
  border-radius: 11px !important;
  min-height: 46px;
  color: white !important;
  font-weight: 760 !important;
  background: linear-gradient(90deg, var(--blue-strong), var(--blue)) !important;
  box-shadow: 0 10px 24px rgba(44, 109, 243, .24);
}

.project-card-title {
  font-size: 20px;
  font-weight: 780;
  color: var(--text);
  margin-bottom: 5px;
}

.project-card-sub {
  color: #7890b0;
  font-size: 13px;
  line-height: 1.5;
  margin-bottom: 14px;
}

.kv-section {
  border-top: 1px solid var(--border);
  margin-top: 12px;
  padding-top: 12px;
}

.kv-kicker {
  color: var(--blue);
  font-size: 10px;
  font-weight: 850;
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.kv-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  margin: 7px 0;
}

.kv-row span:first-child {
  color: #7e90ac;
}

.kv-row span:last-child {
  color: #edf4ff;
  font-weight: 670;
  text-align: right;
  max-width: 64%;
}

.link-row {
  margin-top: 10px;
}

.link-row a {
  color: #62a0ff !important;
  text-decoration: none !important;
  font-size: 12px;
  margin-right: 14px;
}

.result-shell {
  border-radius: 16px;
  padding: 20px 18px;
  min-height: 290px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
}

.result-shell.waiting {
  border: 1px dashed #304563;
  background: rgba(8, 16, 30, .55);
}

.result-shell.success {
  border: 1px solid rgba(31, 210, 154, .70);
  background: linear-gradient(145deg, rgba(5, 54, 40, .86), rgba(7, 26, 24, .78));
}

.result-chip {
  align-self: center;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, .03);
  color: #9cb2d0;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 760;
  padding: 7px 10px;
  margin-bottom: 12px;
}

.result-icon {
  font-size: 34px;
  line-height: 1;
  color: var(--green);
  margin-bottom: 10px;
}

.waiting .result-icon {
  color: #5a8fe4;
}

.result-price {
  font-size: 42px;
  font-weight: 820;
  color: var(--green);
  letter-spacing: -.03em;
  margin-bottom: 6px;
}

.result-title {
  color: var(--text);
  font-size: 20px;
  font-weight: 760;
}

.result-subtitle {
  color: #93a5bf;
  font-size: 13px;
  line-height: 1.55;
  margin-top: 8px;
}

.result-meta-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 16px;
}

.result-meta-box {
  border: 1px solid rgba(255, 255, 255, .09);
  border-radius: 11px;
  background: rgba(255, 255, 255, .03);
  padding: 10px;
  text-align: left;
}

.result-meta-box span {
  display: block;
  color: #82a2d8;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 4px;
}

.result-meta-box strong {
  color: white;
  font-size: 12px;
  font-weight: 760;
}

.info-list {
  color: #93a5bf;
  font-size: 13px;
  line-height: 1.6;
}

.info-list strong {
  color: #eef4ff;
}

.badge-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.badge-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: rgba(8, 16, 30, .50);
}

.badge-card span {
  display: block;
  color: #7d8fae;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 5px;
}

.badge-card strong {
  color: var(--text);
  font-size: 12px;
  font-weight: 760;
}

#vehicle-summary table {
  font-size: 12px !important;
}

.footer-shell {
  border-top: 1px solid var(--border);
  margin-top: 18px;
  padding-top: 14px;
  display: flex;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
  color: #7f90aa;
  font-size: 11px;
}

.footer-links {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.footer-links a,
.footer-links span {
  color: #7f90aa !important;
  text-decoration: none !important;
}

@media (max-width: 1200px) {
  .value-strip, .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .hero-top {
    flex-direction: column;
  }

  .hero-side,
  .badges {
    align-items: flex-start;
    justify-content: flex-start;
  }
}

@media (max-width: 768px) {
  .value-strip,
  .metric-grid,
  .badge-grid,
  .result-meta-row {
    grid-template-columns: 1fr;
  }

  #app-shell {
    padding: 14px;
  }
}
"""

hero_html = f"""
<div class="hero">
  <div class="hero-top">
    <div class="hero-copy">
      <div class="hero-eyebrow">Applied Machine Learning · Recruiter-ready UI</div>
      <h1>Audi Price Intelligence</h1>
      <p>
        Professional ML portfolio demo for estimating used Audi resale prices.
        This interface presents business context, engineering evidence, model
        evaluation, and deployable inference in one dense dashboard without
        wasted layout space.
      </p>
    </div>

    <div class="hero-side">
      <div class="badges">
        <span class="badge live">● Model online</span>
        <span class="badge">{html.escape(model_name)} · {html.escape(model_version)}</span>
        <span class="badge">Python 3.11</span>
      </div>
      <div class="hero-links">
        <a class="hero-link" href="{GITHUB_URL}" target="_blank">GitHub ↗</a>
        <a class="hero-link" href="{PORTFOLIO_URL}" target="_blank">Portfolio ↗</a>
      </div>
    </div>
  </div>

  <div class="value-strip">
    <div class="value-box">
      <span>Use case</span>
      <strong>Used-car benchmark pricing</strong>
    </div>
    <div class="value-box">
      <span>Artifact</span>
      <strong>Persisted sklearn pipeline</strong>
    </div>
    <div class="value-box">
      <span>Validation</span>
      <strong>80 / 20 split + 5-fold CV</strong>
    </div>
    <div class="value-box">
      <span>Inputs</span>
      <strong>{feature_count} structured vehicle features</strong>
    </div>
  </div>
</div>
"""

metric_html = f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Held-out R²</div>
    <div class="metric-value">{metric_value("r2", "0.9654")}</div>
    <div class="metric-sub">Generalization on reserved test split</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mean absolute error</div>
    <div class="metric-value">{metric_value("mae", "£1,517")}</div>
    <div class="metric-sub">Average absolute prediction gap</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RMSE</div>
    <div class="metric-value">{metric_value("rmse", "£2,286")}</div>
    <div class="metric-sub">Penalizes larger estimation mistakes</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Training records</div>
    <div class="metric-value">{training_rows:,}</div>
    <div class="metric-sub">Historical Audi listings used for training</div>
  </div>
</div>
"""

summary_html = f"""
<div class="project-card-title">Project brief</div>
<div class="project-card-sub">
  Portfolio-ready regression system with a deployable preprocessing + model artifact.
</div>

<div class="kv-section">
  <div class="kv-kicker">Business context</div>
  <div class="kv-row"><span>Problem</span><span>Estimate used Audi resale price</span></div>
  <div class="kv-row"><span>Users</span><span>Recruiters · engineers · reviewers</span></div>
  <div class="kv-row"><span>Dataset</span><span>Historical Audi listings</span></div>
  <div class="kv-row"><span>Objective</span><span>Reliable benchmark estimate</span></div>
</div>

<div class="kv-section">
  <div class="kv-kicker">ML system</div>
  <div class="kv-row"><span>Algorithm</span><span>{html.escape(model_name)}</span></div>
  <div class="kv-row"><span>Pipeline</span><span>ColumnTransformer + estimator</span></div>
  <div class="kv-row"><span>Unknowns</span><span>Handled by encoder</span></div>
  <div class="kv-row"><span>Deployment</span><span>Hugging Face Space</span></div>
</div>

<div class="kv-section">
  <div class="kv-kicker">Repository evidence</div>
  <div class="kv-row"><span>Tests</span><span>pytest coverage</span></div>
  <div class="kv-row"><span>CI</span><span>GitHub Actions</span></div>
  <div class="kv-row"><span>Diagnostics</span><span>Residual and error analysis</span></div>
  <div class="kv-row"><span>Packaging</span><span>Reusable inference pipeline</span></div>
</div>

<div class="kv-section">
  <div class="kv-kicker">Built by</div>
  <div class="kv-row"><span>Engineer</span><span>Parmod</span></div>
  <div class="kv-row"><span>Focus</span><span>AI / ML Engineering</span></div>
  <div class="link-row">
    <a href="{GITHUB_URL}" target="_blank">GitHub ↗</a>
    <a href="{PORTFOLIO_URL}" target="_blank">Portfolio ↗</a>
  </div>
</div>
"""

guide_html = """
<div class="info-list">
  <strong>How to read the estimate</strong><br>
  The prediction is a benchmark based on vehicles with similar structured
  characteristics in the training data.<br><br>

  <strong>Good use cases</strong><br>
  Portfolio demonstration, regression evaluation, baseline pricing estimate,
  and ML system presentation.<br><br>

  <strong>Where error increases</strong><br>
  Rare trims, unusual vehicle condition, modifications, service history gaps,
  region-specific pricing, and examples far outside the historical distribution.
</div>
"""

evidence_html = """
<div class="info-list">
  <strong>Leakage control</strong><br>
  Preprocessing is fit inside the training pipeline, not on the full dataset
  before evaluation.<br><br>

  <strong>Evaluation discipline</strong><br>
  Candidate models are compared with cross-validation on the training partition;
  the held-out split is reserved for final baseline reporting.<br><br>

  <strong>Deployment behavior</strong><br>
  The app loads one persisted preprocessing + estimator artifact and performs
  inference only — no retraining per request.
</div>
"""

readiness_html = """
<div class="badge-grid">
  <div class="badge-card">
    <span>Leakage control</span>
    <strong>Pipeline-fitted preprocessing</strong>
  </div>
  <div class="badge-card">
    <span>Unknown values</span>
    <strong>Encoder handles unseen categories</strong>
  </div>
  <div class="badge-card">
    <span>Reproducibility</span>
    <strong>Fixed random state</strong>
  </div>
  <div class="badge-card">
    <span>Delivery</span>
    <strong>CI + deployable Space bundle</strong>
  </div>
  <div class="badge-card">
    <span>Training</span>
    <strong>{training_rows:,} source records</strong>
  </div>
  <div class="badge-card">
    <span>Model family</span>
    <strong>{model_name}</strong>
  </div>
</div>
"""

footer_html = f"""
<div class="footer-shell">
  <div>Audi Price Intelligence · Built by Parmod</div>
  <div class="footer-links">
    <span>Leakage-safe preprocessing</span>
    <span>Held-out evaluation</span>
    <span>Deployable inference</span>
    <a href="{GITHUB_URL}" target="_blank">GitHub ↗</a>
  </div>
</div>
"""

initial_summary = pd.DataFrame(
    [
        ["Status", "Waiting for prediction"],
        ["Model", "Select a vehicle profile"],
        ["Output", "Estimated resale price"],
    ],
    columns=["Attribute", "Value"],
)

with gr.Blocks(
    title="Audi Price Intelligence",
) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(hero_html)
        gr.HTML(metric_html)

        # Top main row: no wasted gutter
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=280), gr.Column(elem_classes=["panel"]):
                gr.HTML(summary_html)

            with gr.Column(scale=4, min_width=420), gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Vehicle input</div>')

                with gr.Row():
                    model = gr.Dropdown(
                        choices=model_choices,
                        label="Audi model",
                        value=model_choices[0][1],
                    )
                    year = gr.Slider(
                        minimum=int(year_cfg["min"]),
                        maximum=int(year_cfg["max"]),
                        value=int(year_cfg["default"]),
                        step=1,
                        label="Year",
                    )

                with gr.Row():
                    transmission = gr.Dropdown(
                        choices=transmission_choices,
                        label="Transmission",
                        value=transmission_choices[0][1],
                    )
                    fuel_type = gr.Dropdown(
                        choices=fuel_choices,
                        label="Fuel type",
                        value=fuel_choices[0][1],
                    )

                with gr.Row():
                    mileage = gr.Number(
                        value=mileage_cfg["default"],
                        label="Mileage",
                    )
                    engine_size = gr.Number(
                        value=engine_cfg["default"],
                        label="Engine size (L)",
                    )

                with gr.Row():
                    mpg = gr.Number(
                        value=mpg_cfg["default"],
                        label="MPG",
                    )
                    tax = gr.Number(
                        value=tax_cfg["default"],
                        label="Road tax (£)",
                    )

                estimate_button = gr.Button(
                    "Estimate vehicle price",
                    variant="primary",
                    elem_id="estimate-button",
                )

            with gr.Column(scale=3, min_width=360), gr.Column(elem_classes=["panel"]):
                result_panel = gr.HTML(render_waiting())

        # Full-width lower knowledge grid — this is what removes the old empty area.
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=330), gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Vehicle summary</div>')
                vehicle_summary = gr.Dataframe(
                    value=initial_summary,
                    headers=["Attribute", "Value"],
                    datatype=["str", "str"],
                    interactive=False,
                    wrap=True,
                    elem_id="vehicle-summary",
                )

            with gr.Column(scale=3, min_width=330), gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Prediction guide</div>')
                gr.HTML(guide_html)

            with gr.Column(scale=3, min_width=330), gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Engineering evidence</div>')
                gr.HTML(evidence_html)

            with gr.Column(scale=3, min_width=330), gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">Production-readiness evidence</div>')
                gr.HTML(readiness_html)

        gr.HTML(footer_html)

    estimate_button.click(
        fn=estimate_price,
        inputs=[model, year, transmission, mileage, fuel_type, tax, mpg, engine_size],
        outputs=[result_panel, vehicle_summary],
    )

if __name__ == "__main__":
    demo.launch(
        theme=Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
