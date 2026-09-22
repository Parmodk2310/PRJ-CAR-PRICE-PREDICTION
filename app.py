from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_price_pipeline.joblib"
METADATA_PATH = ROOT / "model_metadata.json"


if not MODEL_PATH.exists():
    raise RuntimeError(
        "Deployment model is missing. Run `python -m car_price.export_space` "
        "before publishing the hf_space directory."
    )

if not METADATA_PATH.exists():
    raise RuntimeError(
        "model_metadata.json is missing. Run `python -m car_price.export_space` "
        "before publishing the hf_space directory."
    )


pipeline = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def choice_pairs(column: str):
    return [
        (item["label"], item["value"])
        for item in metadata["categorical"][column]
    ]


def numeric(column: str):
    return metadata["numerical"][column]


def estimate_price(
    model,
    year,
    transmission,
    mileage,
    fuel_type,
    tax,
    mpg,
    engine_size,
):
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
    return f"£{prediction:,.0f}", prediction


year_cfg = numeric("year")
mileage_cfg = numeric("mileage")
tax_cfg = numeric("tax")
mpg_cfg = numeric("mpg")
engine_cfg = numeric("engineSize")

with gr.Blocks(title="Audi Used-Car Price Predictor") as demo:
    gr.Markdown(
        """
        # 🚗 Audi Used-Car Price Predictor
        Estimate a used Audi's price from vehicle attributes using a reproducible
        scikit-learn Random Forest pipeline.

        **Portfolio demo only:** this is a machine-learning estimate, not a valuation,
        offer, or financial recommendation.
        """
    )

    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(
                choices=choice_pairs("model"),
                label="Audi model",
            )
            year = gr.Slider(
                minimum=int(year_cfg["min"]),
                maximum=int(year_cfg["max"]),
                value=int(year_cfg["default"]),
                step=1,
                label="Year",
            )
            transmission = gr.Dropdown(
                choices=choice_pairs("transmission"),
                label="Transmission",
            )
            fuel_type = gr.Dropdown(
                choices=choice_pairs("fuelType"),
                label="Fuel type",
            )

        with gr.Column():
            mileage = gr.Number(
                value=mileage_cfg["default"],
                label="Mileage",
            )
            tax = gr.Number(
                value=tax_cfg["default"],
                label="Tax (£)",
            )
            mpg = gr.Number(
                value=mpg_cfg["default"],
                label="MPG",
            )
            engine_size = gr.Number(
                value=engine_cfg["default"],
                label="Engine size (L)",
            )

    predict_button = gr.Button("Estimate price", variant="primary")

    formatted_price = gr.Textbox(
        label="Estimated price",
        interactive=False,
    )
    raw_prediction = gr.Number(
        label="Raw prediction (£)",
        visible=False,
    )

    predict_button.click(
        fn=estimate_price,
        inputs=[
            model,
            year,
            transmission,
            mileage,
            fuel_type,
            tax,
            mpg,
            engine_size,
        ],
        outputs=[formatted_price, raw_prediction],
    )

    gr.Markdown(
        f"""
        **Training rows:** {metadata["training_rows"]:,}

        The public demo loads a pre-trained pipeline; it does not retrain on each request.
        """
    )


if __name__ == "__main__":
    demo.launch()
