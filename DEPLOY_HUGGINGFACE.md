# Hugging Face Spaces Deployment

## A. Prepare the deployment bundle locally

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m car_price.export_space
```

Verify:

```powershell
Get-ChildItem .\hf_space
Get-ChildItem .\hf_space\models
```

You should have:

```text
hf_space/
├── app.py
├── README.md
├── requirements.txt
├── model_metadata.json
└── models/
    └── car_price_pipeline.joblib
```

## B. Manual Hugging Face deployment

1. Create a Gradio Space in your Hugging Face account.
2. Install/login with the `hf` CLI:

```powershell
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

3. Clone your Space (replace the placeholder):

```powershell
cd C:\Projects
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/audi-car-price-predictor
cd audi-car-price-predictor
```

4. Copy the generated Space bundle into the cloned Space:

```powershell
Copy-Item `
  "C:\Projects\PRJ-CAR-PRICE-PREDICTION\hf_space\*" `
  "." `
  -Recurse `
  -Force
```

5. Commit and push:

```powershell
git add -A
git commit -m "deploy: add Audi price prediction Gradio app"
git push
```

Hugging Face rebuilds the Space when the commit is pushed.

## C. GitHub Actions deployment

Create a Hugging Face write token scoped to the Space.

In GitHub repository settings add:

- Secret: `HF_TOKEN`
- Repository variable: `HF_SPACE_ID`

Example value for `HF_SPACE_ID`:

```text
YOUR_HF_USERNAME/audi-car-price-predictor
```

Then open:

```text
GitHub → Actions → Deploy Hugging Face Space → Run workflow
```

The workflow trains/exports the deployment artifact and syncs only `hf_space/` to the Space.

## D. Local Gradio smoke test

After exporting the model:

```powershell
python -m pip install -e ".[app]"
python .\hf_space\app.py
```

Open the local URL printed by Gradio.
