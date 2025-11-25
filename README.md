# ml-project

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --debug
```

## Deploying to Vercel

1. Install the Vercel CLI and log in:
   ```bash
   npm i -g vercel
   vercel login
   ```
2. From the project root run the first deploy (creates a preview):
   ```bash
   vercel --prod
   ```
   The CLI will detect `app.py`, use the Python runtime defined in `vercel.json`, and upload `cost_model.joblib`, `model_features.joblib`, and the `templates/` directory automatically.
3. Subsequent deploys are as simple as `vercel --prod`. Use `vercel env pull` if you later add environment variables.

The `vercel.json` file routes all traffic to the Flask app and marks the template directory as static assets so the UI renders normally.
