from flask import Flask, render_template, request
import joblib
import pandas as pd
import re

app = Flask(__name__)

# Load model + features
model = joblib.load("cost_model.joblib")
feature_cols = joblib.load("model_features.joblib")

# City-based feature filters
sf_keywords = ["street", "block", "lot", "zip", "supervisor"]
chicago_keywords = ["ward", "latitude", "longitude", "community", "census"]

sf_features = [f for f in feature_cols if any(k in f.lower() for k in sf_keywords)]
chicago_features = [f for f in feature_cols if any(k in f.lower() for k in chicago_keywords)]

# Common features
common_features = list(set(feature_cols) - set(sf_features) - set(chicago_features))

def pretty(text):
    return re.sub(r"[_\-]+", " ", text).title()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    selected_city = request.form.get("city", "")

    # Choose fields to show
    if selected_city == "san_francisco":
        active_features = sf_features + common_features
    elif selected_city == "chicago":
        active_features = chicago_features + common_features
    else:
        active_features = []

    # Predict
    if request.method == "POST" and selected_city:
        data = {}

        for col in active_features:
            val = request.form.get(col, "")

            if val == "":
                data[col] = 0
            else:
                try:
                    data[col] = float(val)
                except:
                    data[col] = 0

        # Add missing features
        for col in feature_cols:
            if col not in data:
                data[col] = 0

        df = pd.DataFrame([data], columns=feature_cols)
        prediction = float(model.predict(df)[0])

    return render_template(
        "index.html",
        cities=["san_francisco", "chicago"],
        selected_city=selected_city,
        features=active_features,
        pretty=pretty,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
