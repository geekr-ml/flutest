from flask import Flask
import pickle
import numpy as np

from flask import Blueprint, render_template, request
from model import load_model, make_prediction

model_path = best_extra_trees_model.pkl

with open(model_path, "rb") as f:
        model = pickle.load(f)

app = Flask(__name__)


def make_prediction(model, features):
    features = np.array([features])
    prediction = model.predict(features)[0]
    return prediction

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        print("Received Form Data:", form_data)  # Log form data for debugging

        # Parse inputs into a list for prediction
        features = [
            int(form_data.get("host_sex") == "Female"),
            int(form_data.get("diarrhea")),
            int(form_data.get("cough")),
            int(form_data.get("running_nose")),
            int(form_data.get("myalgia")),
            int(form_data.get("short_breath")),
            int(form_data.get("nausea")),
            int(form_data.get("headache")),
            int(form_data.get("malaise")),
            int(form_data.get("throat")),
            int(form_data.get("fever")),
            int(form_data.get("recurrent_otitis_media")),
            int(form_data.get("tb")),
            int(form_data.get("asthma")),
            int(form_data.get("cancer")),
            int(form_data.get("cardiovascular_disease")),
            int(form_data.get("neurologic_disorder")),
            int(form_data.get("hepatobiliary_disease")),
            int(form_data.get("respiratory_disease")),
            int(form_data.get("urogenital_disease")),
            int(form_data.get("other_chronic_conditions")),
            int(form_data.get("chronic_lung_disease")),
            int(form_data.get("immunosuppression")),
            int(form_data.get("endocrine_metabolic_disorder")),
            int(form_data.get("host_age"))
        ]

        # Check feature length
        print("Parsed Features:", features)

        if len(features) != 25:
            return "Error: Incorrect number of features provided", 400

        # Get prediction
        prediction = make_prediction(model, features)

        if prediction==0:
            res = "Negative"
        else:
            res = "Positive"
            
        return render_template("result.html", prediction=res)

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)