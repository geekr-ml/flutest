from flask import Flask, render_template, request
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Load the pre-trained model
model_path = "best_extra_trees_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Directory to save SHAP plots
shap_plot_dir = "static/shap_plots"
os.makedirs(shap_plot_dir, exist_ok=True)


# Function to make predictions and calculate probabilities
def make_prediction(model, features):
    features = np.array([features])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities


# Function to generate SHAP explanation plot
def generate_shap_plot(model, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values[1], features, plot_type="bar", show=False)
    plot_path = os.path.join(shap_plot_dir, "shap_summary_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form

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

        # Ensure correct feature length
        if len(features) != 25:
            return "Error: Incorrect number of features provided", 400

        # Make prediction and get probabilities
        prediction, probabilities = make_prediction(model, features)

        # Generate SHAP plot
        features_df = np.array([features])
        shap_plot_path = generate_shap_plot(model, features_df)

        # Interpret prediction result
        result = "Positive" if prediction == 1 else "Negative"
        probability_positive = probabilities[1] * 100
        probability_negative = probabilities[0] * 100

        return render_template(
            "result.html",
            prediction=result,
            probability_positive=f"{probability_positive:.2f}%",
            probability_negative=f"{probability_negative:.2f}%",
            shap_plot_path=shap_plot_path,
        )

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
