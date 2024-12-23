from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
import traceback

# Load the pre-trained model
model_path = "best_extra_trees_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Load the training data to use for SHAP explainer
X_train = pd.read_excel("./X_train.xlsx")
feature_names = X_train.columns.tolist()  # Dynamically fetch feature names
display(feature_names)

# Directory to save SHAP plots
shap_plot_dir = "static/shap_plots"
os.makedirs(shap_plot_dir, exist_ok=True)

# Function to make predictions and calculate probabilities
def make_prediction(model, features):
    features = np.array([features])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities


# Function to generate SHAP beeswarm plot and mean SHAP values
def generate_shap_plot_and_values(model, features, X_train, feature_names):
    features_df = pd.DataFrame([features], columns=feature_names)
    try:
        # Create an explainer using SHAP with the model's `predict_proba`
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(features_df)

        # Calculate mean absolute SHAP values
        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

        # Create DataFrame for SHAP values
        overall_shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute SHAP value": mean_abs_shap_values
        })

        # Sort the DataFrame based on 'Mean Absolute SHAP value' in descending order
        sorted_shap_df = overall_shap_df.sort_values(by="Mean Absolute SHAP value", ascending=False)

        # Convert to an HTML table for rendering
        shap_table_html = sorted_shap_df.to_html(classes="table table-striped", index=False)

        # Generate SHAP beeswarm plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values.values,
            features_df,
            feature_names=feature_names,
            show=False
        )
        plot_path = os.path.join(shap_plot_dir, "shap_beeswarm_plot.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        return plot_path, shap_table_html

    except Exception as e:
        print("Error during SHAP processing:", str(e))
        traceback.print_exc()
        return None, None
    

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        print("Received form data:", form_data)

        # Feature names
        feature_names = [
            "Host Sex", "Diarrhea", "Cough", "Running Nose", "Myalgia", 
            "Short Breath", "Nausea", "Headache", "Malaise", "Throat", 
            "Fever", "Recurrent Otitis Media", "TB", "Asthma", "Cancer", 
            "Cardiovascular Disease", "Neurologic Disorder", "Hepatobiliary Disease", 
            "Respiratory Disease", "Urogenital Disease", "Other Chronic Conditions", 
            "Chronic Lung Disease", "Immunosuppression", "Endocrine Metabolic Disorder", 
            "Host Age"
        ]

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
            float(form_data.get("host_age"))  # Ensure proper parsing of numerical input
        ]

        # Create a dictionary of entered values
        entered_values = {
            feature_names[i]: form_data.get(list(form_data.keys())[i]) 
            for i in range(len(feature_names))
        }

        if len(features) != 25:
            return "Error: Incorrect number of features provided", 400

        # Make prediction and get probabilities
        prediction, probabilities = make_prediction(model, features)
        print("Prediction and probabilities:", prediction, probabilities)

        # Generate SHAP plot and SHAP values table
        shap_plot_path, shap_table_html = generate_shap_plot_and_values(model, features, X_train, feature_names)

        if shap_plot_path is None or shap_table_html is None:
            return "Error generating SHAP values or plot.", 500

        print("SHAP plot and values generated")

        # Interpret prediction result
        result = "Positive" if prediction == 1 else "Negative"
        probability_positive = probabilities[1] * 100
        probability_negative = probabilities[0] * 100
        

        # Prepare prediction data for rendering
        prediction_data = {
            "prediction": result,
            "probability_positive": f"{probability_positive:.2f}%",
            "probability_negative": f"{probability_negative:.2f}%",
            "shap_plot_path": url_for("static", filename="shap_plots/shap_beeswarm_plot.png"),
            "shap_table_html": shap_table_html,
            "entered_values": entered_values  # Add the entered feature values to the data
        }

        return render_template("result.html", **prediction_data)

    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
