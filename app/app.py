from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

from app.schemas import PlacementInput


# initialize Flask app
app = Flask(__name__)


# load artifacts
model = joblib.load("artifacts/models/final_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
target_encoder = joblib.load("artifacts/target_encoder.pkl")


# Prediction logic
def make_prediction(input_data: PlacementInput):
    data = {
        "CGPA": input_data.CGPA,
        "Internships": input_data.Internships,
        "Projects": input_data.Projects,
        "Workshops/Certifications": input_data.Workshops,
        "AptitudeTestScore": input_data.AptitudeTestScore,
        "SoftSkillsRating": input_data.SoftSkillsRating,
        "SSC_Marks": input_data.SSC_Marks,
        "HSC_Marks": input_data.HSC_Marks,
    }

    df = pd.DataFrame([data])

    # enforce same feature order as training
    df = df[scaler.feature_names_in_]

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    label = target_encoder.inverse_transform([prediction])[0]

    return label


# REST API endpoint
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    input_obj = PlacementInput(**data)
    result = make_prediction(input_obj)

    return jsonify({"prediction": result})


# Web UI
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        input_obj = PlacementInput(
            CGPA=request.form["CGPA"],
            Internships=request.form["Internships"],
            Projects=request.form["Projects"],
            Workshops=request.form["Workshops"],
            AptitudeTestScore=request.form["AptitudeTestScore"],
            SoftSkillsRating=request.form["SoftSkillsRating"],
            SSC_Marks=request.form["SSC_Marks"],
            HSC_Marks=request.form["HSC_Marks"],
        )

        prediction = make_prediction(input_obj)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
