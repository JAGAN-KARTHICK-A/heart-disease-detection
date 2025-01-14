from flask import Flask, request, jsonify, render_template, flash, redirect
import torch
from model import load_model, HeartModel, checkCondition
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = 'ENTER_YOUR_SECRET_KEY_HERE'

with open("scaler/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = load_model("models/model.pth")

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        input_features = {
            'age' : [int(data["age"])*365],
            'height' : [int(data["height"])],
            'weight' : [int(data["weight"])],
            'ap_hi' : [int(data["ap-hi"])],
            'ap_lo' : [int(data["ap-lo"])],
            'smoke' : [0],
            'active' : [0],
            'alco' : [0],
            'cholesterol_1' : [0],
            'cholesterol_2' : [0],
            'cholesterol_3' : [0],
            'gluc_1' : [0],
            'gluc_2' : [0],
            'gluc_3' : [0],
            'gender_1' : [0],
            'gender_2' : [0],
        }
        
        input_features = checkCondition(input_features, 'smoke', data["smoke"])
        input_features = checkCondition(input_features, 'active', data["active"])
        input_features = checkCondition(input_features, 'alco', data["alcohol"])

        if data["gender"] == "Male": input_features["gender_2"] = [1]
        else: input_features["gender_1"] = [1]

        if data["cholesterol"] == "Normal": input_features["cholesterol_1"] = [1]
        elif data["cholesterol"] == "Above normal": input_features["cholesterol_2"] = [1]
        else: input_features["cholesterol_3"] = [1]

        if data["glucose"] == "Normal": input_features["gluc_1"] = [1]
        elif data["glucose"] == "Above normal": input_features["gluc_2"] = [1]
        else: input_features["gluc_3"] = [1]

        input_features = pd.DataFrame(input_features)

        x_scaled = scaler.transform(input_features)

        input_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predictions = torch.sigmoid(output)
            predictions = (predictions > 0.5).float().numpy()

        if predictions[0] == 0:
            flash(f"The patient have (No Risk or Low Risk) of having a heart disease", category='success')
        else:
            flash(f"The patient have High Risk of having a heart disease", category='danger')
        return redirect("/")
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(port = 5000, debug = True)
