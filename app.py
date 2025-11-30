from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/linear_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Gender & platform impact values
GENDER_IMPACT = {
    "F": 0.5,
    "M": 0.2,
    "Other": 0.35
}

PLATFORM_IMPACT = {
    "Instagram": 0.4,
    "TikTok": 0.7,
    "Snapchat": 0.3,
    "Twitter": 0.2,
    "Facebook": 0.1,
    "YouTube": 0.25
}

# Full page UI without sidebar
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Social Media Mental Health Predictor</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">

<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #c8d8ff, #f8d7ff);
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .container {
        background: white;
        width: 420px;
        padding: 35px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        animation: popUp 0.6s ease;
    }

    @keyframes popUp {
        0% { transform: scale(0.8); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    h2 {
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }

    p {
        font-size: 14px;
        color: #555;
    }

    input, select {
        width: 85%;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #ccc;
        margin-top: 15px;
        font-size: 15px;
        outline: none;
        transition: 0.2s;
    }

    input:focus, select:focus {
        border-color: #7a5af5;
        box-shadow: 0 0 5px rgba(122, 90, 245, 0.3);
    }

    button {
        margin-top: 20px;
        padding: 12px 25px;
        border: none;
        background: #7a5af5;
        color: white;
        border-radius: 12px;
        cursor: pointer;
        font-size: 16px;
        transition: 0.25s;
        width: 85%;
    }

    button:hover {
        background: #6443d1;
        transform: translateY(-2px);
    }

    #result {
        margin-top: 25px;
        font-size: 20px;
        font-weight: 600;
        color: #222;
        opacity: 0;
        transform: translateY(10px);
        transition: 0.3s ease;
    }

    #result.show {
        opacity: 1;
        transform: translateY(0);
    }
</style>
</head>

<body>
<div class="container">
    <h2>📱 Mental Health Predictor</h2>
    <p>Enter your details below:</p>

    <input type="number" id="hours" placeholder="Social media hours (e.g. 3.5)" step="0.1">

    <select id="gender">
        <option value="F">Female</option>
        <option value="M">Male</option>
        <option value="Other">Other</option>
    </select>

    <select id="platform">
        <option>Instagram</option>
        <option>TikTok</option>
        <option>Snapchat</option>
        <option>Twitter</option>
        <option>Facebook</option>
        <option>YouTube</option>
    </select>

    <button onclick="predict()">Predict</button>

    <div id="result"></div>
</div>

<script>
function predict() {
    const hours = document.getElementById('hours').value;
    const gender = document.getElementById('gender').value;
    const platform = document.getElementById('platform').value;

    if (!hours) {
        document.getElementById('result').innerHTML = "⚠ Please enter hours!";
        document.getElementById('result').classList.add("show");
        return;
    }

    fetch(`/predict?hours=${hours}&gender=${gender}&platform=${platform}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML =
                "Predicted Mental Health Score:<br><br><b>" +
                data.predicted_mental_health_score.toFixed(2) +
                "</b>";
            document.getElementById('result').classList.add("show");
        });
}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict")
def predict():
    hours = request.args.get("hours", type=float)
    gender = request.args.get("gender", type=str)
    platform = request.args.get("platform", type=str)

    if hours is None:
        return jsonify({"error": "Missing hours"}), 400

    # Base model prediction
    X = np.array([[hours]])
    X_scaled = scaler.transform(X)
    base = float(model.predict(X_scaled)[0])

    # Add impacts
    gender_effect = GENDER_IMPACT.get(gender, 0)
    platform_effect = PLATFORM_IMPACT.get(platform, 0)

    final_score = base + gender_effect + platform_effect

    return jsonify({
        "predicted_mental_health_score": final_score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
