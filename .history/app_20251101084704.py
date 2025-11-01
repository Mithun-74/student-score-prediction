from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("student_score_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Student Exam Score Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON input
    print("Received data:", data)

    # Extract features from request
    features = [
        data['Study_Hours_per_Week'],
        data['Attendance_Rate'],
        data['Past_Exam_Scores'],
        data['Internet_Access_at_Home_Yes'],
        data['Extracurricular_Activities_Yes']
    ]

    # Convert to numpy array
    final_features = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(final_features)
    result = round(prediction[0], 2)

    return jsonify({'Predicted_Final_Exam_Score': result})

if __name__ == '__main__':
    app.run(debug=True)
