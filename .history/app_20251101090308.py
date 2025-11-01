from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        past_score = float(request.form['past_score'])
        internet = int(request.form['internet'])
        extracurricular = int(request.form['extracurricular'])

        features = np.array([[study_hours, attendance, past_score, internet, extracurricular]])
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction_text=f'🎓 Predicted Final Exam Score: {round(prediction, 2)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
