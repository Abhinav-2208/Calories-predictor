from flask import Flask,render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


app=Flask(__name__)

model = XGBRegressor()
model.load_model('calories_xgb_model.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submit():
    input_data = request.form  # Get the form data

    age = int(input_data['age'])
    height = float(input_data['height'])
    weight = float(input_data['weight'])
    duration = float(input_data['duration'])
    heart_rate = float(input_data['heart_rate'])
    body_temp = float(input_data['body_temperature'])

    # Convert input data to a NumPy array with the same order as the model training features
    input_array = np.array([[age, height, weight, duration, heart_rate, body_temp]])
    

    prediction = model.predict(input_array)[0]
    
    if prediction > 0:
        message = f"The predicted calories burned is {prediction:.2f} calories."
    else:
        message = "Invalid prediction."
        
    return render_template('index.html', prediction=prediction,message=message)

    

if __name__=="__main__":
    app.run(debug=False)
    
