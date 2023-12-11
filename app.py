import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def preprocessing(data, ohe):
    # data.drop(['Person ID'], axis=1, inplace=True)
    
    data["Gender"] = data["Gender"].replace({
    "Male": 0,
    "Female": 1
    })
    
    data_ohe = pd.DataFrame(ohe.transform(data[["BMI Category"]]))
    data.drop(["BMI Category"], axis=1, inplace=True)
    data = pd.concat([data, data_ohe], axis=1)
    
    return data

def index_to_label(index):
    label = {
        0 : "Normal",
        1 : "Insomnia",
        2 : "Sleep Apnea"
    }
    return label[index]

def load_model():
    model = pickle.load(open('model1.pkl', 'rb'))
    return model

def load_ohe():
    ohe = pickle.load(open('ohe1.pkl', 'rb'))
    return ohe

@app.route('/')
def index():
    return render_template('Fibonacci.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    ohe = load_ohe()
    model = load_model()
    
    # Extracting data from the JSON request
    gender = request.form.get("gender")
    age = request.form.get("age")
    weight = float(request.form.get("weight"))
    height = float(request.form.get("height")) 
    sleep_duration = request.form.get("sleep_duration")
    quality_of_sleep = request.form.get("quality_of_sleep")
    activity = float(request.form.get("activity"))
    stress = request.form.get("stress")
    heart_rate = request.form.get("heart_rate")
    daily_steps = request.form.get("daily_steps")

    # Hitung BMI
    height /= 100 
    bmi = weight / (height * height)

    # Kategori BMI
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal Weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Ubah ke Menit
    physical_activity = activity * 60

    # Creating a DataFrame from the JSON data
    data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Sleep Duration": [sleep_duration],
        "Quality of Sleep": [quality_of_sleep],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress],
        "BMI Category": [bmi_category],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps]
    })

    processed_data = preprocessing(data, ohe)
    processed_data.columns = processed_data.columns.astype(str)
    prediction = model.predict(processed_data)
    result = index_to_label(prediction[0])
    result = str(result)
    
    return render_template('insomnia.html', result=result)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/recommendapp')
def recommendapp():
    return render_template('recommendapp.html')

@app.route('/ourteam')
def ourteam():
    return render_template('ourteam.html')

if __name__ == '__main__':
    app.run(debug=True)