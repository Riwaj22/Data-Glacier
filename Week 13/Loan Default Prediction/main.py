from flask_cors import cross_origin, CORS
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import requests
import numpy as np

app = Flask(__name__)
cors = CORS(app)
data = pd.read_csv('bank_data.csv')

# Load the pickled machine learning model
with open('wwek_12.pkl', 'rb') as file:
    model = pickle.load(file)


data['job'] = data['job'].astype(str)
data['marital'] = data['marital'].astype(str)
data['education'] = data['education'].astype(str)
data['default'] = data['default'].astype(str)
data['housing'] = data['housing'].astype(str)
data['loan'] = data['loan'].astype(str)
data['contact'] = data['contact'].astype(str)
data['month'] = data['month'].astype(str)
data['day_of_week'] = data['day_of_week'].astype(str)
data['age'] = data['age'].astype(int)
data['duration'] = data['duration'].astype(float)
data['campaign'] = data['campaign'].astype(float)
data['emp.var.rate'] = data['emp.var.rate'].astype(float)
data['cons.price.idx'] = data['cons.price.idx'].astype(float)
data['cons.conf.idx'] = data['cons.conf.idx'].astype(float)

@app.route('/')
def index():
    # Processor = sorted(data['Processor'].unique())
    Job = sorted(data['job'].unique())
    marital = sorted(data['marital'].unique())
    education = sorted(data['education'].unique())
    default = sorted(data['default'].unique())
    housing = sorted(data['housing'].unique())
    loan = sorted(data['loan'].unique())
    contact = sorted(data['contact'].unique())
    month = sorted(data['month'].unique())
    day = sorted(data['day_of_week'].unique())
    age = sorted(data['age'])
    duration = sorted(data['duration'])
    campaign = sorted(data['campaign'])
    emp = sorted(data['emp.var.rate'])
    price = sorted(data['cons.price.idx'])
    conf = sorted(data['cons.conf.idx'])


    # ... (other variables for rendering the template)
    return render_template('index.html', Job = Job,
                           marital =marital,
                           education = education,
                           default = default,
                           housing = housing,
                           loan = loan,
                           contact = contact,
                           month = month,
                           day = day,
                           age = age,
                           duration = duration,
                           campaign = campaign,
                           emp =emp,
                           price = price,
                           conf = conf
                           )

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Retrieve form data
    Job = request.form.get('Job')
    marital = request.form.get('marital')
    education = request.form.get('education')
    default = request.form.get('default')
    housing = request.form.get('housing')
    loan = request.form.get('loan')
    contact = request.form.get('contact')
    month = request.form.get('month')
    day = request.form.get('day')
    age = request.form.get('age')
    duration = request.form.get('duration')
    campaign = request.form.get('campaign')

    # Handle 'emp.var.rate' field, provide a default value if it doesn't exist
    emp = request.form.get('emp.var.rate')
    if emp is None:
        emp = 0.0  # You can set a suitable default value

    price = request.form.get('cons.price.idx')
    conf = request.form.get('cons.conf.idx')

    # Check if any of the numeric fields are missing or empty
    if age is None or duration is None or campaign is None or price is None or conf is None:
        return jsonify({'error': 'Please fill out all required fields before predicting.'})

    # Convert to appropriate data types
    age = int(age)
    duration = float(duration)
    campaign = float(campaign)
    emp = float(emp)
    price = float(price)
    conf = float(conf)

    # Make a prediction using the loaded model
    prediction = model.predict(pd.DataFrame(
        columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'age', 'duration', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx'],
        data=np.array([Job, marital, education, default, housing, loan, contact, month, day, age, duration, campaign, emp, price, conf]).reshape(1, 15)
    ))

    print(jsonify({'prediction': prediction[0]}))

    return jsonify({'prediction': prediction[0]})  # Convert to list for JSON serialization


if __name__ == "__main__":
    # Start your Flask app
    app.run(debug=True)
