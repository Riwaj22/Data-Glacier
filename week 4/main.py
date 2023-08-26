from flask_cors import cross_origin, CORS
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

from pickle import load
laptop = pd.read_csv('laptop.csv')

# Load the pickled model
with open('laptop.pkl', 'rb') as file:
    model = pickle.load(file)

app=Flask(__name__)
cors=CORS(app)


@app.route('/')

def index():
    company = sorted(laptop['Company'].unique())
    typename = sorted(laptop['TypeName'].unique())
    CPU = sorted(laptop['Cpu'].unique())
    GPU =sorted(laptop['Gpu'].unique())
    OpSys = sorted(laptop['OpSys'].unique())
    Weight = sorted(laptop['Weight'].unique())
    Ram = sorted(laptop['Ram'].unique())
    flash = sorted(laptop['Size_Flash Storage'].unique())
    hdd = sorted(laptop['Size_HDD'].unique())
    hybrid = sorted(laptop['Size_Hybrid'].unique())
    ssd = sorted(laptop['Size_SSD'].unique())
    ppi = sorted(laptop['ppi'].unique())
    return render_template('index.html', companies=company, Typename = typename, CPU = CPU, GPU=GPU, OpSys=OpSys,
                           Weight = Weight,
                           flash= flash,
                           hdd = hdd,
                           ppi =ppi,
                           hybrid = hybrid,
                           ssd=ssd,
                           Ram =Ram

                           )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        Typename = request.form.get('Typename')
        Inches = float(request.form.get('Inches'))
        CPU = request.form.get('CPU')
        Ram = float(request.form.get('Ram'))
        GPU = request.form.get('GPU')
        OpSys = request.form.get('OpSys')
        Weight = float(request.form.get('Weight'))
        Flash = float(request.form.get('Flash'))
        HDD = float(request.form.get('HDD'))
        hybrid = float(request.form.get('hybrid'))
        ssd = float(request.form.get('ssd'))
        ppi = float(request.form.get('ppi'))

        # Handle empty string values
        if '' in [company, Typename, CPU, GPU, OpSys]:
            return "Please fill out all fields"

        # Rest of your prediction code

        prediction = model.predict(pd.DataFrame(
            columns=['Company', 'TypeName', 'Inches', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight', 'Size_Flash Storage',
                     'Size_HDD', 'Size_Hybrid', 'Size_SSD', 'ppi'],
            data=np.array(
                [company, Typename, Inches, CPU, Ram, GPU, OpSys, Weight, Flash, HDD, hybrid, ssd, ppi]).reshape(
                1, 13)))

        return str(np.round(prediction[0], 2))

    except ValueError as e:
        return 'Please Complete The Form'


if __name__== "__main__":
    app.run(debug = True)


