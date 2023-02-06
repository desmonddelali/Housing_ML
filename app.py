import pickle
from  flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

app=Flask(__name__)
regmodel= pickle.load(open('Reg_model.pkl','rb'))
scaler =pickle.load(open('Scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['Post'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values().reshape[1,-1])))
    new_data=scaler.transform(np.array(data.values().reshape[1,-1]))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)