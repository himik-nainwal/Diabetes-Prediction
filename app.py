from flask import Flask,request, url_for, redirect, render_template  ## importing necessary libraries
import pickle  ## pickle for loading model(Diabetes.pkl)
import pandas as pd
import numpy as np## to convert the input data into a dataframe for giving as a input to the model

app = Flask(__name__, template_folder='templates')  # still relative to module
  ## setting up flask name

model = pickle.load(open("Diabetes Prediction\model_pickle.pkl", "rb"))  ##loading model


@app.route('/')             ## Defining main index route
def home():
    return render_template("index.html")   ## showing index.html as homepage


@app.route('/predict',methods=['POST','GET'])  ## this route will be called when predict button is called
def predict(): 
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
         
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        age = request.form['age']
        
        data = np.array([[preg, glucose,st,insulin, bmi,age]])
        my_prediction = model.predict(data)
         
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)          ## Running the app as debug==True