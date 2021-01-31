from wsgiref import simple_server
from flask import Flask, request, render_template
import pickle
import json
import numpy as np
from keras.models import model_from_yaml
from keras.models import model_from_json
"""
*****************************************************************************
*
* filename:       main.py
* version:        1.0
* author:         Harish
* creation date:  22-JAN-2021
*
* change history:
*
* who             when           version  change (include bug# if apply)
* ----------      -----------    -------  ------------------------------
* HARISH          22-JAN-2021    1.0      initial creation
*
*
* description:    flask main file to run application
*
****************************************************************************
"""

app = Flask(__name__)

def predict_churn(creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary):
    """
    * method: predict_Rainfall
    * description: method to predict the results
    * return: prediction result
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          22-JAN-2021    1.0      initial creation
    *
    """
    yaml_file = open('models/model_clr.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_clr = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_clr.load_weights("models/model_clr1.h5")
    print("Loaded model from disk")

    with open('models/Churn.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("models/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    x = np.zeros(len(data_columns))
    x = np.reshape(x, newshape=(1, len(data_columns)), order='C')

    x[0][0] = creditscore
    x[0][1] = geography
    x[0][2] = gender
    x[0][3] = age
    x[0][4] = tenure
    x[0][5] = balance
    x[0][6] = numofproducts
    x[0][7] = hascrcard
    x[0][8] = isactivemember
    x[0][9] = estimatedsalary

    #if model.predict(x) == 0:
    #    str1 = 'Exited'
    #else:
    #    str1 = 'Retained'

    return loaded_clr.predict(x)

@app.route('/')
def index_page():
    """
    * method: index_page
    * description: method to call index html page
    * return: index.html
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          22-JAN-2021    1.0      initial creation
    *
    * Parameters
    *   None
    """
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
    * method: predict
    * description: method to predict
    * return: index.html
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          22-JAN-2021    1.0      initial creation
    *
    * Parameters
    *   None
    """
    if request.method == 'POST':
        creditscore = request.form['creditscore']
        geography = request.form["geography"]
        gender = request.form["gender"]
        age = request.form["age"]

        tenure = request.form['tenure']
        balance = request.form["balance"]
        numofproducts = request.form["numofproducts"]
        hascrcard = request.form["hascrcard"]
        isactivemember = request.form["isactivemember"]

        estimatedsalary = request.form['estimatedsalary']

        output = predict_churn(creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary)

        return render_template('index.html',show_hidden=True, prediction_text='This Project done by Harish Musti and Customer will {}'.format(output))


if __name__ == "__main__":
    #app.run(debug=True)
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
