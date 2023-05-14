#-------------------------------------------------
import pickle

#GESTION DE REQUETTE---------------------------------
from flask import Flask, request

app = Flask(__name__)

import numpy as np


#si on Recoit un Post vers l'URL/model
@app.route('/model', methods=['POST'])
def myModel():
    #get data from request
    data = request.get_json(force=True)
    age = data['age']
    salary = data['salary']
    data_transformation = data['data_transformation']

    #load the model
    if(data_transformation==0):
        model = pickle.load(open("myModel_standardScaler.pkl", "rb"))
        transformation = pickle.load(open("myTransf_standardScaler.pkl", "rb"))
    elif(data_transformation==1):
        model = pickle.load(open("myModel_normalization.pkl", "rb"))
        transformation = pickle.load(open("myTransf_normalization.pkl", "rb"))
    elif(data_transformation==2):
        model = pickle.load(open("myModel_rescale.pkl", "rb"))
        transformation = pickle.load(open("myTransf_rescale.pkl", "rb"))
    elif(data_transformation==3):
        model = pickle.load(open("myModel_binary.pkl", "rb"))
        transformation = pickle.load(open("myTransf_binary.pkl", "rb"))


    #Transform
    predictionArray = np.array([[age,salary]])
    predictionArray_Transformed = transformation.transform(predictionArray)

    #predict
    prediction = model.predict(predictionArray_Transformed)
    prediction_proba = model.predict_proba(predictionArray_Transformed)

    if(prediction[0] == 0):
        biggerPrediction = prediction_proba[0][0]*100
        lowerPrediction = prediction_proba[0][1]*100
    elif(prediction[0] == 1):
        biggerPrediction = prediction_proba[0][1]*100
        lowerPrediction = prediction_proba[0][0]*100

    
    return {
        'Predicted Class': float(prediction[0]),
        'prediction_proba': float(biggerPrediction),
        'lower_prediction': float(lowerPrediction)
    }


if __name__ == '__main__':
    app.run(port=2703, debug=True)