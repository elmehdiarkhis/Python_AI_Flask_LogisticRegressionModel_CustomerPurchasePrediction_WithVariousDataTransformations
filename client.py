import streamlit 
import json
import requests
import pandas
import numpy
 
#Website Header--------------------------------------
streamlit.write("""
    #Simple Purchase Prediction App
    This app predicts whether a costumer will 
    purchase a product or not based on his age and salary
""")
#----------------------------------------------------

#Website Sidebar-------------------------------------
streamlit.sidebar.header("User Input Parameters")

#User Input--------------------
def user_imput():
    age = streamlit.sidebar.number_input('Enter age',min_value=18,max_value=80)
    salary = streamlit.sidebar.number_input('Enter salary',min_value=0,max_value=500000)

    data = {
        'age':age,
        'salary': salary
    }

    features = pandas.DataFrame(data,index=[0])
    return features


df = user_imput()
streamlit.subheader('User Input Parameters')
streamlit.write(df)
#--------------------


#extract age and salary
age = df['age'][0]
salary = df['salary'][0]
#--------------------


def user_choice_3_DataTansfortm_radioButton():
    choice = streamlit.sidebar.radio('Select a Data Transformation',('standardScaler','normalization','rescale','binary'))
    return choice



#extract data transformation
data_transformation = user_choice_3_DataTansfortm_radioButton()

if(data_transformation=='standardScaler'):
    mySc = 0
elif(data_transformation=='normalization'):
    mySc = 1
elif(data_transformation=='rescale'):
    mySc = 2
elif(data_transformation=='binary'):
    mySc = 3
#---------------------------


#----------------------------------------------------




#Envoyer la Requette post vers le serveur
request_data = json.dumps({'age':int(age),'salary':float(salary),'data_transformation':int(mySc)})
response = requests.post("http://localhost:2703/model",request_data)
#----------------------------------------------



#Recuperer la reponse--------------------------------
res = response.json()
prediction = res['Predicted Class']
prediction_proba = res['prediction_proba']
lower_prediction = res['lower_prediction']
#----------------------------------------------------




#Afficher la reponse --------------------------------
streamlit.subheader('Prediction')
if(prediction == 0):
    streamlit.write(f"{prediction} : The costumer will not purchase the product")
elif(prediction == 1):
    streamlit.write(f"{prediction} : The costumer will purchase the product")

streamlit.subheader('Probability Of my Prediction')
streamlit.write(prediction_proba)

streamlit.subheader('Probability of the other class')
streamlit.write(lower_prediction)
#-----------------------------------------------------


#run : streamlit run client.py