#read data
from pandas import read_csv

data = read_csv("storepurchasedata.csv")
#-----------------

#Describe data----------------
description = data.describe()
#-----------------

#split the data
array = data.values 
X = array[:,0:-1] 
Y = array[:,-1]
#-----------------

#split the data into train and test---------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)
#-------------------------------------------------------------

#data transformation Rescale
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#------------------------------------------


#train the model---------------------------
model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)
#------------------------------------------


#predict-----------------------------------
Y_pred = model.predict(X_test)
Y_proba = model.predict_proba(X_test)
#------------------------------------------

#Matric de confusion-----------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

matrix = confusion_matrix(Y_test, Y_pred)
Tn,Fp,Fn,Tp = matrix.ravel();


print("Accuracy Global",accuracy_score(Y_test, Y_pred)*100,"%")
#------------------------------------------


#Same model and Transform with pickle----------------
import pickle

#save the model
pickle.dump(model, open("myModel_rescale.pkl", "wb"))
pickle.dump(sc, open("myTransf__rescale.pkl", "wb"))
#----------------------------------------------------







