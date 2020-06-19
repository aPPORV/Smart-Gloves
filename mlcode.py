import numpy as np
import pandas as pd
import serial

from string import ascii_uppercase

finaldata = pd.DataFrame(columns=(['little','ring','middle','index','thumb']))

j=0

#For datasets having 3,000 values - Characters
for i in ascii_uppercase:
  if i=='F':
    break
  # datapath = "/content/drive/My Drive/flex sensor readings/Alphabet "+i+".csv"
  datapath = "Alphabet "+i+".csv"
  data = pd.read_csv(datapath)
  for k in data.columns:     #data.columns[w:] if you have w column of line description 
    data[k] = data[k].fillna(data[k].median())
  data = data.filter(['little','ring','middle','index','thumb'])
  data.insert(5,'Character',j)
  j=j+1
  finaldata = pd.concat([finaldata, data], sort=False, ignore_index=True)

j=26

#For dataset of digits
for i in range(1,6,1):
  # datapath = "/content/drive/My Drive/flex sensor readings/Digit "+str(i)+".csv"
  datapath = "Digit "+str(i)+".csv"
  data = pd.read_csv(datapath)
  for k in data.columns:     #data.columns[w:] if you have w column of line description 
    data[k] = data[k].fillna(data[k].median())
  data = data.filter(['little','ring','middle','index','thumb'])
  data.insert(5,'Character',j)
  j=j+1
  finaldata = pd.concat([finaldata, data], sort=False, ignore_index=True)

finaldata = finaldata[ (finaldata['little'].between(280, 550)) & (finaldata['ring'].between(330, 550)) &
                      (finaldata['middle'].between(350, 550)) & (finaldata['index'].between(400, 600)) &
                      (finaldata['thumb'].between(400, 510)) ]

print("Null values as per column:\nColumn\tNull values present?")
print(finaldata.isnull().any())

feature_cols = ['little','ring','middle','index','thumb']
X = finaldata[feature_cols].to_numpy() 
Y = finaldata.Character.to_numpy()
  
from sklearn.model_selection import train_test_split
# Splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# import pickle
# model = pickle.load(open('modelmodel.sav', 'rb'))
# # result = model.score(X_test, y_test)
# from sklearn import metrics 

# y_pred=clf.predict(X_test)


# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, oy_pred))

#Voice output module
import pyttsx3

engine = pyttsx3.init("sapi5", True)

#Testing
#engine.say("Welcome to the Matrix.")
engine.runAndWait()
#engine.stop()
#engine.say()
import serial

try:
  arduino = serial.Serial('COM4',9600)
except:
  print('please check the port')

lst=[]
while True:
    e=(int)(arduino.readline())
    if(e==0):
      continue
    elif (e!=1):
      lst.append(e)
    if(e==1):
      a =np.array(lst)
      b= a.reshape(1, -1)
      val= clf.predict(b)
      if val>=0 and val<=25:
        val2 = chr(val+65)
      elif val>=26 and val<=35:
        val2 = chr(val+23)
      print("a= ")
      print(a)
      print("Predicted: ",val)
      engine.say(val2)
      engine.runAndWait()
      lst.clear()

