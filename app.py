from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import csv

l=[]
j=0
with open('ex3.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for lines in csvFile:
      arr=[]
      i=0
      
      for data in lines:
          arr.append(data)
          i+=1
      l.append(arr)
      j+=1

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__, static_url_path="/static")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/location')
def location():
    return render_template("location.html")

@app.route('/input')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)

@app.route("/predict1",methods=['POST'])
def predict1():
    c=request.form['Location']
    req=next(row for row in l if row[1] == c)
    
   
    
    print(int(req[0]))
    N = float(l[int(req[0])][6])
    P = float(l[int(req[0])][8])
    K = float(l[int(req[0])][9])
    temp = float(l[int(req[0])][3])
    humidity = float(l[int(req[0])][5])
    ph = float(l[int(req[0])][7])
    rainfall = float(l[int(req[0])][4])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('location.html',result = result)  


if __name__ == "__main__":
    app.run(debug=True)