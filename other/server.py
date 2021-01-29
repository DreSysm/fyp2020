from flask import Flask
from flask_pymongo import PyMongo
import pymongo
from pymongo import MongoClient
import predict
import requests



app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client["bird"]
collection = db["bird_info"]


@app.route('/return_info') # receive user chosen and return bird info to user.
def return_info():
    ans = predict.result("Pied_Kingfisher.jpg")
    bird_info = collection.find_one({"name":ans[0]})

    return bird_info["description"]

@app.route('/predict_result') #receive photo and predict result ,then give user five options.
def predict_result():



    return 0






if __name__ == '__main__':
   app.run(debug=True)