from flask import Flask , request , jsonify
from flask_pymongo import PyMongo
import pymongo
from pymongo import MongoClient
import predict
import requests
import os
import json
from bson.json_util import dumps



app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client["bird"]
collection = db["bird_info"]
dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/predict_result', methods=['POST']) #receive photo and predict result ,then give user five options.
def predict_result():
    uploaded_file = request.files['image']
    uploaded_file.save(dir_path+'/image/'+uploaded_file.filename)
    ans = predict.result('./image/'+uploaded_file.filename)
    os.remove(dir_path+'/image/'+uploaded_file.filename)
    
    print("OK")
    return  json.dumps(ans)


@app.route('/return_info',methods=['POST']) # receive user chosen and return bird info to user.
def return_info():
    bird_name = request.json["bird_name"]
    bird_info = collection.find_one({"name":bird_name})
    return json.dumps(bird_info)

@app.route('/bird_list') # return all the birds information to the application
def bird_list():
    all_bird = collection.find()
    all_bird = list(all_bird)

    return json.dumps(all_bird)

@app.route('/search',methods=['POST'])  #get the keyword from the application and return search result to the application
def search():
    bird_name = request.json["search_bird"]
    print(bird_name)
    myquery = { "name": { "$regex": bird_name ,"$options": "-i"} }
    bird_info = collection.find(myquery)
    bird_info = list(bird_info)

    return json.dumps(bird_info)

if __name__ == '__main__':
   app.run(debug=True)