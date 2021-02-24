from flask import Flask , request , jsonify
from flask_pymongo import PyMongo
import pymongo
from pymongo import MongoClient
import predict
import requests
import os
import json



app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client["bird"]
collection = db["bird_info"]
dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/predict_result', methods=['POST']) #receive photo and predict result ,then give user five options.
def predict_result():
    json_bird = []
    uploaded_file = request.files['image']
    uploaded_file.save(dir_path+'/image/'+uploaded_file.filename)
    ans = predict.result('./image/'+uploaded_file.filename)
    os.remove(dir_path+'/image/'+uploaded_file.filename)
    # print("complete")
    # print(ans)
    # for x in ans:
    #     print(x)

    return  json.dumps(ans)


@app.route('/return_info',methods=['POST']) # receive user chosen and return bird info to user.
def return_info():
    bird_name = request.json["bird_name"]
    bird_info = collection.find_one({"name":bird_name})
    return json.dumps(bird_info)



if __name__ == '__main__':
   app.run(debug=True)