from flask import Flask
from flask_pymongo import PyMongo
import pymongo
from pymongo import MongoClient



cluster = MongoClient("mongodb+srv://sydres:sydres312154cluster0.8ddot.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = cluster["bird"]
collection = db["bird_info"]

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'






# if __name__ == '__main__':
#    app.run()