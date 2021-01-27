import pymongo
from pymongo import MongoClient



cluster = MongoClient("mongodb+srv://sydres:sydres312154cluster0.8ddot.mongodb.net/<dbname>?retryWrites=true&w=majority")
db = cluster["bird"]
collection = db["bird_info"]