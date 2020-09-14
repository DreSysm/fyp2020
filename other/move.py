import os
import numpy as np
import  shutil 

dir_path = os.path.dirname(os.path.realpath(__file__))
d = dir_path+"/datasets/"
s = dir_path+"/images/Aberts_Towhee/abc.jpg"
for x in os.listdir("images") :
    print(x)
    if(os.path.exists(dir_path+"/images/"+x)):
        for y in os.listdir(dir_path+"/images/"+x):
            shutil.copy(dir_path+"/images/"+x+"/"+y, d)
