import os
import numpy as np
f = open("classes.txt")
a = f.read().split()
a = np.array(a).reshape(-1, 2)
dir_path = os.path.dirname(os.path.realpath(__file__))
for x in os.listdir("images") :
    if(os.path.exists(dir_path+"/images/"+x)):
        i = 0
        for y in os.listdir(dir_path+"/images/"+x):
            i = i+1
            os.rename((dir_path+"/images/"+x+"/"+y),(dir_path+"/images/"+x+"/"+x+"_"+str(i)+".jpg"))
