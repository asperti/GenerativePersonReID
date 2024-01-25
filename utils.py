##Data parameters
import numpy as np
from PIL import Image
import os


def load_data(data_path, name):
    print(f"loading {name} data.. ")
    x_data = []
    y_data = []
    c_data = []
    
    for image_path in os.listdir( data_path+"/"):
        if image_path[-4:] != ".jpg" :
            continue
        path = data_path+"/"+image_path
        x_data.append(  np.array( Image.open(path).convert('RGB') ) )
        
        pid,cid = image_path.split("_")[-1].split(".")[0].split("c")
        
        y_data.append( int(pid) )
        c_data.append( int(cid) )

    ##normalize input
    x_data = np.array(x_data).astype(np.float32) / 255.
    y_data = np.array(y_data)
    c_data = np.array(c_data)
    
    
    return x_data,y_data,c_data


###########################################

