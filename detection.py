import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#OPEN CV
import cv2
import imutils as imutils
import pandas as pd
import requests
import requests
import pandas as pd 
from flask import Flask, redirect, url_for, request
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

def converImgToPixel(image_name):
      # using api get rect for grabcut
  api_key = 'acc_88676af2cb3d05d'
  api_secret = 'f7c1da18660f8adcf25dddc239ac3518'
  image_path = image_name

  response = requests.post(
      'https://api.imagga.com/v2/croppings',
      auth=(api_key, api_secret),
      params={'no_scaling':1},
      files={'image': open(image_path, 'rb')})
  response.json()
  print(response)
  res = response.json().get('result').get('croppings')[0]
  x1 = res.get('x1')
  x2 = res.get('x2')
  y1 =  res.get('y1')
  y2 = res.get('y2')
  h1 = res.get('target_height')
  w1  = res.get('target_width')
  # using grabcut
  img = cv2.imread(image_name) 
  height, width, channels = img.shape
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  mask = np.zeros(img.shape[:2], np.uint8)
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)
    # rect toa do khung
  rect = (x1,y1,w1,h1)
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
  mask2 = np.where((mask==2)|(mask==0), 0,1).astype('uint8')
  img = img*mask2[:,:, np.newaxis]
  img = img[y1:y2,0:width]
  plt.imshow(img)
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img_gray = cv2.resize(img_gray,(28,28))
  pixel_values = img_gray.flatten()
  return pixel_values

@app.route('/api/image-analyzer', methods=['POST'])
def  analyzerImage():
    class_names = ['T-shirt','Trouser',' Pullover','Dress', 'Coat'
    , 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot','Lips','Eyewear'
    ,'Nails','Headewear', 'FlipLops', 'Ties']
    file = request.files['file']
    file_name = file.filename
    file.save(file_name)
    products = converImgToPixel(file_name)
    model1 =load_model('fashion.h5')
    imgs = np.array(products)
    imgs = imgs.reshape(1,28,28,1)
    real_predictions = model1.predict(imgs)
    real_predictions
    max_value = np.amax(real_predictions[0])
    default_value = 0.8
    num_product = 1;
    if (max_value < default_value):
        num_product = 2;
    prediction_value = real_predictions[0]
    indices = (-prediction_value).argsort()[:num_product]
    indices
    output=[class_names[indices[0]]]
    if (num_product ==2):
        output.append(class_names[indices[1]])
    if(output!=None and os.path.exists(file_name)):
        os.remove(file_name)
    return {"data":output}
if __name__ == '__main__':
    app.run()