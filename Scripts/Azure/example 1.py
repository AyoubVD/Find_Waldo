from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import json
import io 
from array import array
import os
from PIL import Image
import sys
import time

'''
Authenticate
Authenticates your credentials and creates a client.
'''
#subscription_key = "PASTE_YOUR_COMPUTER_VISION_SUBSCRIPTION_KEY_HERE"
#endpoint = "PASTE_YOUR_COMPUTER_VISION_ENDPOINT_HERE"

credential = json.load(open('./Scripts/Azure/credentials.json'))
API_KEY = credential['API_KEY']
PREDICTION_KEY = credential['PREDICTION_KEY']
ENDPOINT = credential['ENDPOINT']

cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(PREDICTION_KEY))
print(os.getcwd())
image_url = './Scripts/images/testing/waldo/10_15_4.jpg'
from os.path import exists
file_exists = exists(image_url)
print(file_exists)
img = Image.open(image_url)
#Tells your client to read a file from local directory
#response = cv_client.read_in_stream(image=img,raw = True, )
response = cv_client.analyze_image_in_stream(img)
#use .read() for url

operationlocation = response.headers['Operation-Location']
print('he')