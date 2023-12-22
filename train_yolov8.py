import os
import torch
import ultralytics
from ultralytics import YOLO
from matplotlib import pyplot as plt

# ultralytics.checks()

'''
This program is used to detect birds in an image so a higher-resolution image can be taken of the bird.
Basically, a camera is pointed in the general direction of a bird feeder. When a bird is detected, the camera will center itself on the bird and take a 
'''




def main():
    model = YOLO('models/yolov8n.pt')

    hummer = 'data/hummer.jpg'

    # Inference
    results = model.predict(hummer)
    # print(results)



if __name__ == "__main__":
    main() 