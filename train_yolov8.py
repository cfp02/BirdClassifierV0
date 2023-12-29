import os
import torch
import ultralytics
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
from PIL import Image



from ultralytics.engine.results import Results
# ultralytics.checks()

'''
This program is used to detect birds in an image so a higher-resolution image can be taken of the bird.
Basically, a camera is pointed in the general direction of a bird feeder. When a bird is detected, the camera will center itself on the bird and take a photo
'''


def get_useful_results(results: Results) -> list[dict[str,float, list[int], tuple[int]]]:
    '''
    This function takes in the results of the model and returns useful info from the results
    Returns:
        class_id: The class id of the detected object
        conf: The confidence of the detected object
        coords: The coordinates of the bounding box
        center: The center of the bounding box
    '''
    res = results
    # Check if results is a list, if it is, get the first (only) element
    if type(results) == list:
        res = results[0]

    useful_boxes = []

    # Loop through the boxes
    for i in range(len(res.boxes)):
        # Get the bounding box coordinates
        class_id, conf, coords, center = info_from_box(res.boxes[i], res)
        useful_boxes.append({'class_id': class_id, 'conf': conf, 'coords': coords, 'center': center})

    return useful_boxes


def info_from_box(box, res: Results) -> list:
    coords = box.xyxy[0].tolist()
    coords = [int(i) for i in coords]
    class_index = int(box.cls.tolist()[0])
    class_id = res.names[class_index]
    conf = round(box.conf.tolist()[0], 3)
    # print(coords)
    # Get the center of the bounding box
    center = round((coords[0] + coords[2]) / 2), round((coords[1] + coords[3]) / 2)
    return class_id, conf, coords, center

def draw_box(img:Image, coords, class_id, conf) -> Image:
    '''
    This function draws a box around the detected object
    '''
    # Draw the box
    img = cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
    # Put the class name and confidence on the image
    img = cv2.putText(img, "{}: {}".format(class_id, conf), (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

def show_box(bare_img: Image, coords, class_id, conf):
    '''
    This function shows the image with the box drawn around the object, and makes the window resizable but fixed ratio
    '''
    img: Image = draw_box(bare_img, coords, class_id, conf)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def main():
    model = YOLO('models/yolov8n.pt')

    hummer = 'data/hummer.jpg'

    # Inference
    results = model.predict(hummer)[0]
    boxes = get_useful_results(results)
    if len(boxes) == 1:
        class_id, conf, coords, center = boxes[0].values()
    print(class_id, conf, coords, center)

    # Show the image with the box drawn around the object
    img = cv2.imread(hummer)
    show_box(img, coords, class_id, conf)




if __name__ == "__main__":
    main() 