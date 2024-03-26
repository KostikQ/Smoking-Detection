from fastapi import FastAPI, File
import uvicorn
from PIL import Image
import numpy as np
import io
from yolo import YOLOv8
import matplotlib.pyplot as plt
from typing import Any
import cv2
from pydantic import BaseModel

DETECTION_MODEL = r'models/ppe.onnx'

app = FastAPI()
    
@app.post("/")
def root():
    pass

@app.post('/detection')
def post_pose(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:,:,::-1].copy()
    #plt.imshow(image)
    #plt.show()

    yolo = YOLOv8(DETECTION_MODEL, conf_thres=0.7, iou_thres=0.5)
    yolo(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #detect = yolo.draw_detections(rgb_image)
    #plt.imshow(detect)
    #plt.show()
    
    if len(yolo.scores) != 0:
        scores_list = yolo.scores.tolist()
        boxes_list = yolo.boxes.tolist()
        class_ids_list = yolo.class_ids.tolist()
        return {"boxes": boxes_list, "class_ids": class_ids_list, "scores": scores_list}
    
    return {"boxes": yolo.boxes, "class_ids": yolo.class_ids, "scores": yolo.scores}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
