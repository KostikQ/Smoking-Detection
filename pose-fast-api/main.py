from fastapi import FastAPI, File
import uvicorn
from PIL import Image
import numpy as np
import io
from pose import MMPose
import matplotlib.pyplot as plt
from typing import Any
import cv2
import json
from pydantic import BaseModel

POSE_MODEL = r'models/pose_model.onnx'

app = FastAPI()
    
@app.post("/")
def root():
    pass

@app.post('/pose')
def post_pose(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = np.array(image)
    #plt.imshow(image)
    #plt.show()
    pose = MMPose(POSE_MODEL)    
    keypoints = pose.predict(image)
    #detect = pose.draw_points(image, keypoints, (int)((image.shape[0] + image.shape[1]) / 100))
    #plt.imshow(detect)
    #plt.show()
    return {'keypoints': keypoints}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
