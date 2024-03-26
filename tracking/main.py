import cv2
import requests
import math
import numpy as np
import matplotlib.pyplot as plt

def send_to_ppe_server(imencoded):
    file = {'file': ('image.jpg', imencoded.tobytes(), 'image/jpeg', {'Expires': '0'})}
    response = requests.post("http://127.0.0.1:8000/detection", files=file, timeout=50)
    return response

def send_to_pose_server(imencoded):
    file = {'file': ('image.jpg', imencoded.tobytes(), 'image/jpeg', {'Expires': '0'})}
    response = requests.post("http://127.0.0.1:8001/pose", files=file, timeout=50)
    return response

def crop2img(point, box):
    return [point[0] + box[0], point[1] + box[1]]


def start_video_processing():
    source = 'video/IMG_3991.mp4'
    video = cv2.VideoCapture(source)
    while True:
        ret, frame = video.read()
        if ret:
            _, jpeg = cv2.imencode('.jpg', frame)
            response = send_to_ppe_server(jpeg)
            response = response.json()
            boxes, scores, class_ids = response['boxes'], response['scores'], response['class_ids']
            keypoints_list = []
            for i in range(len(boxes)):
                box = list(map(int, boxes[i]))
                crop = np.copy(frame)
                crop = crop[box[1]:box[3], box[0]:box[2]]
                _, jpeg = cv2.imencode('.jpg', crop)
                response = send_to_pose_server(jpeg)
                response = response.json()
                keypoints = response['keypoints']
                keypoints = [crop2img(point, box) for point in keypoints]
                keypoints_list.append(keypoints)
            vis_image = np.copy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for keypoints in keypoints_list:
                for point in keypoints:
                    radius = (int)((frame.shape[0] + frame.shape[1]) / 400)
                    cv2.circle(vis_image, tuple(map(int, point)), radius, (0, 255, 0), -1)
            plt.imshow(vis_image)
            plt.show()
            print(f'boxes = {boxes},\nclass_ids = {class_ids},\nscores = {scores},\nkeypoints = {keypoints_list}')
            break

if __name__ == '__main__':
    start_video_processing()