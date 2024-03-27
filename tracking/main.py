import cv2
import requests
from math import sqrt
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort


max_cosine_distance = 0.2
max_iou_distance = 0.8
embedder = 'mobilenet'
max_distance = 100
num_of_frames = 100
max_age = 24
n_init = 4
nms_max_overlap = 1
bgr = False

tracker = DeepSort(max_age=max_age,
                   n_init=n_init,
                   max_cosine_distance=max_cosine_distance,
                   max_iou_distance=max_iou_distance,
                   nms_max_overlap=nms_max_overlap,
                   bgr=bgr,
                   embedder=embedder,
                   max_distance=max_distance
                   )

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

def get_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def start_video_processing():
    source = r'd:/Work/Cyclop/Smoking/tracking/video/IMG_3991.mp4'
    video = cv2.VideoCapture(source)
    global_people_track = {}
    track_people = {}
    while True:
        ret, frame = video.read()
        
        if ret:
            #frame = cv2.imread(r'video/man.png')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, jpeg = cv2.imencode('.jpg', frame)
            response = send_to_ppe_server(jpeg)
            response = response.json()
            boxes, scores, class_ids = response['boxes'], response['scores'], response['class_ids']
            keypoints_list = []
            ic.disable()
            ic(boxes)
            ic(class_ids)
            ic(keypoints_list)
            detections = list(zip(boxes, scores, class_ids))
            ic(detections)
            tracks = tracker.update_tracks(raw_detections=detections, frame=rgb)
            ic(tracks)
            ic.enable()

            for track in tracks:
                if not track.is_confirmed():
                    continue
                l, t, w, h = map(int, track.to_ltrb())
                ic(track.to_ltrb())
                ic(track.track_id)
                track_id = track.track_id
                crop = np.copy(frame[t:h-t, l:w-l])
                _, jpeg = cv2.imencode('.jpg', crop)
                response = send_to_pose_server(jpeg)
                plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                plt.axis(False)
                plt.show()
                ic(response)
                response = response.json()
                ic(response)
                keypoints = response['keypoints']
                ic(keypoints)
                k_image = np.copy(rgb)
                keypoints = [(int(x + l), int(y + t)) for x, y in keypoints]
                ic(keypoints)
                thickness = int((k_image.shape[0] + k_image.shape[1])*0.002)
                k_image = cv2.rectangle(k_image, (l, t), (w-l, h-t), (255, 0 , 0), thickness)               
                for p_ind in range(len(keypoints)):
                    point = keypoints[p_ind]
                    k_image = cv2.circle(k_image, point, thickness, (255, 255, 0), -1)
                    k_image = cv2.putText(k_image, f"{p_ind}", point, cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness//4, (255, 255, 255), thickness//4)
                nose = keypoints[0]
                lw_dist, rw_dist, lb_dist, rb_dist = map(lambda x: get_distance(x, nose), keypoints[9:13])
                ic(lw_dist)
                ic(rw_dist)
                ic(lb_dist)
                ic(rb_dist)
                ic(track_people.keys())
                if not(track_id in track_people.keys()):
                    track_people[track_id] = [rw_dist/rb_dist < 0.2 or lw_dist/lb_dist < 0.2]
                else:
                    track_people[track_id].append(rw_dist/rb_dist < 0.2 or lw_dist/lb_dist < 0.2)
                ic(track_people)
                plt.imshow(k_image)
                plt.axis(False)
                plt.show()



                


if __name__ == '__main__':
    start_video_processing()