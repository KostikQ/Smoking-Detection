import onnxruntime
import cv2
import numpy as np

class MMPose:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(self.model_path, 
                                                    providers=[
                                                               'CUDAExecutionProvider'])
        self.output_tensor = [node.name for node in self.session.get_outputs()]
        self.input_tensor = self.session.get_inputs()
        
        
    def predict(self, image):
        data = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(192, 256), 
                                     mean=[0.485, 0.456, 0.406],
                                     crop=False)
        output_result = self.session.run(self.output_tensor, input_feed={self.input_tensor[0].name: data})
        heatmap = output_result[0]
        image_shape = image.shape[:2]
        keypoints = []
        num_keypoints = heatmap.shape[1]
        for i in range(num_keypoints):
            keypoint_map = heatmap[0, i]
            max_loc = np.unravel_index(np.argmax(keypoint_map), keypoint_map.shape)
            x = max_loc[1] * image_shape[1] / keypoint_map.shape[1]
            y = max_loc[0] * image_shape[0] / keypoint_map.shape[0]
            keypoints.append((x, y))
        return keypoints
    
    def draw_points(self, img, keypoints, radius):
        color = (0, 255, 0)
        thickness = -1
        vis_image = np.copy(img)
        for point in keypoints:
            cv2.circle(vis_image, tuple(map(int, point)), radius, color, thickness)
        return vis_image