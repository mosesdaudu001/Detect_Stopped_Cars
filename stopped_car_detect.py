from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os

class ObjectDetection():

    def __init__(self, capture, result):
        self.capture = capture
        self.result = result
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolo_weights/yolov8n.pt")
        model.fuse()

        return model
    
    def predict(self, img):
        results = self.model(img, stream=True)
        return results
    
    def plot_boxes(self, results, detections, counter):

        for r in results:
            counter+=1
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w,h = x2-x1, y2-y1

                # Classname
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                # Confodence score
                conf = math.ceil(box.conf[0]*100)/100

                if conf > 0.5:
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))
                    
        return detections, counter
   
    def track_detect(self, img, detections, tracker, last_centroids, stopped_vehicles, counter):
        resultTracker = tracker.update(detections)

        for res in resultTracker:
            x1,y1,x2,y2,id = res
            x1,y1,x2,y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w,h = x2-x1, y2-y1

            cvzone.putTextRect(img, f'{id}', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))
            cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)   

            # For detecting stopped vehicles
            if counter % 10 == 0:

                if id in last_centroids:
                    last_cx, last_cy = last_centroids[id]
                    distance = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5  # Euclidean distance between last and current centroids

                    if distance < 5:
                        stopped_vehicles.append(id)
                last_centroids[id] = (cx, cy)

            if stopped_vehicles.count(id) > 1:
                cv2.putText(img, f'Vehicle_ID: {id} not moving',(10,50),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),8)
                print("Object {} is not moving in location LOC_OBJECT ".format(id))



        return img

    def __call__(self):

        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        result_path = os.path.join(self.result, 'results.avi')

        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
        vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        mask = cv2.imread('masks/mask_traffic_2.png')

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        if not os.path.exists(self.result):
            os.makedirs(self.result)
            print("Result folder created successfully")
        else:
            print("Result folder already exist")

        last_centroids = {}  # dictionary to store the last centroid of each object
        stopped_vehicles = []  # dictionary to store the current centroid of each object
        counter = 0

        while True:

            _, img = cap.read()
            assert _
            img_reg = cv2.bitwise_and(img, mask)
            
            detections = np.empty((0,5))
            results = self.predict(img_reg)
            detections, counter = self.plot_boxes(results, detections, counter)
            detect_frame = self.track_detect(img, detections, tracker, last_centroids, stopped_vehicles, counter)

            out.write(detect_frame)
            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture="Videos/traffic.mp4", result='result')
detector()
