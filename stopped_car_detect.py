from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

FILE_PATH = "C:/Users/itani/Downloads/roi/GitHub_Projects/Detect_Stopped_Cars/Videos/traffic.mp4"
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# For Mask
mask = cv2.imread('C:/Users/itani/Downloads/roi/GitHub_Projects/Detect_Stopped_Cars/masks/mask_traffic_2.png')

# For Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
cap  = cv2.VideoCapture(FILE_PATH)
model = YOLO("C:/Users/itani/Downloads/roi/GitHub_Projects/Detect_Stopped_Cars/yolo_weights/yolov8n.pt") 


last_centroids = {}  # dictionary to store the last centroid of each object
current_centroids = {}  # dictionary to store the current centroid of each object
counter = 0

while True:
    _, img = cap.read()
    img_reg = cv2.bitwise_and(img, mask)

    results = model(img_reg, stream=True)
    detections = np.empty((0,5))

    for r in results:
        counter+=1
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            # Classname
            cls = int(box.cls[0])

            # Confodence score
            conf = math.ceil(box.conf[0]*100)/100
            if conf > 0.5:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

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
                    # Object did not move or moved less than the threshold distance
                    print("Object {} is not moving in location LOC_OBJECT ".format(id))
                    cv2.putText(img, f'Vehicle_ID: {id} not moving',(10,50),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),8)
            last_centroids[id] = (cx, cy)
            current_centroids[id] = (cx, cy)


    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
