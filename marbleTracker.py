import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("grrMarbleDetector/best.pt")

# Open the image or video
#image = cv2.imread("geodinium.jpg")

cam = cv2.VideoCapture(2)


class Box:
    def __init__(self, x1, y1, x2, y2, type, conf, id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2 
        self.type = type
        self.conf = conf
        self.id = id

        self.center = [(x1 + x2) / 2, (y1 + y2) / 2]

    def __str__(self):
        return f"Type: {self.type}, ID: {self.id}, Center: {self.center}"

boxList = []

while True:
    ret, image = cam.read()
    if not ret:
        continue
    
    # Perform inference
    results = model.predict(image)

    # Visualize the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            type = box.cls[0] 
            conf = box.conf[0] #confidence
            thisBoxCenter = [(x1 + x2) / 2, (y1 + y2) / 2]

            identified = False
            for foundBox in boxList:
                distance = np.sqrt( pow(thisBoxCenter[0] - foundBox.center[0], 2) + pow(thisBoxCenter[1] - foundBox.center[1], 2))

                #decides if the marble is in the found list
                if(distance < 100):
                    identified = True
    
            if (not identified):
                boxList.append(Box(float(x1), float(y1), float(x2), float(y2), type, conf, id=len(boxList)+1))

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{model.names[int(type)]} {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image", cv2.resize(image, (1380, 720)))
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

for box in boxList:
    print(box)

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()