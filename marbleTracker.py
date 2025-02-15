import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

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
        type_word = ""
        match (self.type):
            case 0.0:
                type_word = "Brass"
            case 1.0:
                type_word = "Nylon"
            case 2.0:
                type_word = "Steel"

            
        return f"{self.id}: {type_word}, Center: {self.center}"

boxList = []
LINE_CUTOFF = 150

while True:
    ret, image = cam.read()
    if not ret:
        continue

    #draw limit line
    cv2.line(image, (0, LINE_CUTOFF), (640, LINE_CUTOFF), (0, 255, 0), 3)
    
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

            #only look at most recent three marbles to compare
            #anything older than that has probably left the machine
            candidates = boxList[-3:]

            for foundBox in candidates:
                distance = np.sqrt(pow(thisBoxCenter[0] - foundBox.center[0], 2) + pow(thisBoxCenter[1] - foundBox.center[1], 2))

                #decides if the marble is in the found list
                if(distance < 150 and type == foundBox.type):
                    identified = True
                    foundBox.center = [float(thisBoxCenter[0]), float(thisBoxCenter[1])]
    
            #if the marble does not match any of the previously seen
            if (not identified and conf > 0.85):
                boxList.append(Box(float(x1), float(y1), float(x2), float(y2), type, conf, id=len(boxList)+1))

            if (conf > 0.5):
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{model.names[int(type)]} {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image", image)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

for box in boxList:
    if(box.center[1] < LINE_CUTOFF):
        print(box)

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()