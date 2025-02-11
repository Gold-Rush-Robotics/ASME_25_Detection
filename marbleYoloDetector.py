import cv2
from ultralytics import YOLO

model = YOLO("grrMarbleDetector/best.pt")

# Open the image or video
#image = cv2.imread("geodinium.jpg")

cam = cv2.VideoCapture(2)


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
            class_id = box.cls[0]
            conf = box.conf[0]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{model.names[int(class_id)]} {conf:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image", cv2.resize(image, (1380, 720)))
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()