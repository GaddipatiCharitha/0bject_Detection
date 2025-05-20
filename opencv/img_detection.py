'''from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model (trained on COCO)
model = YOLO("yolov8n.pt")  # 'n' is for nano (fastest), you can use 's', 'm', 'l', or 'x'

# Load an image
image_path = ["photos/food.jpg","photos/work.jpg"]  # Replace with your image file path
results = model(image_path)

# Show results
results[0].show()  # Show image with bounding boxes and labels
results[1].show()

# Optionally, save the result image
results[0].save(filename="result.jpg")

# Display using OpenCV (optional)
img = cv2.imread("result.jpg")
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#video detection
from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")

# Open video file (ensure it's fully downloaded; .crdownload might be incomplete)
cap = cv2.VideoCapture("videos/traffic.crdownload")

# Define desired frame size
target_width = 640
target_height = 480
count=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize the frame
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Run YOLOv8 inference
    results = model(resized_frame, stream=True)

    # Draw bounding boxes
    for r in results:
        resized_frame = r.plot()

    # Show the frame
    cv2.imshow("YOLOv8", resized_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

