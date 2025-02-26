import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")  # Ensure the model file is in the correct location

# Load video
video_path = "videos/input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video is loaded
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
out = None  # Initialize output video writer

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize the video writer once frame dimensions are known
    if out is None:
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    # Run YOLOv8 on frame
    results = model(frame)

    # Draw bounding boxes & count people
    person_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Class 0 in COCO dataset is "Person"
            if cls == 0 and conf > 0.5:
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"People Count: {person_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Show frame
    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()  # Release video writer
cv2.destroyAllWindows()
