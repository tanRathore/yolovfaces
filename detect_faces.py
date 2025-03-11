import os
import cv2
from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")  
image_folder = "images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read file.")
        continue
    results = model(image)

    # Draw bounding boxes around detected faces
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)
    print(f"Processed {image_name}: Faces saved to {output_path}")

print("Face detection complete for all images.")
