import cv2
import numpy as np
import os

# Load YOLO
# net = cv2.dnn.readNet("yolov8n.pt")  # Use your trained weights and model config
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("coco.names").read().strip().split("\n")

def detect_objects(image_path, conf_threshold=0.5, nms_threshold=0.4):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    filtered_class_ids = [class_ids[i] for i in indices.flatten()]
    filtered_boxes = [boxes[i] for i in indices.flatten()]
    
    return filtered_class_ids, filtered_boxes

def classify_images(image_folder, conf_threshold=0.5, nms_threshold=0.4):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            class_ids, _ = detect_objects(image_path, conf_threshold, nms_threshold)
            
            # Debug: Print detected classes for each image
            detected_classes = [classes[class_id] for class_id in class_ids]
            print(f"Image: {filename} | Detected classes: {detected_classes}")

            has_face = 'person' in detected_classes
            has_other_object = any(cls != 'person' for cls in detected_classes)

            if has_face and has_other_object:
                new_folder = 'cheating'
            elif has_face:
                new_folder = 'non-cheating'
            elif not has_face and len(detected_classes) > 0:
                new_folder = 'differentX'
            else:
                continue

            new_path = os.path.join(image_folder, new_folder)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            os.rename(image_path, os.path.join(new_path, filename))

# Example usage
classify_images(r'C:\Users\Farrukh\Weapons-Detection-Tracking\dataset_\imagesX', conf_threshold=0.5, nms_threshold=0.4)
