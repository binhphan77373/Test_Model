import cv2
# Detect objects using YOLO
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

# Draw bounding boxes and labels
def draw_labels_and_boxes(img, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 2)
    return img

# Process detections
def process_detections(results, height, width):
    boxes = []
    confidences = []
    class_ids = []
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            confidence = detection.conf[0]
            class_id = detection.cls[0]
            if confidence > 0.5:
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    return boxes, confidences, class_ids