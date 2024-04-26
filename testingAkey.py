import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 125
offset1 = 20
imgSize = 300
labels = ["A", "B", "C", "E", "G", "H", "L", "O", "1", "2", "3"]
single_hand_signs = ["L", "C", "1", "2", "3", "O"]  # Signs that only require one hand

# Initialize variables for sentence building
sentence = []
current_label = None
blink_start_time = None

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # Check if at least one hand is detected
    if hands:
        # Process only the first hand
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        cx, cy = hand["center"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset1:cy + h + offset1, x - offset1:cx + w + offset1]

        # Adjust aspect ratio and resize
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if 0 <= index < len(labels):
            current_label = labels[index]  # Update the current label
        else:
            current_label = None

        # Draw bounding box and current label
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.putText(imgOutput, current_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    # Handle key press 'a' to add label
    if cv2.waitKey(1) & 0xFF == ord('a'):
        if current_label and current_label not in sentence:
            sentence.append(current_label)
            blink_start_time = time.time()  # Start blink timer

    # Blink the box green if label was just added
    if blink_start_time:
        elapsed_time = time.time() - blink_start_time
        if elapsed_time <= 0.5:  # Blink duration of 0.5 seconds
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)  # Blink in green
        else:
            blink_start_time = None  # Stop blinking

    # Show the content of the sentence when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Current sentence:", sentence)

    # Clear the sentence when 'r' is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        sentence.clear()
        print("Sentence cleared.")

    # Delete the latest added item from the sentence when 'd' is pressed
    if cv2.waitKey(1) & 0xFF == ord('d'):
        if sentence:
            sentence.pop()
            print("Latest item deleted.")

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
