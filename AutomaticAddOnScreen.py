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
labels = ["A", "B", "C", "D", "E", "H", "L", "O", "1", "2", "3", "HELLO", "GOOD", "MORNING", " "]
single_hand_signs = ["L", "C", "1", "2", "3", "O", "HELLO", "GOOD", "MORNING", " "]  # Signs that only require one hand

# Initialize variables for sentence building
sentence = []
start_time = None
current_label = None
current_label_color = None
blink_start_time = None

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # Check if at least one hand is detected
    if hands:
        # If both hands are detected, analyze them together
        if len(hands) == 2:
            # Considered as one sign
            hand = hands[0]
            x1, y1, w1, h1 = hand["bbox"]
            x2, y2, w2, h2 = hands[1]["bbox"]
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            cx = (x1 + x2 + w1 + w2) // 2
            cy = (y1 + y2 + h1 + h2) // 2

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset1:cy + h + offset1, x - offset1:cx + w + offset1]
        else:
            # Process only the first hand
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            cx, cy = hand["center"]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset1:cy + h + offset1, x - offset1:cx + w + offset1]

        # Adjust aspect ratio
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

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if 0 <= index < len(labels):
            current_label = labels[index]  # Update the current label
        else:
            current_label = None

        # Draw bounding box and label
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, current_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        print("Current Label:", current_label)
        print("Sentence:", sentence)
        print("Start Time:", start_time)

        # Check if the label should be added to the sentence
        if current_label:
            if current_label not in sentence:
                if start_time is None:
                    start_time = time.time()
                    current_label_color = (255, 0, 255)
                elif time.time() - start_time >= 3:
                    sentence.append(current_label)
                    start_time = None  # Reset start time after adding the label
                    current_label_color = (0, 255, 0)  # Change color to green
                    # Set the start time for the blink
                    blink_start_time = time.time()

    # Show the content of the sentence when 's' is pressed
    if sentence:
        sentence_text = " ".join(sentence)
        cv2.putText(imgOutput, f"Sentence: {sentence_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Delete the latest added item from the sentence when 'd' is pressed
    if cv2.waitKey(1) & 0xFF == ord('d'):
        if sentence:
            sentence.pop()
            print("Latest item deleted.")

    # Clear the sentence when 'r' is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        sentence.clear()
        print("Sentence cleared.")

    # Update the bounding box color and blinking
    if current_label:
        if blink_start_time:
            # Calculate elapsed time for blinking
            elapsed_time = time.time() - blink_start_time
            # Toggle between original color and black for 1 second
            if elapsed_time <= 1:  # Blink for 1 second
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), current_label_color, 4)
            else:
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)  # Original color
                blink_start_time = None

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
