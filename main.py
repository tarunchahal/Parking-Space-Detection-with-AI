import cv2
import pickle
import numpy as np

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)  # Green for free space
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied space
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

    # Change the text to "Parking slots available"
    text = f'Parking slots available: {spaceCounter}/{len(posList)}'
    
    # Get the size of the text to create a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
    
    # Draw a filled rectangle behind the text
    cv2.rectangle(img, (100 - 10, 50 - text_height - 10), (100 + text_width + 10, 50 + baseline), (0, 200, 0), cv2.FILLED)

    # Use cv2.putText to display the text with a different font
    cv2.putText(img, text, (100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    
    if not success:
        break  # Break the loop if there are no frames to read

    # Apply background subtraction
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Detect moving objects (people)
    fgMask = backSub.apply(img)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box for detected people

    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)

    # Check for the 'q' key to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()