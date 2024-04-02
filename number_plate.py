import cv2
from easyocr import Reader
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture('test4.mp4')

# Initialize EasyOCR reader
reader = Reader(['en'])

# Load the Haar Cascade XML file for Indian number plates
harcascade = "model/indian_license_plate.xml"


# Create a text file to save the number plates
text_file = open('number_plates.txt', 'w')

while True:
    # Read frames from the video
    ret, frame = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates using Haar Cascade
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        # Draw a rectangle around the detected number plates
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the number plate region
        number_plate = frame[y:y+h, x:x+w]

        # Use EasyOCR to read text from the number plate
        result = reader.readtext(number_plate)

        # Extract text and display it
        text = result[0][-2] if result else ''
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Save the extracted text to the text file
        if(text!=''):
            text_file.write(text + '\n')

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()
text_file.close()