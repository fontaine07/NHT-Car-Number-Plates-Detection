import cv2
import easyocr

#Load the haarcascade file
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize the easyOCR reader with the desired language
reader = easyocr.Reader(['en'])

# Set up the video capture using the webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640) # width
cap.set(4, 480) # height

min_area = 500
count = 0


while True:
    success, img = cap.read()

    # Convert each frame to grayscale & use the haarcascade classifier for detection
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        # Check if the area is greater than a minimum area threshold
        if area > min_area:
            # Draw a rectangle around the license plate & add a text label
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Extract the region of interest
            img_roi = img[y: y+h, x: x+w]
            cv2.imshow("ROI", img_roi)

            # Use easyOCR to recognize text on the number plate
            result = reader.readtext(img_roi)
            
            # Print the recognized text to the terminal
            if len(result) > 0:
                text = result[0][1]
                print("Detected Text:", text)
    
    cv2.imshow("Result", img)

    # Save scan image by pressing S key to the 'plates' folder 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1
        

