import numpy as np
import cv2
from keras.models import load_model

# Load the saved model
model = load_model('mnist_model.h5')

# Create a black canvas to draw on
canvas = np.ones((280, 280), dtype=np.uint8) * 255  # White canvas

# Variables to track mouse events
drawing = False
start_point = (-1, -1)

# Mouse callback function
def draw(event, x, y, flags, param):
    global start_point, drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, start_point, (x, y), (0, 0, 0), thickness=5)  # Draw black line
            start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Function to predict the drawn digit
def predict_digit(canvas):
    # Preprocess the canvas
    img = cv2.resize(canvas, (28, 28))  # Resize to 28x28
    img = cv2.bitwise_not(img)  # Invert colors (white background to black)
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0  # Normalize and reshape
    
    # Display the processed image for debugging (optional)
    cv2.imshow("Processed Image", img[0].reshape(28, 28) * 255)  # Show the processed image
    cv2.waitKey(1)  # Wait for a short time

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    return digit

# Create a window
cv2.namedWindow("Drawing")
cv2.setMouseCallback("Drawing", draw)

while True:
    cv2.imshow("Drawing", canvas)
    
    # Wait for the user to press 'Enter' to predict or 'c' to clear the canvas
    key = cv2.waitKey(1)
    if key == 13:  # Enter key
        predicted_digit = predict_digit(canvas)
        print(f'Predicted Digit: {predicted_digit}')
    elif key == ord('c'):  # Clear canvas on 'c' key
        canvas = np.ones((280, 280), dtype=np.uint8) * 255  # Reset to white canvas
        print("Canvas cleared. Draw again!")
    elif key == ord('q'):  # Exit on 'q' key
        break

cv2.destroyAllWindows()
