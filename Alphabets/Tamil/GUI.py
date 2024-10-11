import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('tamil_character_recognition_model.h5')

# Define Tamil alphabets for display
tamil_alphabets = ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஒ', 'ஓ', 'ஐ', 
                   'க்', 'ச்', 'ஞ்', 'ட்', 'த்', 'ண்', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்', 'ற்', 'ன்']

# Function to preprocess the user-drawn image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (100, 100))  # Resize to match the input size of the model
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize the drawing window
window_name = 'Draw a Tamil Character'
drawing = False
canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255  # Create a white canvas

# Mouse callback function for drawing
def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 5, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Create a window and set the mouse callback function
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

while True:
    cv2.imshow(window_name, canvas)
    
    # Wait for the 'p' key to predict or 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        # Preprocess the drawn image
        processed_img = preprocess_image(canvas)
        
        # Predict the character
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        
        # Display the predicted character
        print(f"Predicted Tamil Character: {tamil_alphabets[predicted_class]}")
        canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255  # Reset the canvas
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
