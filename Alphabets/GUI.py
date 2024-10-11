import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw

# Load the pre-trained model
model = load_model('trained_model.h5')

# Create a new Tkinter window
class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Letter")
        self.canvas = Canvas(self.master, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.button_predict = Button(self.master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = Button(self.master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new("L", (280, 280), 255)  # Create a new blank image
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        # Draw on the canvas
        x, y = event.x, event.y
        r = 15  # Radius of the brush
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black', outline='black')

    def reset(self, event):
        pass  # Do nothing on mouse release

    def clear_canvas(self):
        # Clear the canvas and image
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Reset image
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Prepare the image for prediction
        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to match model input

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        print(f'Predicted Letter: {chr(predicted_class + 65)}')  # Convert 0-25 to A-Z

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
