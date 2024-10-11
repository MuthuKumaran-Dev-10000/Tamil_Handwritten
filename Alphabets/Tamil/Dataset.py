import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random

# Define the Tamil alphabets (example list)
tamil_alphabets = ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஒ', 'ஓ', 'ஐ', 
                   'க்', 'ச்', 'ஞ்', 'ட்', 'த்', 'ண்', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்', 'ற்', 'ன்']

# Path to the Tamil font
font_path = r"C:\Users\muthu\Muruga-LLM\lohit_tamil\Lohit-Tamil.ttf"  # Use 'r' for raw string to avoid escape sequences
font_size = 48

# Create a directory for the dataset
dataset_dir = "tamil_alphabets_dataset_with_mistakes"
os.makedirs(dataset_dir, exist_ok=True)

# Define a function to apply transformations (simulating human mistakes)
def apply_human_errors(image):
    # Randomly rotate the image
    if random.random() < 0.5:  # 50% chance of rotation
        angle = random.uniform(-15, 15)  # Small random rotations
        image = image.rotate(angle, fillcolor=(255, 255, 255))
    
    # Add random noise
    if random.random() < 0.3:  # 30% chance of adding noise
        noise = np.random.randint(0, 50, (image.size[1], image.size[0], 3), dtype='uint8')  # Random noise
        noise_img = Image.fromarray(np.clip(np.array(image) + noise, 0, 255).astype('uint8'))
        image = Image.blend(image, noise_img, alpha=0.3)  # Blend with noise
    
    # Randomly shift the image (simulate imperfect alignment)
    if random.random() < 0.5:  # 50% chance of translation
        max_shift = 5  # Max shift of 5 pixels
        x_shift = random.randint(-max_shift, max_shift)
        y_shift = random.randint(-max_shift, max_shift)
        image = ImageOps.expand(image, (x_shift, y_shift, max_shift - x_shift, max_shift - y_shift), fill=(255, 255, 255))
    
    return image

# Generate images with Tamil alphabets
for idx, char in enumerate(tamil_alphabets):
    for i in range(10):  # Generate 10 variations for each alphabet
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError as e:
            print(f"Error loading font: {e}")
            font = ImageFont.load_default()  # Fallback to default font if necessary
        
        # Draw the Tamil character
        draw.text((10, 25), char, font=font, fill=(0, 0, 0))  # Black text
        
        # Apply random transformations (mistakes)
        img_with_mistakes = apply_human_errors(img)
        
        # Save the image
        img_with_mistakes.save(f"{dataset_dir}/{char}_{i}.png")

print("Tamil alphabet images with human errors generated successfully!")
