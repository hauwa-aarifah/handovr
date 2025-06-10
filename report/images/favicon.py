# Python script to create favicon
from PIL import Image
import os

# Open your logo
img = Image.open('images/handovr-logo.png')

# Create multiple sizes for better compatibility
sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]

# Create ICO file with multiple resolutions
img.save('images/favicon.ico', format='ICO', sizes=sizes)