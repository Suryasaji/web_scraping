import os
import cv2
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
from urllib.parse import urljoin

# URL of the website
url = "https://sjcetpalai.ac.in/csehome/"

# Fetch HTML content
html_content = urlopen(url).read()

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find all image tags
images = soup.find_all("img")

# Directory where images will be saved
download_dir = "downloaded_images_with_faces"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Return True if faces are detected, else False
    return len(faces) > 0

# Download and filter images containing faces
for img in images:
    img_url = urljoin(url, img["src"])
    img_name = os.path.join(download_dir, os.path.basename(img_url))
    
    # Download the image
    urlretrieve(img_url, img_name)
    
    # Check if the image contains a face
    if detect_faces(img_name):
        print(f"Downloaded image with face: {img_name}")
    else:
        # If no face detected, remove the file
        os.remove(img_name)
        print(f"Removed non-face image: {img_name}")
