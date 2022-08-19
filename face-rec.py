import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display


import os
from os import listdir

from cv2 import *
from cv2 import (VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite)



# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

known_face_encodings=[]
known_face_names=[]
folder_dir = "/Users/alphinkj/Documents/Face/pictures"

for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        print(images)
        image_load=face_recognition.load_image_file("pictures/"+images)
        image_face_encoding=face_recognition.face_encodings(image_load)[0]
        known_face_encodings.append(image_face_encoding)
        known_face_names.append(images)

# Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# # Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# alphin_image=face_recognition.load_image_file("alphin.jpg")
# alphin_image_encoding = face_recognition.face_encodings(alphin_image)[0]



# Create arrays of known face encodings and their names
# known_face_encodings = [
#     obama_face_encoding,
#     biden_face_encoding,
#     # alphin_image_encoding
# ]
# print(type(known_face_encodings))
# known_face_names = [
#     "Barack Obama",
#     "Joe Biden",
#     "Alphin K Jose"
# ]
print('Learned encoding for', len(known_face_encodings), 'images.')


# Load an image with an unknown face
# cam_port = 0
# cam = VideoCapture(cam_port)
# result, image = cam.read()
# if result:
  
#     # showing result, it take frame name and image 
#     # output
#     # imshow("GeeksForGeeks", image)
  
#     # # saving image in local storage
#     imwrite("two_people.jpg", image)
#     unknown_image = face_recognition.load_image_file("two_people.jpg")
  
#     # If keyboard interrupt occurs, destroy image 
#     # window
    
  
# # If captured image is corrupted, moving to else part
# else:
#     unknown_image = face_recognition.load_image_file("two_people.jpg")
#     print("Taking userdefined image")


unknown_image = face_recognition.load_image_file("two_people.jpg")

# Find all the faces and face encodings in the unknown image

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
# display(pil_image)
pil_image.show()

# load= Image.open(pil_image)
# load.show()
