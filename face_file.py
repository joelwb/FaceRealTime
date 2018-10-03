import face_recognition
import cv2
import os
from PIL import Image, ImageDraw

scale = 1
path = os.path.dirname(os.getcwd())

img = face_recognition.load_image_file(path+"/teste.jpg")
face_locations = face_recognition.face_locations(img)

print(len(face_locations))

pil_image = Image.open(path+"/teste.jpg").convert("RGB")
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= int(1/scale)
        right *= int(1/scale)
        bottom *= int(1/scale)
        left *= int(1/scale)

        # Desenha o quadrado na face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))


pil_image.show()