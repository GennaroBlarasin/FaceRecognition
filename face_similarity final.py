import face_recognition
from pathlib import Path
from PIL import Image

# Carga de la imagen a la cual le buscamos un parecido
known_image = face_recognition.load_image_file("test_face.jpg")

# Encode the known image
known_image_encoding = face_recognition.face_encodings(known_image)[0]

print(known_image_encoding)

# Variables to keep track of the most similar face match we've found
best_face_distance = 1.0
best_face_image = None
no_recognized_faces = []

# Loop over all the images we want to check for similar people
#for image_path in Path("people").glob("*.png"):
for image_path in Path("people_1").glob("*.png"):
    # Load an image to check
    unknown_image = face_recognition.load_image_file(image_path)
    print(image_path)
    # Get the location of faces and face encodings for the current image
    face_encodings = face_recognition.face_encodings(unknown_image)
    #print(type(face_encodings))
    #print(face_encodings)
    # Get the face distance between the known person and all the faces in this image
    #face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]
    face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)

    print(face_distance)
    # If this face is more similar to our known image than we've seen so far, save it
    if face_distance.size > 0: 
        if face_distance < best_face_distance:
            # Save the new best face distance
            best_face_distance = face_distance
            # Extract a copy of the actual face image itself so we can display it
            best_face_image = unknown_image
    else:
        no_recognized_faces.append(str(image_path))
        
# Display the face image that we found to be the best match!
print("Las caras de ",no_recognized_faces,"no fueron encontrada, por favor reenviar.")
print("Parecido al ",(1-best_face_distance[0])*100,"%")
pil_image = Image.fromarray(best_face_image)
pil_image.show()
