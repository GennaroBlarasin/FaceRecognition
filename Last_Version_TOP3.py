# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:01:07 2021

@author: gennaro.blarasin
"""
import numpy as np
import time
import pandas as pd
import face_recognition
from pathlib import Path
from PIL import Image
import csv

# Este script va a generar un csv con todos los encodings de las imagenes una sola vez
# Este csv, será utilizado como entrada por el script de reconocimiento

keys = []
values = []


for image_path in Path("people_2").glob("*.png"):
    # Load an image to check
    unknown_image = face_recognition.load_image_file(image_path)
    if len(face_recognition.face_locations(unknown_image)) == 1:
        image_path_str = str(image_path)
        
        start = image_path_str.find("\\")
        end = image_path_str.find(".png")
        name_of_unknown_person = image_path_str[start+1:end].upper()
        keys.append(name_of_unknown_person)
        # Get the location of faces and face encodings for the current image
        face_encodings = face_recognition.face_encodings(unknown_image)
        values.append(face_encodings)
        #rellenar el diccionario
        print(name_of_unknown_person)
        
diccionario_caras = dict(zip(keys,values))
diccionario_caras_df = pd.DataFrame.from_dict(diccionario_caras)
diccionario_caras_df.to_csv('Encodings_90.csv',index = False)




t0 = time.time()

matchs = []
porcentajes = []
parecido2 = []
porcentaje2 = []
parecido3 = []
porcentaje3 = []
knownEncodings = pd.read_csv('Encodings_90.csv')


nombres_empleados = keys
nombre_porcentaje = pd.DataFrame(columns=['persona',
                                          'parecido1','porcentaje1',
                                          'parecido2','porcentaje2',
                                          'parecido3','porcentaje3'])
for nombre in nombres_empleados:
# Carga de la imagen a la cual le buscamos un parecido
    #known_image = face_recognition.load_image_file("BLARASIN_GENNARO.jpg")
    nombre_foto =nombre+".png"
    known_image = face_recognition.load_image_file(nombre_foto)
    #foto_str = "BLARASIN_GENNARO.jpg"
    foto_str = nombre_foto
    end = foto_str.find(".png")
    name_of_known_person = foto_str[0:end].upper()
    
    # Encode the known image
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    print("Comparando a ",name_of_known_person," con:")
    
    best_face_distance = 1.0
    best_face_image = None
    no_recognized_faces = []
    
    parecido_array = []
    porcentaje_array = []

    for i in range(len(knownEncodings.columns)):
        name_of_unknown_person = knownEncodings.columns[i]
        if name_of_unknown_person != name_of_known_person:
            print(name_of_unknown_person)
            parecido_array.append(name_of_unknown_person)
            db_image_encoding = knownEncodings.iloc[0,i].replace("     ",",")
            db_image_encoding = db_image_encoding.replace("    ",",")
            db_image_encoding = db_image_encoding.replace("   ",",")
            db_image_encoding = db_image_encoding.replace("  ",",")
            db_image_encoding = db_image_encoding.replace(" ",",")
            db_image_encoding = db_image_encoding.replace(",]","\n")
            db_image_encoding = db_image_encoding.translate({ord(i): None for i in '\n[]'})
            
            # Transformo string a lista. Separa elementos por ","
            db_image_encoding = db_image_encoding.split(",")
                    
            # Casteo lista de strings a array de floats
            db_image_encoding = np.array(db_image_encoding, dtype=np.float64)
            
            
            # saco distancia euclideana (norma)
            face_distance = np.linalg.norm(db_image_encoding - known_image_encoding)
            print("{:.2f}".format(round(float((1-face_distance)*100), 2)),"%")
            distancia_actual = "{:.2f}".format(round(float((1-face_distance)*100), 2)),"%"
            porcentaje_array.append((face_distance,name_of_unknown_person))
            
            # If this face is more similar to our known image than we've seen so far, save it
    
            if face_distance < best_face_distance:
                # Save the new best face distance
                best_face_distance = face_distance
                # Extract a copy of the actual face image itself so we can display it
                # best_face_image = unknown_image
                best_name_of_unknown_person = name_of_unknown_person
    porcentaje_array.sort(key=lambda tup: tup[0])
                
    print("Parecido a ",best_name_of_unknown_person ,"al ","{:.2f}".format(round((1-best_face_distance)*100, 2)),"%")
    matchs.append(best_name_of_unknown_person)
    porcentaje_individual = "{:.2f}".format(round((1-best_face_distance)*100,2))
    porcentajes.append(porcentaje_individual)
    parecido2.append(porcentaje_array[1][1])
    porcentaje2.append("{:.2f}".format(round((1-porcentaje_array[1][0])*100,2)))
    parecido3.append(porcentaje_array[2][1])
    porcentaje3.append("{:.2f}".format(round((1-porcentaje_array[2][0])*100,2)))
    t1 = time.time()
    total = t1-t0
    print("Tiempo de ejecución", total)
       
# dict_matchs = dict(zip(nombres_empleados,matchs))
df_matchs = pd.DataFrame()
df_matchs['nombre'] = nombres_empleados
df_matchs['parecido'] = matchs
df_matchs['porcentaje'] = porcentajes
df_matchs['parecido2'] = parecido2
df_matchs['porcentaje2'] = porcentaje2
df_matchs['parecido3'] = parecido3
df_matchs['porcentaje3'] = porcentaje3
df_matchs.to_csv('Parecidos.csv',index=False)
print(df_matchs)
print("Programa terminado. Ejecución Exitosa")