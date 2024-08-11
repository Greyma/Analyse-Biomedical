import cv2
from ultralytics import YOLO
import os
import torch
import numpy as np
import base64
from flask import jsonify

initial = 0 
def decouper(image) : 
    # Obtenir les dimensions de l'image
    height, width = image.shape[:2]

    # Calculer les dimensions de chaque sous-image
    sub_height = height // 2  # Diviser la hauteur en 2 parties égales
    sub_width = width // 2    # Diviser la largeur en 2 parties égales

    # Créer une liste pour stocker les sous-images
    sub_images = []

    # Découper l'image en 4 parties
    for i in range(2):  # 2 lignes
        for j in range(2):  # 2 colonnes
            # Déterminer les coordonnées de la sous-image
            x_start = j * sub_width
            y_start = i * sub_height
            x_end = x_start + sub_width
            y_end = y_start + sub_height
            
            # Extraire la sous-image
            sub_image = image[y_start:y_end, x_start:x_end]
            sub_images.append(sub_image)
    return sub_images

# Modèle YOLO
def upload_file(file,check):
    global initial 
    initial = initial + 1
    class_counts = {}
    if file:
        # Lire le fichier en mémoire
        file_bytes = np.frombuffer(file.read(), np.uint8)
        
        # Décoder les bytes en une image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        

    # Vérifier si l'image est chargée correctement
    if image is None:
        return "Image not found"
    
    if check == '2' : 
        model = YOLO("white-blood-cell-obg1g.pt")
        class_counts['your choice'] = "detect type of White Blood Cell"
    elif check == '1' : 
        model = YOLO("Count_50mb.pt")
        class_counts['your choice'] = "Counting different cellule"
    else : 
        model = YOLO("Count_RBC.pt")
        class_counts['your choice'] = "Detect and Calculle"

    #images = decouper(image)
    images = [image]

    if torch.cuda.is_available():
        model.to('cuda')
        results = model.predict(images, device='cuda')
        print("Avec GPU")
    else : 
        results = model.predict(images)
        print("Avec CPU")
    
    i = 0
    files = {}
    class_count = {}
    for result in results : 
        detections = result.boxes

        for detection in detections:
            class_id = int(detection.cls)  
            class_name = model.names[class_id] 
            
            if class_name in class_count:
                class_count[class_name] += 1
            else:
                class_count[class_name] = 1
        i = i+1

        # Drawing bounding boxes and saving images
        for class_name in set(model.names[int(detection.cls)] for detection in detections):
            image_copy = image.copy()
            for detection in detections:
                class_id = int(detection.cls)
                if model.names[class_id] == class_name:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    #cv2.putText(image_copy, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', image_copy)
        
            # Encoder le buffer en Base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            files[class_name] = image_base64

    class_counts['file'] = files
    class_counts['class_counts'] = class_count
    return jsonify(class_counts)