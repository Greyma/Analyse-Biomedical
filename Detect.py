from flask import Flask, request, render_template, jsonify , send_from_directory
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os
import torch
from GG import predict

app = Flask(__name__)
CORS(app)
initial = 0 
# Configuration du dossier de téléchargement
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['Analyse_FOLDER'] = 'images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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
def upload_file():
    class_counts = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            form_data = request.form.to_dict()
            check = form_data['selected']   
            if check == '2' : 
                model = YOLO("white-blood-cell-obg1g.pt")
                class_counts['your choice'] = "detect type of White Blood Cell"
            elif check == '1' : 
                model = YOLO("Count_50mb.pt")
                class_counts['your choice'] = "Counting different cellule"
            else : 
                class_counts['class_counts'] = {'Calcule red Blood Cell' : predict(filepath) }
                class_counts['your choice'] = "Calculle RBC"
                return class_counts

            # Charger l'image
            image = cv2.imread(filepath)

            # Vérifier si l'image est chargée correctement
            if image is None:
                return "Image not found"
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
            app.config[f'FOLDER{initial}'] = f"images/D{initial}"
            os.makedirs(app.config[f'FOLDER{initial}'], exist_ok=True)
            for result in results : 
                detections = result.boxes  # Obtenir les détections
                class_counts[f'results{i}'] = result.path   
                for detection in detections:
                    class_id = int(detection.cls)  # ID de la classe détectée
                    class_name = model.names[class_id]  # Nom de la classe
                    # Mettre à jour le compte pour cette classe
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

                    output_filepath = os.path.join(f"images/D{initial}", f"image{class_name}.jpg")
                    cv2.imwrite(output_filepath, image_copy)
                    files[class_name] = output_filepath

    class_counts['file'] = files
    class_counts['class_counts'] = class_count
    return class_counts

@app.route('/kimathb', methods=['POST'])
def returne () :
    global initial 
    initial = initial + 1
    #return jsonify({'message': 'Hello, World!'})
    return jsonify(upload_file())
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/images/<path:filename>')
def detect_file(filename):
    return send_from_directory(app.config['Analyse_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)

