import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import shutil

# Définir les chemins pour les fichiers YOLOv5, le modèle, l'image et les données
yolov5_path = r'C:\Users\Moi\venv2\Lib\site-packages\yolov5'  # Chemin vers le répertoire YOLOv5
weights_path = r'C:\Users\Moi\venv2\yolov5s.pt'  # Modèle YOLOv5 pré-entrainé
image_path = r'C:\Users\Moi\venv2\intro\data_plum\images\train\cracked_plum_6.png'  # Image d'entrée
data_path = r'C:\Users\Moi\venv2\Lib\site-packages\yolov5\data\custom-data-plum.yaml'  # Fichier YAML des données
output_dir = r'C:\Users\Moi\venv2\results'  # Répertoire de sortie pour les résultats
labels_dir = os.path.join(output_dir, 'detection_results', 'labels')  # Dossier temporaire de sauvegarde pour les labels

# Noms des classes selon le fichier YAML
class_names = ["bruised", "cracked", "rotten", "spotted", "unaffected", "unripe"]

# Vérification de l'existence du fichier de données YAML
if not os.path.exists(data_path):
    print(f"Erreur : Le fichier de données YAML n'a pas été trouvé à {data_path}.")
else:
    # Étape 1 : Entraîner le modèle avec train.py en utilisant les données personnalisées
    print("Début de l'entraînement...")
    train_command = (
        f'python {yolov5_path}\\train.py --img 640 --batch 10 --epochs 3 '
        f'--data {data_path} --weights {weights_path} --project {output_dir} --name custom_training'
    )
    os.system(train_command)
    print("Entraînement terminé.")

    # Étape 2 : Effectuer une détection avec detect.py en utilisant le modèle entraîné
    print("Début de la détection...")
    detect_command = (
        f'python {yolov5_path}\\detect.py --weights {output_dir}\\custom_training\\weights\\best.pt '
        f'--img 640 --conf 0.25 --source {image_path} --project {output_dir} --name detection_results --save-txt --exist-ok'
    )
    os.system(detect_command)
    print("Détection terminée.")

    # Déplacement du fichier de résultats vers le dossier final
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_label_file = os.path.join(labels_dir, f'{image_name}.txt')
    final_label_path = os.path.join(output_dir, f'{image_name}_results.txt')
    
    if os.path.exists(result_label_file):
        shutil.move(result_label_file, final_label_path)
        print(f"Le fichier {image_name}.txt a été déplacé vers {final_label_path}.")
    else:
        print(f"Le fichier de résultats pour l'image {image_name} n'a pas été trouvé.")

    # Étape 3 : Charger l'image et afficher les résultats avec OpenCV et Matplotlib
    result_image_path = os.path.join(output_dir, 'detection_results', os.path.basename(image_path))
    if os.path.exists(result_image_path):
        img = cv2.imread(result_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB pour Matplotlib

        # Charger et afficher les prédictions
        if os.path.exists(final_label_path):
            with open(final_label_path, 'r') as file:
                predictions = file.readlines()
                for line in predictions:
                    # Chaque ligne contient la classe et les coordonnées normalisées
                    parts = line.strip().split()
                    class_id = int(parts[0])

                    # Les coordonnées normalisées de YOLO : (centre_x, centre_y, largeur, hauteur)
                    center_x, center_y, width, height = map(float, parts[1:])

                    # Calculer les coordonnées des coins pour le rectangle
                    img_height, img_width = img_rgb.shape[:2]
                    x1 = int((center_x - width / 2) * img_width)
                    y1 = int((center_y - height / 2) * img_height)
                    x2 = int((center_x + width / 2) * img_width)
                    y2 = int((center_y + height / 2) * img_height)

                    # Dessiner un rectangle autour de l'objet détecté
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Afficher le nom de la classe
                    label = f"{class_names[class_id]}" if class_id < len(class_names) else "Inconnu"
                    print(f"Objet détecté : {label}")

                    # Ajouter le nom de la classe au-dessus du cadre
                    cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Afficher l'image avec les résultats
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis('off')  # Ne pas afficher les axes
            plt.show()
        else:
            print(f"Le fichier {final_label_path} n'a pas été trouvé.")
    else:
        print("L'image résultante n'a pas été trouvée.")
