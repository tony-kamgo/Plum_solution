import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class PlantAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant AI - Prune Quality Detector")
        self.root.geometry("1000x700")
        
        # Configuration des chemins
        self.yolov5_path = r'yolov5'
        self.output_dir = r'results'
        self.class_names = ["bruised", "cracked", "rotten", "spotted", "unaffected", "unripe"]
        
        # Variables
        self.image_path = None
        self.result_image = None
        self.detection_text = ""
        
        # Interface utilisateur
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de sélection d'image
        select_frame = ttk.LabelFrame(main_frame, text="Sélection d'image")
        select_frame.pack(fill=tk.X, pady=5)
        
        self.btn_select = ttk.Button(select_frame, text="Sélectionner une image", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.lbl_image_path = ttk.Label(select_frame, text="Aucune image sélectionnée")
        self.lbl_image_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Frame de résultats
        result_frame = ttk.LabelFrame(main_frame, text="Résultats de détection")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas pour l'image
        self.image_canvas = tk.Canvas(result_frame, bg='white')
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame pour les informations textuelles
        info_frame = ttk.Frame(result_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.lbl_detection = ttk.Label(info_frame, text="État de la prune: ", font=('Arial', 12))
        self.lbl_detection.pack(pady=10)
        
        self.txt_results = tk.Text(info_frame, height=15, width=30, state=tk.DISABLED)
        self.txt_results.pack(fill=tk.Y, expand=True)
        
        # Bouton de détection
        self.btn_detect = ttk.Button(main_frame, text="Analyser la prune", command=self.detect_plum, state=tk.DISABLED)
        self.btn_detect.pack(pady=10)
        
    def select_image(self):
        filetypes = (("Images", "*.png *.jpg *.jpeg"), ("Tous les fichiers", "*.*"))
        self.image_path = filedialog.askopenfilename(title="Sélectionner une image de prune", filetypes=filetypes)
        
        if self.image_path:
            self.lbl_image_path.config(text=self.image_path)
            self.display_image(self.image_path)
            self.btn_detect.config(state=tk.NORMAL)
            self.txt_results.config(state=tk.NORMAL)
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.config(state=tk.DISABLED)
            self.lbl_detection.config(text="État de la prune: ")
    
    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 600))  # Redimensionner pour l'affichage
            
            # Mettre à jour le canvas
            self.image_canvas.delete("all")
            self.result_image = ImageTk.PhotoImage(img)
            self.image_canvas.create_image(
                self.image_canvas.winfo_width()/2, 
                self.image_canvas.winfo_height()/2, 
                anchor=tk.CENTER, 
                image=self.result_image
            )
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
    
    def detect_plum(self):
        if not self.image_path:
            return
            
        # Désactiver le bouton pendant le traitement
        self.btn_detect.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            # Étape 1: Exécuter la détection avec YOLOv5 en utilisant le modèle personnalisé
            detect_command = (
                f'python {self.yolov5_path}\\detect.py '
                f'--weights {self.output_dir}\\best.pt '
                f'--img 640 --conf 0.25 '
                f'--source {self.image_path} '
                f'--project {self.output_dir} '
                f'--name detection_results '
                f'--save-txt --exist-ok'
            )
            os.system(detect_command)
            
            # Étape 2: Récupérer les résultats
            image_name = os.path.splitext(os.path.basename(self.image_path))[0]
            result_label_file = os.path.join(self.output_dir, 'detection_results', 'labels', f'{image_name}.txt')
            result_image_path = os.path.join(self.output_dir, 'detection_results', os.path.basename(self.image_path))
            
            if os.path.exists(result_image_path):
                # Afficher l'image avec les détections
                self.display_image(result_image_path)
                
                # Afficher les résultats textuels
                if os.path.exists(result_label_file):
                    with open(result_label_file, 'r') as file:
                        predictions = file.readlines()
                        detected_classes = set()
                        
                        self.txt_results.config(state=tk.NORMAL)
                        self.txt_results.delete(1.0, tk.END)
                        
                        for line in predictions:
                            class_id = int(line.strip().split()[0])
                            if class_id < len(self.class_names):
                                detected_classes.add(self.class_names[class_id])
                        
                        if detected_classes:
                            status = ", ".join(detected_classes)
                            self.lbl_detection.config(text=f"État de la prune: {status}")
                            self.txt_results.insert(tk.END, f"Résultats de détection:\n\n")
                            for cls in detected_classes:
                                self.txt_results.insert(tk.END, f"- {cls}\n")
                        else:
                            self.lbl_detection.config(text="État de la prune: Non détecté")
                            self.txt_results.insert(tk.END, "Aucun défaut détecté")
                        
                        self.txt_results.config(state=tk.DISABLED)
                else:
                    self.lbl_detection.config(text="État de la prune: Non détecté")
                    self.txt_results.config(state=tk.NORMAL)
                    self.txt_results.delete(1.0, tk.END)
                    self.txt_results.insert(tk.END, "Aucun défaut détecté")
                    self.txt_results.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Erreur lors de la détection: {e}")
            self.txt_results.config(state=tk.NORMAL)
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.insert(tk.END, f"Erreur: {str(e)}")
            self.txt_results.config(state=tk.DISABLED)
        finally:
            self.btn_detect.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantAIApp(root)
    root.mainloop()
