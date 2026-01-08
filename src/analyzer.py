import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, scrolledtext

class NeuralSight_HumanAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURALSIGHT | Human Action Intelligence")
        self.root.geometry("1100x850")
        self.root.configure(bg="#05070A") # Matching the main dark theme

        # Device Configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths Configuration (Assuming script is in src/)
        self.vlm_path = os.path.join("..", "models", "neural_sight_v1")
        self.yolo_path = os.path.join("..", "models", "yolov8n.pt")

        # Load Models
        self.load_models()

        # Build UI
        self.setup_ui()

    def load_models(self):
        print("Initializing Neural Engines...")
        try:
            # Load YOLOv8 for human detection
            self.yolo_model = YOLO(self.yolo_path)
            
            # Load Fine-tuned BLIP
            self.processor = BlipProcessor.from_pretrained(self.vlm_path)
            self.model = BlipForConditionalGeneration.from_pretrained(self.vlm_path).to(self.device)
            print("Neural Engines Active.")
        except Exception as e:
            print(f"Initialization Error: {e}")

    def setup_ui(self):
        # Header Section
        header = tk.Frame(self.root, bg="#0D1117", height=50)
        header.pack(fill="x", side="top")
        tk.Label(header, text="HUMAN ACTION ANALYZER", font=("Avenir Next", 16, "bold"), 
                 fg="#00F5FF", bg="#0D1117").pack(pady=10)

        # Left Panel (Visual Output)
        self.left_frame = tk.Frame(self.root, bg="#05070A")
        self.left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=10)

        self.btn_select = tk.Button(self.left_frame, text="⚡ SELECT IMAGE FOR ANALYSIS", command=self.process_image,
                                   bg="#1F6FEB", fg="white", font=("Inter", 11, "bold"), 
                                   relief="flat", pady=12, cursor="hand2")
        self.btn_select.pack(fill="x", pady=(0, 10))

        # Canvas for result image
        self.image_display = tk.Label(self.left_frame, bg="#0D1117", highlightthickness=1, highlightbackground="#1F6FEB")
        self.image_display.pack(pady=10, expand=True, fill="both")

        # Right Panel (Data Insights)
        self.right_frame = tk.Frame(self.root, bg="#0D1117", width=400)
        self.right_frame.pack(side="right", fill="both", padx=10, pady=10)

        tk.Label(self.right_frame, text="NEURAL INSIGHTS", font=("Inter", 12, "bold"), 
                 bg="#0D1117", fg="#FFD700").pack(pady=15)

        self.result_area = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, bg="#05070A",
                                                   fg="#C9D1D9", insertbackground="white",
                                                   width=45, height=35, font=("Consolas", 10))
        self.result_area.pack(padx=10, pady=10, fill="both", expand=True)

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.result_area.delete('1.0', tk.END)
        self.result_area.insert(tk.END, "> STARTING NEURAL SCAN...\n")
        self.root.update()

        # 1. Detection Phase (Focusing only on class 0: Person)
        img = Image.open(file_path).convert("RGB")
        results = self.yolo_model.predict(source=img, classes=[0], conf=0.45)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        draw = ImageDraw.Draw(img)
        
        # Font Configuration
        try:
            # Looks for font in common system paths
            font = ImageFont.truetype("arial.ttf", 25)
        except:
            font = ImageFont.load_default()

        self.result_area.delete('1.0', tk.END)

        if len(boxes) == 0:
            self.result_area.insert(tk.END, "[!] STATUS: NO HUMAN TARGETS DETECTED.")
        else:
            self.result_area.insert(tk.END, f"[+] SCAN COMPLETE: {len(boxes)} TARGET(S) FOUND\n\n")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                person_id = i + 1

                # Visual Feedback: Bounding Box & ID
                draw.rectangle([x1, y1, x2, y2], outline="#00F5FF", width=4)
                draw.rectangle([x1, y1-35, x1+80, y1], fill="#00F5FF")
                draw.text((x1+10, y1-35), f"ID:{person_id}", fill="black", font=font)

                # Interpretation Phase: BLIP Action Analysis
                crop = img.crop((x1, y1, x2, y2))
                inputs = self.processor(crop, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    # Using beam search for more descriptive output
                    out = self.model.generate(**inputs, max_new_tokens=50, num_beams=5)
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                # Log results to panel
                self.result_area.insert(tk.END, f"● NODE ID {person_id:02d}\n", "id_tag")
                self.result_area.insert(tk.END, f"Action: {caption.capitalize()}\n")
                self.result_area.insert(tk.END, "-"*40 + "\n")
                self.result_area.tag_config("id_tag", font=("Consolas", 11, "bold"), foreground="#00F5FF")
                self.root.update()

        # Resize and display final image
        display_img = img.copy()
        display_img.thumbnail((700, 700))
        img_tk = ImageTk.PhotoImage(display_img)
        self.image_display.configure(image=img_tk)
        self.image_display.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralSight_HumanAnalyzer(root)
    root.mainloop()