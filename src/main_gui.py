import torch
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import time

class NeuralSight_Final:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURALSIGHT | Multimodal Intelligence Station")
        self.root.geometry("1400x950")
        self.root.configure(bg="#05070A")

        # --- AI Engine Path Configuration ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Absolute path handling for models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vlm_path = os.path.abspath(os.path.join(current_dir, "..", "models", "neural_sight_v1"))
        yolo_path = os.path.abspath(os.path.join(current_dir, "..", "models", "yolov8n.pt"))

        if not self.load_engines(vlm_path, yolo_path):
            return 

        self.setup_styles()
        self.build_layout()
        self.is_processing = False

    def load_engines(self, vlm_path, yolo_path):
        """Initializes neural models with high-level error handling"""
        try:
            print(f"[*] Deploying YOLOv8 from: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            
            print(f"[*] Deploying BLIP VLM from: {vlm_path}")
            self.vlm_processor = BlipProcessor.from_pretrained(vlm_path)
            self.vlm_model = BlipForConditionalGeneration.from_pretrained(vlm_path).to(self.device)
            
            print("✅ Neural Engines Deployed Successfully.")
            return True
        except Exception as e:
            messagebox.showerror("Engine Error", f"Failed to load models.\nPath: {vlm_path}\nError: {str(e)}")
            self.root.destroy()
            return False

    def setup_styles(self):
        """Styles the UI with a professional dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", background="#0D1117", foreground="#C9D1D9", fieldbackground="#0D1117", 
                        borderwidth=0, font=("Segoe UI", 10), rowheight=40) # Increased row height
        style.configure("Treeview.Heading", background="#161B22", foreground="#00F5FF", relief="flat", font=("Segoe UI", 11, "bold"))
        style.configure("TProgressbar", thickness=10, background="#00F5FF", troughcolor="#0D1117")

    def toggle_all_filters(self):
        """Selection logic for object classes"""
        any_unselected = any(not var.get() for var in self.check_vars.values())
        new_state = True if any_unselected else False
        for var in self.check_vars.values():
            var.set(new_state)
        self.btn_select_all.config(text="UNSELECT ALL" if new_state else "SELECT ALL CATEGORIES")

    def build_layout(self):
        """Constructs the station interface"""
        # --- HEADER ---
        header = tk.Frame(self.root, bg="#0D1117", height=60)
        header.pack(fill="x", side="top")
        tk.Label(header, text="NEURALSIGHT", font=("Avenir Next", 24, "bold"), fg="#00F5FF", bg="#0D1117").pack(side="left", padx=25)
        self.status_label = tk.Label(header, text="SYSTEM READY", font=("Consolas", 10), fg="#00FF41", bg="#0D1117")
        self.status_label.pack(side="right", padx=25)

        # --- SIDEBAR ---
        self.sidebar = tk.Frame(self.root, width=300, bg="#0D1117", padx=15, pady=20)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="MODE SELECTION", font=("Inter", 9, "bold"), fg="#8B949E", bg="#0D1117").pack(anchor="w", pady=(0,5))
        self.global_mode_var = tk.BooleanVar(value=True)
        self.cb_global = tk.Checkbutton(self.sidebar, text="GLOBAL INTERPRETATION", variable=self.global_mode_var,
                                       bg="#0D1117", fg="#FFD700", selectcolor="#05070A",
                                       activebackground="#0D1117", font=("Consolas", 9, "bold"), cursor="hand2")
        self.cb_global.pack(anchor="w", pady=(0, 20))

        tk.Label(self.sidebar, text="DETECTION FILTERS", font=("Inter", 9, "bold"), fg="#8B949E", bg="#0D1117").pack(anchor="w", pady=(0,5))
        self.btn_select_all = tk.Button(self.sidebar, text="SELECT ALL CATEGORIES", command=self.toggle_all_filters,
                                        bg="#1F6FEB", fg="white", font=("Inter", 8, "bold"), relief="flat",
                                        activebackground="#58A6FF", cursor="hand2", pady=8)
        self.btn_select_all.pack(fill="x", pady=(0, 10))
        
        self.filter_canvas = tk.Canvas(self.sidebar, bg="#0D1117", highlightthickness=0)
        self.filter_scroll = ttk.Scrollbar(self.sidebar, orient="vertical", command=self.filter_canvas.yview)
        self.scroll_frame = tk.Frame(self.filter_canvas, bg="#0D1117")
        self.scroll_frame.bind("<Configure>", lambda e: self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all")))
        self.filter_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.filter_canvas.configure(yscrollcommand=self.filter_scroll.set)
        self.filter_canvas.pack(side="left", fill="both", expand=True)
        self.filter_scroll.pack(side="right", fill="y")

        self.check_vars = {}
        for idx, name in self.yolo_model.names.items():
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.scroll_frame, text=f"{idx:02d} | {name.upper()}", variable=var, 
                               bg="#0D1117", fg="#58A6FF", selectcolor="#05070A", 
                               activebackground="#0D1117", font=("Consolas", 8))
            cb.pack(anchor="w", pady=1)
            self.check_vars[name] = var

        # --- MAIN VIEWPORT ---
        self.main_content = tk.Frame(self.root, bg="#05070A", padx=30)
        self.main_content.pack(side="right", fill="both", expand=True)

        self.btn_action = tk.Button(self.main_content, text="▶ START NEURAL SCAN", command=self.start_analysis, 
                                   bg="#00F5FF", fg="#05070A", font=("Inter", 12, "bold"), 
                                   relief="flat", padx=60, pady=15, cursor="hand2")
        self.btn_action.pack(pady=25)

        self.canvas_stage = tk.Canvas(self.main_content, bg="#0D1117", height=450, 
                                      highlightthickness=1, highlightbackground="#1F6FEB")
        self.canvas_stage.pack(fill="x", pady=5)

        self.progress = ttk.Progressbar(self.main_content, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=20)

        # Updated Table View with stretched column to prevent "..."
        table_container = tk.Frame(self.main_content, bg="#0D1117")
        table_container.pack(fill="both", expand=True, pady=(5, 20))

        self.tree = ttk.Treeview(table_container, columns=("id", "entity", "desc"), show="headings")
        self.tree.heading("id", text="NODE ID")
        self.tree.heading("entity", text="OBJECT CLASS")
        self.tree.heading("desc", text="NEURAL INSIGHT (VLM DESCRIPTION)")
        
        self.tree.column("id", width=100, anchor="center", stretch=False)
        self.tree.column("entity", width=150, anchor="center", stretch=False)
        self.tree.column("desc", width=850, anchor="w", stretch=True) # Stretched to max
        
        self.tree.tag_configure('global', background="#161B22", foreground="#FFD700") 
        
        table_scroll = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=table_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")

    def get_vlm_description(self, image, prompt="A photo of"):
        """Generates a clean description and strips keyword metadata"""
        inputs = self.vlm_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Increased tokens slightly for more descriptive insights
            out = self.vlm_model.generate(**inputs, max_new_tokens=80) 
        
        raw_text = self.vlm_processor.decode(out[0], skip_special_tokens=True)
        
        # Comprehensive cleaning logic
        clean_text = raw_text.replace(prompt, "").strip()
        
        # Remove anything from "Keywords:" onwards to keep the sentence clean
        if "Keywords:" in clean_text:
            clean_text = clean_text.split("Keywords:")[0].strip()
        if "seo alt text" in clean_text.lower():
            clean_text = clean_text.lower().replace("seo alt text :", "").strip()

        return clean_text

    def start_analysis(self):
        if self.is_processing: return
        path = filedialog.askopenfilename()
        if not path: return

        self.is_processing = True
        self.status_label.config(text="NEURAL SCANNING...", fg="#FFD700")
        for i in self.tree.get_children(): self.tree.delete(i)
        
        original_img = Image.open(path).convert("RGB")
        self.update_image_on_canvas(original_img)
        self.root.update()

        self.trigger_scan_animation()
        self.progress['value'] = 0

        # Phase 1: Global Scene
        if self.global_mode_var.get():
            self.status_label.config(text="ANALYZING GLOBAL CONTEXT...")
            global_desc = self.get_vlm_description(original_img, "This is a photo of")
            self.tree.insert("", "end", values=("GLOBAL_00", "ENTIRE SCENE", global_desc.upper()), tags=('global',))
            self.progress['value'] = 20
            self.root.update()

        # Phase 2: Object Detection
        active_filters = [n for n, v in self.check_vars.items() if v.get()]
        if active_filters:
            self.status_label.config(text="EXTRACTING OBJECT INSIGHTS...", fg="#00F5FF")
            results = self.yolo_model(path)[0]
            cv_img = cv2.imread(path)
            target_boxes = [b for b in results.boxes if self.yolo_model.names[int(b.cls[0])] in active_filters]
            
            obj_id = 0
            for box in target_boxes:
                obj_id += 1
                cls_name = self.yolo_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                crop = original_img.crop((max(0, x1), max(0, y1), min(original_img.width, x2), min(original_img.height, y2)))
                desc = self.get_vlm_description(crop, "Object details:")
                
                # Inserting data into the Treeview
                self.tree.insert("", "end", values=(f"NODE_{obj_id:02d}", cls_name.upper(), desc.capitalize()))
                
                # CV2 Visualization
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 245, 255), 2)
                cv2.rectangle(cv_img, (x1, y1-25), (x1+85, y1), (0, 245, 255), -1)
                cv2.putText(cv_img, f"ID:{obj_id:02d}", (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                self.progress['value'] += (80 / max(1, len(target_boxes)))
                self.root.update()

            final_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            self.update_image_on_canvas(final_pil)
        
        self.status_label.config(text="ANALYSIS COMPLETE", fg="#00FF41")
        self.progress['value'] = 100
        self.is_processing = False

    def update_image_on_canvas(self, pil_img):
        c_w = self.canvas_stage.winfo_width()
        c_h = self.canvas_stage.winfo_height()
        if c_w < 10: c_w, c_h = 900, 450
        pil_img.thumbnail((c_w, c_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas_stage.delete("all")
        self.canvas_stage.create_image(c_w//2, c_h//2, image=self.tk_img)

    def trigger_scan_animation(self):
        w = self.canvas_stage.winfo_width()
        h = self.canvas_stage.winfo_height()
        line = self.canvas_stage.create_rectangle(0, 0, w, 3, fill="#00F5FF", outline="#00F5FF")
        for i in range(0, h, 10):
            self.canvas_stage.coords(line, 0, i, w, i+3)
            self.root.update()
            time.sleep(0.005)
        self.canvas_stage.delete(line)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralSight_Final(root)
    root.mainloop()