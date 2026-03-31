import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2
import os
from joblib import load

import config
import utils

class LungCancerApp:
    def __init__(self, master):
        self.master = master
        master.title("Hệ Thống Chẩn Đoán Ung Thư Phổi - SVM & Features")
        master.geometry("900x700")
        
        # Load model, scaler và PCA
        try:
            # Kiểm tra file tồn tại trước khi load
            model_files = [config.MODEL_NAME, config.SCALER_NAME, config.PCA_NAME]
            if not all(os.path.exists(f"{config.MODEL_DIR}/{name}") for name in model_files):
                raise FileNotFoundError(f"Không tìm thấy các file model trong thư mục '{config.MODEL_DIR}/'")
                
            self.model = load(f"{config.MODEL_DIR}/{config.MODEL_NAME}")
            self.scaler = load(f"{config.MODEL_DIR}/{config.SCALER_NAME}")
            self.pca = load(f"{config.MODEL_DIR}/{config.PCA_NAME}")
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", f"Không thể tải mô hình: {str(e)}")
            self.master.destroy()
            return
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Phần điều khiển ---
        upload_frame = ttk.LabelFrame(main_frame, text="Thao tác", padding=15)
        upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_btn = ttk.Button(upload_frame, text="Chọn Ảnh CT Scan", command=self.load_image)
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        # --- Phần hiển thị ảnh ---
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Ảnh gốc
        orig_container = ttk.LabelFrame(images_frame, text="1. Ảnh Gốc (Original)", padding=10)
        orig_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.image_label = ttk.Label(orig_container)
        self.image_label.pack(expand=True)
        
        # Ảnh sau xử lý CLAHE
        proc_container = ttk.LabelFrame(images_frame, text="2. Sau khi lọc CLAHE (Pre-processed)", padding=10)
        proc_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.processed_image_label = ttk.Label(proc_container)
        self.processed_image_label.pack(expand=True)
        
        # --- Kết quả dự đoán ---
        result_frame = ttk.LabelFrame(main_frame, text="Kết Quả Phân Tích", padding=15)
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_text = tk.StringVar(value="Vui lòng chọn ảnh để bắt đầu...")
        self.result_label = ttk.Label(
            result_frame, 
            textvariable=self.result_text, 
            font=('Helvetica', 16, 'bold'),
            anchor="center"
        )
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tệp ảnh", "*.jpg *.jpeg *.png")])
        if not file_path: return
            
        try:
            # 1. Đọc ảnh xám gốc
            img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None: raise ValueError("Ảnh không hợp lệ")
            
            # Dùng hàm preprocess_img từ utils
            img_final = utils.preprocess_img(img_gray) 
            
            # 2. Hiển thị ảnh lên giao diện
            # Ảnh gốc (giữ nguyên để người dùng xem)
            orig_pil = Image.open(file_path).convert("RGB")
            orig_pil.thumbnail(config.UI_IMAGE_SIZE)
            photo_orig = ImageTk.PhotoImage(orig_pil)
            self.image_label.configure(image=photo_orig)
            self.image_label.image = photo_orig
            
            # Ảnh sau xử lý (Hiển thị bản img_final để kiểm tra bước Masking)
            proc_pil = Image.fromarray((img_final * 255).astype(np.uint8))
            proc_pil.thumbnail(config.UI_IMAGE_SIZE)
            photo_proc = ImageTk.PhotoImage(proc_pil)
            self.processed_image_label.configure(image=photo_proc)
            self.processed_image_label.image = photo_proc
            
            # 3. Trích xuất đặc trưng (Dùng img_final đã chuẩn hóa)
            features = self.extract_combined_features(img_final)
            
            # 4. Dự đoán và cập nhật giao diện
            p_normal, p_cancer, confidence = self.predict_image_with_confidence(features)
            
            # 5. Cập nhật giao diện theo confidence threshold
            if confidence < config.MODEL_CONFIDENCE_THRESHOLD:
                # Confidence thấp → không đủ tự tin
                diff = abs(p_cancer - p_normal)
                if diff < 0.1:
                    msg = f"⚠️ KHÔNG RÕ RÀNG: {p_normal*100:.1f}% Bình thường vs {p_cancer*100:.1f}% Ung thư\n(Mô hình không đủ tự tin - chênh lệch < 10%)"
                elif p_cancer > p_normal:
                    msg = f"⚠️ CÓ KHUYNH HƯỚNG UNG THƯ: {p_cancer*100:.1f}%\n(Tin cậy chỉ {confidence*100:.1f}% < {config.MODEL_CONFIDENCE_THRESHOLD*100:.0f}% - Có thể do chất lượng ảnh)"
                else:
                    msg = f"⚠️ CÓ KHUYNH HƯỚNG BÌNH THƯỜNG: {p_normal*100:.1f}%\n(Tin cậy chỉ {confidence*100:.1f}% < {config.MODEL_CONFIDENCE_THRESHOLD*100:.0f}% - Có thể do chất lượng ảnh)"
                color = "orange"
            elif p_cancer > p_normal:
                msg = f"🚨 CẢNH BÁO: PHÁT HIỆN UNG THƯ\n{p_cancer*100:.1f}% (Tin cậy: {confidence*100:.1f}%)"
                color = "red"
            else:
                msg = f"✓ TÌNH TRẠNG: BÌNH THƯỜNG\n{p_normal*100:.1f}% (Tin cậy: {confidence*100:.1f}%)"
                color = "green"
                
            self.result_text.set(msg)
            self.result_label.configure(foreground=color)
            
        except Exception as e:
            messagebox.showerror("Lỗi xử lý", str(e))

    def extract_combined_features(self, img_normalized):
        """Trích xuất HOG + LBP features đồng bộ với training"""
        hog_feat = utils.extract_hog_features(img_normalized)
        lbp_feat = utils.extract_lbp_features(img_normalized)
        return np.hstack([hog_feat, lbp_feat])

    def predict_image_with_confidence(self, features):
        """Dự đoán với confidence threshold"""
        scaled = self.scaler.transform([features])
        pca_feat = self.pca.transform(scaled)
        proba = self.model.predict_proba(pca_feat)[0]
        confidence = max(proba)  # Độ tin cậy = xác suất cao nhất
        return proba[0], proba[1], confidence  # p_normal, p_cancer, confidence

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerApp(root)
    root.mainloop()