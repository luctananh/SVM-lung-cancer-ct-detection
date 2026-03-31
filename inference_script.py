import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2
import os
from joblib import load
from skimage.feature import hog, local_binary_pattern

def preprocess_img(img, img_size=(512, 512)):
    # 1. Resize đồng nhất
    img = cv2.resize(img, img_size)
    
    # 2. Xử lý độ sáng tự động (Auto-Gain Control)
    # Tính độ sáng trung bình hiện tại của tấm ảnh
    mean_brightness = np.mean(img)
    
    # Mục tiêu đưa độ sáng về mức 105 (điểm tối ưu dựa trên Boxplot tập dataset)
    target_mean = 105.0
    
    # Tính toán hệ số alpha để điều chỉnh (Target / Current)
    # Ví dụ: Nếu ảnh quá tối (mean=80), alpha sẽ là ~1.3 -> Tăng sáng
    # Nếu ảnh quá sáng (mean=150), alpha sẽ là ~0.7 -> Giảm sáng
    alpha = target_mean / mean_brightness
    
    # Giới hạn alpha trong khoảng an toàn [0.7, 1.3] để tránh làm biến dạng ảnh quá mức
    alpha = max(0.85, min(alpha, 1.3))
    
    # Áp dụng thay đổi độ sáng và tương phản
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, 0)
    
    # 3. Cân bằng độ tương phản cục bộ (CLAHE)
    # Giúp làm nổi bật các nốt mờ, khối u sau khi đã cân bằng độ sáng tổng thể
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # 4. Lọc nhiễu nhẹ
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 5. Masking - Loại bỏ nhiễu nền ngoài vùng phổi
    _, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    return img / 255.0 # Chuẩn hóa về [0, 1] cho mô hình SVM

class LungCancerApp:
    def __init__(self, master):
        self.master = master
        master.title("Hệ Thống Chẩn Đoán Ung Thư Phổi - SVM & CLAHE")
        master.geometry("900x700")
        
        # Khởi tạo bộ lọc CLAHE (Phải khớp tham số với lúc Train)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Load model, scaler và PCA
        try:
            # Kiểm tra file tồn tại trước khi load
            if not all(os.path.exists(f"models/{name}.joblib") for name in ["lung_cancer_svm", "scaler", "pca"]):
                raise FileNotFoundError("Không tìm thấy các file model trong thư mục 'models/'")
                
            self.model = load("models/lung_cancer_svm.joblib")
            self.scaler = load("models/scaler.joblib")
            self.pca = load("models/pca.joblib")
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
            
            # --- SỬA TẠI ĐÂY: Dùng hàm preprocess_img chuẩn ---
            # Hàm này đã bao gồm: Resize, CLAHE, Blur và Masking
            img_final = preprocess_img(img_gray) 
            
            # 2. Hiển thị ảnh lên giao diện
            # Ảnh gốc (giữ nguyên để người dùng xem)
            orig_pil = Image.open(file_path).convert("RGB")
            orig_pil.thumbnail((350, 350))
            photo_orig = ImageTk.PhotoImage(orig_pil)
            self.image_label.configure(image=photo_orig)
            self.image_label.image = photo_orig
            
            # Ảnh sau xử lý (Hiển thị bản img_final để kiểm tra bước Masking)
            # Vì img_final là float [0, 1], cần nhân 255 và chuyển về uint8 để hiển thị
            proc_pil = Image.fromarray((img_final * 255).astype(np.uint8))
            proc_pil.thumbnail((350, 350))
            photo_proc = ImageTk.PhotoImage(proc_pil)
            self.processed_image_label.configure(image=photo_proc)
            self.processed_image_label.image = photo_proc
            
            # 3. Trích xuất đặc trưng (Dùng img_final đã chuẩn hóa)
            features = self.extract_features_logic(img_final) # img_final đã là /255.0 rồi
            
            # 4. Dự đoán và cập nhật giao diện (Giữ nguyên phần dưới của bạn)
            p_normal, p_cancer = self.predict_image(features)
            
            # 5. Cập nhật giao diện
            if p_cancer > p_normal:
                msg = f"CẢNH BÁO: PHÁT HIỆN UNG THƯ ({p_cancer*100:.1f}%)"
                color = "red"
            else:
                msg = f"TÌNH TRẠNG: BÌNH THƯỜNG ({p_normal*100:.1f}%)"
                color = "green"
                
            self.result_text.set(msg)
            self.result_label.configure(foreground=color)
            
        except Exception as e:
            messagebox.showerror("Lỗi xử lý", str(e))

    def extract_features_logic(self, img_normalized):
        """Logic trích xuất đặc trưng đồng bộ với file Training"""
        # HOG
        hog_feat = hog(
            img_normalized, orientations=16, pixels_per_cell=(16, 16), 
            cells_per_block=(3, 3), channel_axis=None
        )
        
        # LBP
        lbp = local_binary_pattern(img_normalized, 24, 3, method="uniform")
        # Khớp số bins: np.arange(0, 27) tạo ra 26 bins cho method='uniform' với P=24
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return np.hstack([hog_feat, hist])

    def predict_image(self, features):
        scaled = self.scaler.transform([features])
        pca_feat = self.pca.transform(scaled)
        proba = self.model.predict_proba(pca_feat)[0]
        return proba[0], proba[1] # p_normal, p_cancer

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerApp(root)
    root.mainloop()