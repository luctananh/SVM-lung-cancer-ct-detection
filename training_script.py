# ----------- [1] Cài đặt & Import thư viện -----------
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# ----------- [2] Thiết lập Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    
    # Giới hạn alpha trong khoảng an toàn [0.85, 1.3] để tránh làm biến dạng ảnh quá mức
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

# ----------- [3] Hàm xử lý dữ liệu và trích xuất -----------
def load_dataset(data_dir, img_size=(512, 512)):
    """Tải và tiền xử lý dataset sử dụng hàm preprocess_img chuẩn."""
    images, labels = [], []
    
    for class_name in ["normal", "cancer"]:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Thiếu thư mục: {class_dir}")
            
        for img_name in tqdm(os.listdir(class_dir), desc=f"Đang tải {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Kiểm tra ảnh lỗi
            if img is None or img.std() < 5: 
                continue 

            #  Gọi hàm preprocess_img đã định nghĩa ở trên ---
            img_final = preprocess_img(img, img_size)
            
            images.append(img_final)
            labels.append(0 if class_name == "normal" else 1)
            
    return np.array(images), np.array(labels)

def extract_features(images):
    """Trích xuất kết hợp đặc trưng HOG và LBP (Mặc định)."""
    features = []
    for img in tqdm(images, desc="Trích xuất đặc trưng HOG & LBP"):
        # HOG
        hog_feat = hog(img, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(3, 3), channel_axis=None)
        
        # LBP
        lbp = local_binary_pattern(img, 24, 3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Nối HOG và LBP
        features.append(np.hstack([hog_feat, hist]))
        
    return np.array(features)

# ----------- [4] Hàm chính (Training Pipeline) -----------
def main():
    DATA_DIR = "dataset"
    os.makedirs("models", exist_ok=True)

    # 1. Tải dữ liệu
    logging.info("Bắt đầu tải dữ liệu...")
    images, labels = load_dataset(DATA_DIR)
    
    # 2. Trích xuất đặc trưng
    X = extract_features(images)
    y = labels

    # 3. Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Chuẩn hóa dữ liệu (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. Giảm chiều dữ liệu (PCA)
    pca = PCA(n_components=100, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # 6. Huấn luyện mô hình SVM (Mặc định)
    logging.info("Bắt đầu huấn luyện mô hình SVM...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    model.fit(X_train, y_train)

    # 7. Đánh giá mô hình
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nĐộ chính xác - Train: {train_acc:.2f}, Test: {test_acc:.2f}")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, model.predict(X_test)))

    # 8. Lưu Model, Scaler, PCA
    dump(model, "models/lung_cancer_svm.joblib")
    dump(scaler, "models/scaler.joblib")
    dump(pca, "models/pca.joblib")
    logging.info("Đã lưu Model, Scaler và PCA vào thư mục 'models/'.")

if __name__ == "__main__":
    main()