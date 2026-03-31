# ----------- [1] Cài đặt & Import thư viện -----------
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

import config
import utils

# ----------- [2] Thiết lập Logging -----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset(data_dir, img_size=None):
    """Tải và tiền xử lý dataset sử dụng hàm preprocess_img chuẩn."""
    if img_size is None:
        img_size = config.IMG_SIZE
    
    images, labels = [], []
    
    for class_name in ["normal", "cancer"]:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Thiếu thư mục: {class_dir}")
            
        for img_name in tqdm(os.listdir(class_dir), desc=f"Đang tải {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Kiểm tra ảnh lỗi
            if img is None or img.std() < config.STD_THRESHOLD: 
                continue 

            # Gọi hàm preprocess_img từ utils
            img_final = utils.preprocess_img(img, img_size)
            
            images.append(img_final)
            labels.append(0 if class_name == "normal" else 1)
            
    return np.array(images), np.array(labels)

# ----------- [4] Hàm chính (Training Pipeline) -----------
def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Tải dữ liệu
    logging.info("Bắt đầu tải dữ liệu...")
    images, labels = load_dataset(config.DATA_DIR)
    logging.info(f"Tổng ảnh: {len(images)} (Normal: {np.sum(labels==0)}, Cancer: {np.sum(labels==1)})")
    
    # 2. Trích xuất đặc trưng
    logging.info("Trích xuất đặc trưng...")
    X = utils.extract_features(images)
    y = labels

    # 3. Chia tập Train/Test
    logging.info("Chia tập Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TRAIN_TEST_SPLIT, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # 3b. Chia Train thành Train/Validation
    logging.info("Chia tập Train/Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_TEST_SPLIT, 
        random_state=config.RANDOM_STATE, stratify=y_train
    )
    
    logging.info(f"  Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # 4. Chuẩn hóa dữ liệu (Standardization)
    logging.info("Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 5. Giảm chiều dữ liệu (PCA)
    logging.info("Giảm chiều dữ liệu với PCA...")
    pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.RANDOM_STATE)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    logging.info(f"  Giải thích được {pca.explained_variance_ratio_.sum():.2%} variance")

    # 6. Huấn luyện mô hình SVM
    logging.info("Bắt đầu huấn luyện mô hình SVM...")
    model = SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        gamma=config.SVM_GAMMA,
        class_weight=config.SVM_CLASS_WEIGHT,
        probability=config.SVM_PROBABILITY,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # 7. Đánh giá mô hình
    logging.info("Đánh giá mô hình...")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ TRAINING")
    print(f"{'='*60}")
    print(f"Độ chính xác - Train: {train_acc:.4f}, Validation: {val_acc:.4f}, Test: {test_acc:.4f}")
    
    print(f"\nBáo cáo phân loại (Test Set):")
    print(classification_report(y_test, model.predict(X_test), target_names=['Normal', 'Cancer']))
    
    print(f"\nMa trận nhầm lẫn (Test Set):")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print(f"{'='*60}\n")

    # 8. Lưu Model, Scaler, PCA
    logging.info("Lưu mô hình...")
    dump(model, f"{config.MODEL_DIR}/{config.MODEL_NAME}")
    dump(scaler, f"{config.MODEL_DIR}/{config.SCALER_NAME}")
    dump(pca, f"{config.MODEL_DIR}/{config.PCA_NAME}")
    logging.info(f"Đã lưu Model, Scaler và PCA vào thư mục '{config.MODEL_DIR}/'.")

if __name__ == "__main__":
    main()