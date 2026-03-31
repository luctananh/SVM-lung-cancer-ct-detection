# Phát Hiện Ung Thư Phổi Từ Ảnh CT Scan

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![Libraries](https://img.shields.io/badge/libraries-OpenCV%2CSklearn%2CPIL-yellow.svg)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Đề tài này xây dựng một hệ thống phát hiện ung thư phổi từ ảnh chụp cắt lớp vi tính (CT scan) sử dụng các kỹ thuật xử lý ảnh và học máy. Hệ thống bao gồm một script để huấn luyện mô hình phân loại và một ứng dụng giao diện người dùng đồ họa (GUI) đơn giản để thực hiện dự đoán trên ảnh mới.

## Cấu Trúc Dự Án

```
├── config.py                    # Cấu hình hyperparameter & tham số
├── utils.py                     # Các hàm tiện ích (preprocessing, feature extraction)
├── training_script.py           # Script huấn luyện mô hình
├── inference_script.py          # GUI dự đoán
├── models/                      # Thư mục chứa các file mô hình
│   ├── lung_cancer_svm.joblib
│   ├── scaler.joblib
│   └── pca.joblib
├── dataset/                     # Dữ liệu huấn luyện
│   ├── normal/
│   └── cancer/
├── test/                        # Dữ liệu test
├── requirements.txt             # Danh sách thư viện
└── README.md
```

## Các Thư Viện Sử Dụng

Dự án này sử dụng các thư viện Python sau:

- `opencv-python` (cv2)
- `scikit-learn` (sklearn)
- `scikit-image`
- `joblib`
- `tqdm`
- `PIL` (Pillow)
- `numpy`
- `matplotlib`
- `imagehash`
- `tkinter`

## Cách Hoạt Động Của Code

### 1. Huấn luyện mô hình (`training_script.py`)

Script huấn luyện này thực hiện các bước sau:

1.  **Tiền xử lý và Tải dữ liệu:** Tải ảnh từ thư mục `dataset` (chia thành `normal` và `cancer`). Mỗi ảnh được tiền xử lý qua hàm `preprocess_img()` trong `utils.py` bao gồm:
    - **Padding ảnh** để giữ nguyên aspect ratio (thay vì resize trực tiếp để tránh biến dạng hình ảnh). Ảnh được padding thành hình vuông rồi mới resize về 512×512.
    - **Cân bằng độ sáng tự động** (Auto-Gain Control) để xử lý ảnh quá tối hoặc quá sáng.
    - **Cân bằng độ tương phản thích ứng cục bộ (CLAHE)** để làm nổi bật các đặc điểm nhỏ như khối u.
    - **Lọc nhiễu** bằng Gaussian Blur.
    - **Masking** để loại bỏ các vùng nhiễu không phải phổi.
2.  **Trích xuất Đặc trưng:** Sử dụng kết hợp Histogram of Oriented Gradients (HOG) và Local Binary Patterns (LBP) để tạo ra vector đặc trưng cho mỗi ảnh.
3.  **Chuẩn bị Dữ liệu:** Dữ liệu được chia thành 3 tập:
    - **Tập huấn luyện (64%):** Dùng để training mô hình SVM.
    - **Tập xác thực (16%):** Dùng để kiểm tra overfitting trong quá trình training.
    - **Tập kiểm thử (20%):** Dùng để đánh giá hiệu suất cuối cùng của mô hình.

    Sau đó, tất cả dữ liệu được chuẩn hóa bằng `StandardScaler` và giảm chiều bằng Principal Component Analysis (PCA) xuống 100 thành phần để tăng tốc độ training.

4.  **Huấn luyện Mô hình:** Một mô hình Support Vector Machine (SVM) với kernel RBF, `C=1.0`, `gamma='scale'` và `class_weight='balanced'` được huấn luyện trên tập dữ liệu đã xử lý.
5.  **Đánh giá Mô hình:** Đánh giá hiệu suất của mô hình trên cả tập huấn luyện và kiểm thử, bao gồm độ chính xác và báo cáo phân loại chi tiết.
6.  **Lưu Mô hình:** Mô hình SVM đã huấn luyện, `StandardScaler` và `PCA` được lưu vào thư mục `models/` dưới dạng file `.joblib` để sử dụng cho việc dự đoán.
7.  **Ghi Log:** Thông tin và các lỗi quan trọng trong quá trình huấn luyện được ghi vào `app.log`.

### 2. Giao diện người dùng (`inference_script.py`)

Script này cung cấp một ứng dụng GUI đơn giản, cho phép người dùng:

1.  **Tải ảnh:** Chọn một file ảnh CT scan (JPG, JPEG, PNG) từ máy tính.
2.  **Hiển thị ảnh:** Hiển thị ảnh gốc và ảnh đã qua tiền xử lý (`preprocess_img`) trực tiếp trên giao diện để người dùng dễ dàng quan sát quá trình xử lý.
3.  **Dự đoán:** Sử dụng các mô hình đã lưu (`lung_cancer_svm.joblib`, `scaler.joblib`, `pca.joblib`) để dự đoán xem ảnh có dấu hiệu ung thư phổi hay không. Quá trình trích xuất đặc trưng cho ảnh mới được đồng bộ hoàn toàn với quá trình huấn luyện.
4.  **Hiển thị kết quả:** Hiển thị nhãn dự đoán ("BÌNH THƯỜNG" hoặc "CẢNH BÁO: PHÁT HIỆN UNG THƯ") cùng với độ tin cậy của dự đoán, với màu sắc cảnh báo tương ứng (xanh lá cho bình thường, đỏ cho ung thư).

## Hướng dẫn cài đặt và chạy

Để cài đặt và chạy dự án, làm theo các bước sau:

### 1. Cài đặt các thư viện cần thiết

Mở terminal hoặc Command Prompt và chạy lệnh sau để cài đặt tất cả các thư viện Python yêu cầu:

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

- Tạo thư mục `dataset` trong thư mục gốc của dự án.
- Bên trong `dataset`, tạo hai thư mục con: `normal` và `cancer`.
- Đặt tất cả ảnh CT scan của các trường hợp bình thường vào thư mục `dataset/normal/`.
- Đặt tất cả ảnh CT scan của các trường hợp ung thư vào thư mục `dataset/cancer/`.

### 3. Tùy chỉnh Cấu hình

Các hyperparameter và tham số cấu hình có thể được điều chỉnh trong file `config.py`:

- **IMG_SIZE:** Kích thước ảnh sau khi xử lý (mặc định: 512×512)
- **TARGET_BRIGHTNESS:** Độ sáng mục tiêu sau cân bằng (mặc định: 105.0)
- **ALPHA_MIN / ALPHA_MAX:** Giới hạn hệ số điều chỉnh độ sáng
- **CLAHE_CLIP_LIMIT:** Tham số cảnh báo cho CLAHE (mặc định: 3.0)
- **PCA_N_COMPONENTS:** Số thành phần PCA (mặc định: 100)
- **SVM_C / SVM_GAMMA / SVM_KERNEL:** Tham số SVM
- **MODEL_CONFIDENCE_THRESHOLD:** Ngưỡng độ tin cậy để dự đoán (mặc định: 0.65)

### 4. Huấn luyện mô hình

Chạy script huấn luyện để tạo ra mô hình dự đoán. Điều này sẽ lưu các file mô hình cần thiết vào thư mục `models/`:

```bash
python training_script.py
```

### 5. Chạy ứng dụng giao diện

Sau khi mô hình đã được huấn luyện và lưu, bạn có thể chạy ứng dụng GUI để dự đoán trên các ảnh mới:

```bash
python inference_script.py
```

Ứng dụng sẽ mở ra một cửa sổ, cho phép bạn chọn ảnh CT scan và nhận kết quả dự đoán.

## Demo

**Hình 1: Test trên tập dữ liệu mới - Mẫu 1**
![image](https://github.com/user-attachments/assets/54820537-7815-4ce0-9b33-c781c3af28a7)

**Hình 2: Test trên dữ liệu mới - Mẫu 2**
![image](https://github.com/user-attachments/assets/19818075-4c5d-462c-b3ac-137e4eea74e2)

## Lưu ý quan trọng

- **Padding ảnh:** Hệ thống sử dụng **padding** thay vì resize trực tiếp để giữ nguyên aspect ratio của ảnh. Điều này giúp tránh biến dạng hình ảnh và cải thiện độ chính xác dự đoán, đặc biệt với ảnh có kích thước không vuông.

- **Confidence Threshold:** Khi dự đoán, nếu mô hình không đủ tự tin (confidence < 65%), hệ thống sẽ hiển thị cảnh báo "Không rõ ràng" cùng với khuynh hướng dự đoán. Điều này giúp tránh sai lầm do chất lượng ảnh kém.

- **Validation Set:** Dữ liệu được chia thành 3 tập (train/validation/test) để phát hiện overfitting. Nếu accuracy trên validation set thấp hơn significantly so với training set, cần điều chỉnh hyperparameter hoặc lấy thêm dữ liệu huấn luyện.

- **Dữ liệu test ngoài:** Khi test trên dữ liệu mới từ các nguồn khác, chất lượng ảnh, độ sáng, và định dạng có thể khác biệt. Nếu kết quả không như mong đợi, sử dụng `debug_test.py` để phân tích chi tiết.
