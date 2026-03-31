# Phát Hiện Ung Thư Phổi Từ Ảnh CT Scan

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![Libraries](https://img.shields.io/badge/libraries-OpenCV%2CSklearn%2CPIL-yellow.svg)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Đề tài này xây dựng một hệ thống phát hiện ung thư phổi từ ảnh chụp cắt lớp vi tính (CT scan) sử dụng các kỹ thuật xử lý ảnh và học máy. Hệ thống bao gồm một script để huấn luyện mô hình phân loại và một ứng dụng giao diện người dùng đồ họa (GUI) đơn giản để thực hiện dự đoán trên ảnh mới.

## Cấu trúc dự án

```
.
├── BCT_projeck_1/             # (Nếu có) Thư mục chứa các phần liên quan đến dự án BCT (không rõ mục đích cụ thể)
├── dataset/
│   ├── cancer/                # Chứa ảnh CT scan của bệnh nhân ung thư
│   └── normal/                # Chứa ảnh CT scan của bệnh nhân không ung thư
├── models/                    # Chứa các model đã huấn luyện (SVM, StandardScaler, PCA)
├── test/                      # (Nếu có) Thư mục chứa các ảnh test riêng biệt
├── huong an.txt               # File hướng dẫn (nếu có nội dung)
├── inference_script.py        # Script ứng dụng GUI để dự đoán trên ảnh mới
├── README.md                  # File hướng dẫn và mô tả dự án này
└── training_script.py         # Script để huấn luyện mô hình phát hiện ung thư phổi
```

## Các Thư Viện Sử Dụng

Dự án này sử dụng các thư viện Python sau:

- `opencv-python` (cv2)
- `scikit-learn` (sklearn)
- `scikit-image`
- `joblib`
- `tqdm`
- `PIL` (Pillow)
- `tkinter`
- `numpy`

## Cách Hoạt Động Của Code

### 1. Huấn luyện mô hình (`training_script.py`)

Script huấn luyện này thực hiện các bước sau:

1.  **Tiền xử lý và Tải dữ liệu:** Tải ảnh từ thư mục `dataset` (chia thành `normal` và `cancer`). Mỗi ảnh được tiền xử lý qua hàm `preprocess_img` bao gồm:
    - Resize về kích thước đồng nhất (512x512).
    - Xử lý ảnh cháy sáng cục bộ.
    - Cân bằng độ tương phản thích ứng cục bộ (CLAHE) để làm nổi bật các đặc điểm nhỏ như khối u.
    - Lọc nhiễu nhẹ bằng Gaussian Blur.
    - Masking để loại bỏ các vùng nhiễu không phải phổi.
2.  **Trích xuất Đặc trưng:** Sử dụng kết hợp Histogram of Oriented Gradients (HOG) và Local Binary Patterns (LBP) để tạo ra vector đặc trưng cho mỗi ảnh.
3.  **Chuẩn bị Dữ liệu:** Dữ liệu được chia thành tập huấn luyện và kiểm thử (80/20), sau đó chuẩn hóa bằng `StandardScaler` và giảm chiều bằng Principal Component Analysis (PCA) xuống 100 thành phần.
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
pip install opencv-python scikit-learn scikit-image joblib tqdm matplotlib imagehash Pillow numpy
```

### 2. Chuẩn bị dữ liệu

- Tạo thư mục `dataset` trong thư mục gốc của dự án.
- Bên trong `dataset`, tạo hai thư mục con: `normal` và `cancer`.
- Đặt tất cả ảnh CT scan của các trường hợp bình thường vào thư mục `dataset/normal/`.
- Đặt tất cả ảnh CT scan của các trường hợp ung thư vào thư mục `dataset/cancer/`.

### 3. Huấn luyện mô hình

Chạy script huấn luyện để tạo ra mô hình dự đoán. Điều này sẽ lưu các file mô hình cần thiết vào thư mục `models/`.

```bash
python training_script.py
```

### 4. Chạy ứng dụng giao diện

Sau khi mô hình đã được huấn luyện và lưu, bạn có thể chạy ứng dụng GUI để dự đoán trên các ảnh mới:

```bash
python inference_script.py
```

Ứng dụng sẽ mở ra một cửa sổ, cho phép bạn chọn ảnh CT scan và nhận kết quả dự đoán.
