"""
Utility functions for image preprocessing and feature extraction
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
import config


def preprocess_img(img, img_size=None):
    """
    Tiền xử lý ảnh CT Scan một cách chuẩn hóa.
    
    Các bước xử lý:
    1. Padding ảnh để giữ aspect ratio (thay vì resize trực tiếp)
    2. Cân bằng độ sáng tự động (Auto-Gain Control)
    3. Cân bằng độ tương phản cục bộ (CLAHE)
    4. Lọc nhiễu Gaussian
    5. Masking - Loại bỏ nền
    
    Args:
        img: Ảnh grayscale
        img_size: Kích thước đích (mặc định từ config)
    
    Returns:
        Ảnh được xử lý, chuẩn hóa về [0, 1]
    """
    if img_size is None:
        img_size = config.IMG_SIZE
    
    # 1. Padding ảnh để giữ aspect ratio (không bị méo như resize)
    h, w = img.shape
    max_dim = max(h, w)
    
    # Tạo canvas vuông
    canvas = np.zeros((max_dim, max_dim), dtype=img.dtype)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    canvas[pad_h:pad_h+h, pad_w:pad_w+w] = img
    
    # Resize canvas vuông thành target size (không bị méo)
    img = cv2.resize(canvas, img_size)
    
    # 2. Xử lý độ sáng tự động (Auto-Gain Control)
    mean_brightness = np.mean(img)
    alpha = config.TARGET_BRIGHTNESS / mean_brightness
    alpha = max(config.ALPHA_MIN, min(alpha, config.ALPHA_MAX))
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, 0)
    
    # 3. Cân bằng độ tương phản cục bộ (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_SIZE
    )
    img = clahe.apply(img)
    
    # 4. Lọc nhiễu
    img = cv2.GaussianBlur(img, config.GAUSSIAN_BLUR_SIZE, 0)
    
    # 5. Masking - Loại bỏ nền
    _, mask = cv2.threshold(img, config.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    return img / 255.0  # Chuẩn hóa về [0, 1]


def extract_hog_features(img):
    """Trích xuất HOG features"""
    return hog(
        img,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        channel_axis=None
    )


def extract_lbp_features(img):
    """Trích xuất LBP features"""
    lbp = local_binary_pattern(
        img,
        config.LBP_N_POINTS,
        config.LBP_RADIUS,
        method=config.LBP_METHOD
    )
    # P=24 với method='uniform' tạo 26 bins
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_features(images, verbose=True):
    """
    Trích xuất kết hợp HOG + LBP từ danh sách ảnh.
    
    Args:
        images: Mảng numpy của ảnh đã xử lý
        verbose: Hiển thị progress bar
    
    Returns:
        Mảng numpy chứa tất cả features
    """
    features = []
    iterator = tqdm(images, desc="Trích xuất đặc trưng HOG & LBP") if verbose else images
    
    for img in iterator:
        hog_feat = extract_hog_features(img)
        lbp_feat = extract_lbp_features(img)
        features.append(np.hstack([hog_feat, lbp_feat]))
    
    return np.array(features)
