# ============ CONFIGURATION FILE ============
# Tập trung quản lý tất cả hyperparameter và cấu hình

# ----------- Preprocessing -----------
IMG_SIZE = (512, 512)
TARGET_BRIGHTNESS = 105.0
ALPHA_MIN = 0.6  # Cho phép giảm sáng nhiều hơn cho ảnh quá trắng (trước: 0.85)
ALPHA_MAX = 1.5  # Cho phép tăng sáng cho ảnh quá tối (trước: 1.3)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
GAUSSIAN_BLUR_SIZE = (3, 3)
MASK_THRESHOLD = 30

# ----------- Feature Extraction -----------
HOG_ORIENTATIONS = 16
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)

LBP_RADIUS = 3
LBP_N_POINTS = 24
LBP_METHOD = "uniform"

# ----------- Dataset -----------
DATA_DIR = "dataset"
TRAIN_TEST_SPLIT = 0.2
VAL_TEST_SPLIT = 0.2  # 20% của train dùng cho validation
RANDOM_STATE = 42
STD_THRESHOLD = 5  # Bỏ qua ảnh có std < threshold

# ----------- PCA & Scaling -----------
PCA_N_COMPONENTS = 100

# ----------- SVM Hyperparameters -----------
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_CLASS_WEIGHT = 'balanced'
SVM_PROBABILITY = True

# ----------- Model Inference -----------
MODEL_CONFIDENCE_THRESHOLD = 0.65  # Chỉ dự đoán nếu độ tin cậy > này
MODEL_DIR = "models"
MODEL_NAME = "lung_cancer_svm.joblib"
SCALER_NAME = "scaler.joblib"
PCA_NAME = "pca.joblib"

# ----------- UI Parameters -----------
UI_IMAGE_SIZE = (350, 350)
UI_FONT = ('Helvetica', 16, 'bold')
