# ====================================================================
# FASTAPI API DETEKSI BUAH (DENGAN API KEY + UPLOAD FILE + ENV)
# ====================================================================

import cv2
import numpy as np
import joblib
import pywt
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, hog
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image

# ===============================================================
# LOAD .ENV
# ===============================================================
load_dotenv()

APP_KEY_NAME = os.getenv("APP_KEY_NAME", "X-API-Key")
APP_KEY_VALUE = os.getenv("APP_KEY_VALUE", "DEFAULT_KEY")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# ===============================================================
# LOAD MODEL
# ===============================================================
base_path = "model"
class_names = ['apel_malang', 'jambu_guava', 'jeruk_bali', 'mangga', 'melon']

print("🔄 Loading Model...")
svm_model = joblib.load(os.path.join(base_path, "model_svm_ultimate.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler_ultimate.pkl"))
print(" Model & Scaler Loaded!")

# ===============================================================
# FASTAPI INIT
# ===============================================================
app = FastAPI(
    title="API Deteksi Buah Ultimate",
    description="Upload gambar untuk mendeteksi jenis buah",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================================================
# PREPROCESSING & FEATURE EXTRACTION
# ===============================================================

def resize_with_padding(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def preprocess_image(image):
    image = apply_clahe(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 50, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            x, y, w, h = cv2.boundingRect(largest)
            roi = image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            roi = cv2.bitwise_and(roi, roi, mask=mask_roi)
            return resize_with_padding(roi)

    return resize_with_padding(image)


def extract_features(image):
    roi = preprocess_image(image)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)

    color_feats = []
    for channel in [h, s]:
        color_feats.extend([
            np.mean(channel), np.std(channel),
            skew(channel.flatten()), kurtosis(channel.flatten())
        ])

    _, _, v = cv2.split(hsv)
    coeffs = pywt.wavedec2(v, 'haar', level=3)
    LL1 = pywt.wavedec2(v, 'haar', level=1)[0]
    LL1_norm = cv2.normalize(LL1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    glcm = graycomatrix(LL1_norm, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, "contrast").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()

    shades = []
    prominences = []
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            m = glcm[:, :, i, j]
            N = m.shape[0]
            r, c = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            mean_i = np.sum(r * m)
            mean_j = np.sum(c * m)
            term = (r + c - mean_i - mean_j)
            shades.append(np.sum((term ** 3) * m))
            prominences.append(np.sum((term ** 4) * m))

    wavelet_std = [np.std(coeffs[0])]
    for level in coeffs[1:]:
        wavelet_std.extend([np.std(c) for c in level])

    img_resized = cv2.resize(roi, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm="L2-Hys")

    all_features = np.hstack([
        color_feats,
        [contrast, energy, homogeneity, np.mean(shades), np.mean(prominences)],
        wavelet_std,
        hog_feats
    ])

    return np.nan_to_num(all_features).reshape(1, -1)


# ===============================================================
# ENDPOINT PREDIKSI
# ===============================================================
@app.post("/predict")
async def predict_fruit(
    image_buah: UploadFile = File(...),
    api_key: str = Header(None, alias=APP_KEY_NAME)
):
    # cek API key
    if api_key != APP_KEY_VALUE:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # cek tipe file
    if image_buah.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File harus JPG atau PNG")

    contents = await image_buah.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Gambar tidak valid")

    feats = extract_features(img)
    scaled = scaler.transform(feats)

    pred_idx = svm_model.predict(scaled)[0]
    predicted = class_names[pred_idx]

    return {"success": True, "filename": image_buah.filename, "predicted_fruit": predicted}


@app.get("/")
def home():
    return {"message": "API Deteksi Buah berjalan!"}


# ===============================================================
# AUTO RUN SERVER SESUAI PORT & API KEY DI .ENV
# ===============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n==========================================")
    print(" 🚀 API DETEKSI BUAH ULIMATE - RUNNING")
    print("==========================================")
    print(f"🔑 API Key Name : {APP_KEY_NAME}")
    print(f"🔑 API Key Value: {APP_KEY_VALUE}")
    print(f"🌐 Running on Port: {APP_PORT}")
    print("==========================================\n")

    uvicorn.run("app_api:app", host="0.0.0.0", port=APP_PORT, reload=True)
