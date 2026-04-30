import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from filters.smoothing import apply_gaussian_blur
from filters.edge_detection import apply_canny
from filters.noise_reduction import apply_median_filter
from filters.segmentation import apply_otsu_threshold

# 1. Görüntüleri Yükle
img = cv.imread('images/woman.png')
if img is None:
    print("Hata: Görüntü yüklenemedi!")
    exit()

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# --- UYGULAMA 1: İSTATİSTİKSEL ANALİZ ÇIKTILARI ---
min_val = np.min(gray)
max_val = np.max(gray)
mean_val = np.mean(gray)
median_val = np.median(gray)

print("\n" + "="*30)
print(" UYGULAMA 1: SAYISAL ANALİZ")
print("="*30)
print(f"Minimum Piksel Değeri: {min_val}")
print(f"Maksimum Piksel Değeri: {max_val}")
print(f"Ortalama (Mean) Değer: {mean_val:.2f}")
print(f"Medyan (Median) Değer: {median_val}")
print("="*30 + "\n")

# --- UYGULAMA 3: OTSU EŞİKLEME ÇIKTISI ---
otsu_val, otsu_img = apply_otsu_threshold(gray)

print("="*30)
print(" UYGULAMA 3: OTSU ANALİZİ")
print("="*30)
print(f"Otomatik Belirlenen Eşik Değeri: {otsu_val}")
print("="*30 + "\n")

# --- DİĞER İŞLEMLER ---
img_negative = 255 - img_rgb
gaussian_blur = apply_gaussian_blur(img_rgb, (5, 5))
median_cleaned = apply_median_filter(gray, 5)
canny_edges = apply_canny(median_cleaned, 50, 150)
R, G, B = cv.split(img_rgb)

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1); plt.imshow(img_rgb); plt.title('Orijinal RGB')
plt.subplot(2, 3, 2); plt.imshow(img_negative); plt.title('Negatif')
plt.subplot(2, 3, 3); plt.hist(gray.ravel(), 256, [0, 256]); plt.title('Histogram')
plt.subplot(2, 3, 4); plt.imshow(canny_edges, cmap='gray'); plt.title('Canny Kenarlar')
plt.subplot(2, 3, 5); plt.imshow(otsu_img, cmap='gray'); plt.title(f'Otsu (Eşik: {otsu_val})')
plt.subplot(2, 3, 6); plt.imshow(G, cmap='gray'); plt.title('Yeşil Kanal (Griye En Yakın)')

plt.tight_layout()
print("Grafikler oluşturuldu. Lütfen pencereyi kontrol et.")
plt.show()
