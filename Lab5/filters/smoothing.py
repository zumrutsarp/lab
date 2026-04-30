import cv2 as cv

def apply_gaussian_blur(image, kernel_size=(3, 3)):
    """Görüntüye Gaussian Blur uygular."""
    return cv.GaussianBlur(image, kernel_size, 0)

def apply_mean_blur(image, kernel_size=(3, 3)):
    """Görüntüye Mean (Ortalama) Blur uygular."""
    return cv.blur(image, kernel_size)