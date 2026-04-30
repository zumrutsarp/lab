import cv2 as cv

def apply_otsu_threshold(image):
    """Otsu'nun eşikleme metodunu uygular."""
    # Giriş görüntüsü gri seviyede olmalıdır.
    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return ret, thresh