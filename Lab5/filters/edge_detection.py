import cv2 as cv
import numpy as np

def apply_laplacian(image):
    """
    Görüntüdeki kenarları Laplacian operatörü ile tespit eder.
    Görüntünün ikinci türevini alarak ani değişimleri bulur.
    """
    # Görüntü gri seviyeye dönüştürülmüş olmalıdır.
    # CV_64F kullanıyoruz çünkü türev işlemi negatif değerler üretebilir.
    laplacian = cv.Laplacian(image, cv.CV_64F)
    
    # Negatif değerleri görselleştirebilmek için mutlak değere çevirip 8-bit'e dönüştürüyoruz.
    laplacian_8bit = np.uint8(np.absolute(laplacian))
    return laplacian_8bit

def apply_sobel(image, axis='x'):
    """
    Sobel operatörü ile belirli bir yöndeki (x veya y) kenarları bulur.
    """
    if axis == 'x':
        sobel = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    else:
        sobel = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
        
    return np.uint8(np.absolute(sobel))


def apply_canny(image, low_threshold=50, high_threshold=150):
    """
    Canny algoritması ile daha temiz ve belirgin kenarlar bulur.
    """
    # Canny fonksiyonu gürültü azaltma ve kenar takibini kendi içinde yapar.
    return cv.Canny(image, low_threshold, high_threshold)