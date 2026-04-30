import cv2 as cv

def apply_median_filter(image, kernel_size=3):
    """
    Görüntüdeki tuz-biber gürültüsünü temizlemek için Median filtresi uygular.
    kernel_size: 3, 5, 7 gibi tek sayı olmalıdır.
    """
    # OpenCV'de medianBlur fonksiyonu doğrudan kernel boyutunu tam sayı olarak alır
    return cv.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Kenarları koruyarak gürültüyü azaltan gelişmiş bir filtredir.
    Raporuna ekstra bilgi olarak ekleyebilirsin.
    """
    return cv.bilateralFilter(image, d, sigmaColor, sigmaSpace)