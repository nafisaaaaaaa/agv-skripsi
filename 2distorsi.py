import cv2 as cv
import numpy as np

# Load data kalibrasi
data = np.load("C:\\Users\\Windows\\Downloads\\skripsi\\kalibrasi_kamera_baru\\calibration_images\\MultiMatrix.npzz")
camMatrix = data["camMatrix"]
distCof = data["distCoef"]

# Buka webcam
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Tidak bisa membuka kamera.")
    exit()

# Ambil resolusi dari frame
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Hitung new camera matrix
newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCof, (frame_width, frame_height), 1, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Tidak bisa membaca frame.")
        break

    # Undistort frame
    undistorted = cv.undistort(frame, camMatrix, distCof, None, newCamMatrix)

    # (opsional) crop sesuai ROI
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Resize ke ukuran original
    undistorted_resized = cv.resize(undistorted, (frame_width, frame_height))

    # Tampilkan
    cv.imshow("Original", frame)
    cv.imshow("Undistorted", undistorted_resized)

    if cv.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv.destroyAllWindows()
