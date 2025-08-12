#https://github.com/SiliconJelly/OpenCV/blob/main/Distance%20Estimation/5.%20distance_check/check.py

import cv2 as cv
import os
import numpy as np

# Chess/checker board size, dimensions
CHESS_BOARD_DIM = (9, 6)

# The size of squares in the checker board design.
SQUARE_SIZE = 25  # millimeters (change it according to printed size)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# The images directory path
image_dir_path = "C:\\Users\\fhija\\Downloads\\agv\\kalibrasi_kamera_baru\\calibration1_images"

# Check if the directory exists, if not, create it
if not os.path.exists(image_dir_path):
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already exists.')

# Prepare object points, i.e. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the given images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane

# List of names of all the files present in the directory
files = os.listdir(image_dir_path)
grayScale = None  # Initialize grayScale variable
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)
        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows() 

# Calibrate the camera
if grayScale is not None:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None)
    print("calibrated")

    # Save the calibration data
    print("dumping the data into one file using numpy")
    np.savez(f"{image_dir_path}/MultiMatrix", camMatrix=mtx, distCoef=dist, rVector=rvecs, tVector=tvecs)

    print("-------------------------------------------")

    # Load the calibration data
    print("loading data stored using numpy savez function\n \n \n")
    data = np.load(f"{image_dir_path}/MultiMatrix.npz")
    camMatrix = data["camMatrix"]
    distCof = data["distCoef"]
    rVector = data["rVector"]
    tVector = data["tVector"]

    print("loaded calibration data successfully")
else:
    print("No images found for calibration.")