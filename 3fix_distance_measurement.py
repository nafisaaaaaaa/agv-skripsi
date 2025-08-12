import cv2 as cv
from cv2 import aruco
import numpy as np
import math

# Load calibration data

calib_data_path = r"C:\\Users\\Windows\\Downloads\\skripsi\\kalibrasi_kamera_baru\\calibration_images\\MultiMatrix.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Konstanta arena & marker

MARKER_SIZE    = 15    # cm
ARENA_WIDTH    = 240   # cm
ARENA_HEIGHT   = 240   # cm
ARENA_CORNERS_WORLD = np.array([
    [0, 0],
    [ARENA_WIDTH, 0],
    [ARENA_WIDTH, ARENA_HEIGHT],
    [0, ARENA_HEIGHT]
], dtype=np.float32)

# Setup ArUco

marker_dict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
param_markers = aruco.DetectorParameters()
param_markers.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
param_markers.cornerRefinementWinSize       = 5
param_markers.cornerRefinementMaxIterations = 30
param_markers.cornerRefinementMinAccuracy   = 0.1

# Inisialisasi kamera

cap = cv.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Gagal membuka kamera")
    cap.release()
    exit()

h, w = frame.shape[:2]
new_cam_mtx, roi = cv.getOptimalNewCameraMatrix(cam_mat, dist_coef, (w, h), 1, (w, h))
x_roi, y_roi, w_roi, h_roi = roi

scale_factor = 1.5  # display upscaling

# Variabel global

arena_corners_image   = None
transformation_matrix = None
grid_step_cm          = 40

# Helper functions

def calculate_distance(tVec):
    return float(np.linalg.norm(tVec))

def compute_transformation():
    global transformation_matrix
    pts = arena_corners_image.astype(np.float32)
    transformation_matrix = cv.getPerspectiveTransform(pts, ARENA_CORNERS_WORLD)

def camera_to_world_coords(pixel_pt):
    if transformation_matrix is None:
        return np.array([0.,0.], dtype=np.float32)
    pt = np.array([[pixel_pt]], dtype=np.float32)
    wp = cv.perspectiveTransform(pt, transformation_matrix)[0][0]
    return np.round(wp,1)

def refine_corner(img, corner, win_size=11):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c = np.array([corner], dtype=np.float32)
    r = cv.cornerSubPix(gray, c, (win_size,win_size), (-1,-1), crit)
    return r[0]

# Mouse callback untuk set 4 corner arena

def mouse_callback(evt, u, v, flags, param):
    global arena_corners_image
    if evt != cv.EVENT_LBUTTONDOWN:
        return
    u0 = int(u / scale_factor)
    v0 = int(v / scale_factor)
    if not (0 <= u0 < w_roi and 0 <= v0 < h_roi):
        return
    print(f"[Klik] Posisi pixel (ROI): ({u0}, {v0})")  # Print koordinat pixel yang diklik
    corner = np.array([u0, v0], dtype=np.float32)
    refined = refine_corner(undistorted_roi, corner)
    
    if arena_corners_image is None:
        arena_corners_image = np.array([refined])
    else:
        arena_corners_image = np.vstack([arena_corners_image, refined])
    if len(arena_corners_image) == 4:
        # sort TL,TR,BR,BL
        pts = arena_corners_image.copy()
        idx = np.lexsort((pts[:,0],pts[:,1]))
        s = pts[idx]
        top2, bot2 = s[:2], s[2:]
        tl, tr = top2[np.argsort(top2[:,0])]
        bl, br = bot2[np.argsort(bot2[:,0])]
        arena_corners_image[:] = np.array([tl, tr, br, bl], dtype=np.float32)
        compute_transformation()

cv.namedWindow("Camera View")
cv.namedWindow("Arena Tracking")
cv.setMouseCallback("Camera View", mouse_callback)
# print("Klik arena corners: TL,TR,BR,BL (4x)")

# Main loop

with open("data_posisi.txt","w") as logfile:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # undistort full frame
        und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
        undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        display_frame   = cv.resize(
            undistorted_roi, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv.INTER_LINEAR
        )

        # Tampilkan instruksi klik di jendela kamera
        # instruksi = "Klik 4 titik sudut arena (TL,TR,BR,BL)"
        # cv.putText(display_frame, instruksi,
        #            (10, 25), cv.FONT_HERSHEY_SIMPLEX,
        #            0.7, (0, 255, 255), 2, cv.LINE_AA)
        # if arena_corners_image is not None:
        #     txt_count = f"Corner ke-{len(arena_corners_image)}/4"
        #     cv.putText(display_frame, txt_count,
        #                (10, 55), cv.FONT_HERSHEY_SIMPLEX,
        #                0.6, (255, 200, 0), 1, cv.LINE_AA)

        # prepare top-down arena view
        dh,dw = display_frame.shape[:2]
        arena_view = 255*np.ones((dh,dw,3),dtype=np.uint8)
        a_scale = min(dw/ARENA_WIDTH, dh/ARENA_HEIGHT)
        ox = int((dw-ARENA_WIDTH*a_scale)/2)
        oy = int((dh-ARENA_HEIGHT*a_scale)/2)

        if transformation_matrix is not None:
            # Boundary & grid
            cv.rectangle(
                arena_view,
                (ox,oy),
                (ox+int(ARENA_WIDTH*a_scale), oy+int(ARENA_HEIGHT*a_scale)),
                (0,0,0),2
            )
            step = int(grid_step_cm*a_scale)
            for yy in range(0, int(ARENA_HEIGHT*a_scale)+1, step):
                cv.line(arena_view,(ox,oy+yy),(ox+int(ARENA_WIDTH*a_scale),oy+yy),(200,200,200),1)
            for xx in range(0, int(ARENA_WIDTH*a_scale)+1, step):
                cv.line(arena_view,(ox+xx,oy),(ox+xx,oy+int(ARENA_HEIGHT*a_scale)),(200,200,200),1)

        # detect markers on original frame
        gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_full, marker_dict, parameters=param_markers)

        if ids is not None:
            rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            for i, mid in enumerate(ids.flatten()):
                # undistort marker corners to ROI
                pts = corners[i].reshape(-1,1,2).astype(np.float32)
                pts_ud = cv.undistortPoints(pts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1,2)
                pts_ud -= np.array([x_roi,y_roi])
                refined = np.array([refine_corner(undistorted_roi, c, win_size=5) for c in pts_ud])
                center_px = refined.mean(axis=0)

                # world coords & 3D distance
                world_xy = camera_to_world_coords(center_px)
                dist3d = calculate_distance(tVecs[i][0])

                # gambar di display_frame
                poly = (pts_ud*scale_factor).astype(int)
                cv.polylines(display_frame, [poly], True, (0,255,255), 2, cv.LINE_AA)

                # gambar axis kecil
                axis_len = MARKER_SIZE*0.75
                axis3d = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0]]).reshape(-1,3)
                imgpts, _ = cv.projectPoints(axis3d, rVecs[i], tVecs[i], cam_mat, dist_coef)
                imgpts = cv.undistortPoints(imgpts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1,2)
                imgpts = ((imgpts - np.array([x_roi,y_roi]))*scale_factor).astype(int)
                org = tuple(imgpts[0])
                cv.line(display_frame, org, tuple(imgpts[1]), (0,0,255),2)
                cv.line(display_frame, org, tuple(imgpts[2]), (0,255,0),2)
                cv.putText(display_frame, "X", tuple(imgpts[1]), cv.FONT_HERSHEY_PLAIN, 1*scale_factor, (0,0,255),1)
                cv.putText(display_frame, "Y", tuple(imgpts[2]), cv.FONT_HERSHEY_PLAIN, 1*scale_factor, (0,255,0),1)

                # teks ID & x,y dengan background box
                tr = tuple(poly[1])
                font      = cv.FONT_HERSHEY_SIMPLEX
                scale_txt = 0.5*scale_factor
                thk       = 1
                txt1 = f"ID:{mid} D:{dist3d:.1f}cm"
                txt2 = f"x:{tVecs[i][0][0]:.1f} y:{tVecs[i][0][1]:.1f}"
                (w1,h1),_ = cv.getTextSize(txt1, font, scale_txt, thk)
                (w2,h2),_ = cv.getTextSize(txt2, font, scale_txt, thk)
                pad = 4
                bx = tr[0]
                by = tr[1] - (h1+h2+pad*3)
                cv.rectangle(display_frame,
                             (bx, by),
                             (bx + max(w1,w2)+pad*2, by + h1+h2+pad*2),
                             (0,0,0), cv.FILLED)
                cv.putText(display_frame, txt1,
                           (bx+pad, by+h1+pad//2),
                           font, scale_txt, (0,255,255), thk, cv.LINE_AA)
                cv.putText(display_frame, txt2,
                           (bx+pad, by+h1+h2+pad//2),
                           font, scale_txt, (0,255,255), thk, cv.LINE_AA)

                # simpan data
                logfile.write(f"ID:{mid}\n")
                logfile.write(f"tVec: {tVecs[i][0]}  Dist: {dist3d:.2f}cm  World: {world_xy}\n")
                logfile.write("---\n")

                # gambar di arena_view
                if transformation_matrix is not None:
                    ax = int(world_xy[0]*a_scale)+ox
                    ay = int(world_xy[1]*a_scale)+oy
                    cv.circle(arena_view, (ax,ay), 6, (255,0,0), -1)
                    # ID
                    cv.putText(arena_view, f"id: {mid}", (ax+8, ay-8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                    # world coords
                    txt_w = f"world: ({world_xy[0]:.1f},{world_xy[1]:.1f})cm"
                    cv.putText(arena_view, txt_w, (ax+8, ay+12),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                    
       # Rapikan header teks di Arena Tracking
        fps = int(cap.get(cv.CAP_PROP_FPS))
        pad = 10; font = cv.FONT_HERSHEY_SIMPLEX; fs=0.6; th=1; lh=int(20*fs)
        overlay = arena_view.copy()
        box_h = lh*2 + pad*2; box_w = 200
        cv.rectangle(overlay, (pad,pad), (pad+box_w,pad+box_h), (255,255,255), -1)
        arena_view = cv.addWeighted(overlay, 0.6, arena_view, 0.4, 0)
        cv.putText(arena_view, f"Arena: {ARENA_WIDTH}x{ARENA_HEIGHT} cm",
                   (pad*2, pad+lh), font, fs, (0,0,0), th, cv.LINE_AA)
        txt_fps = f"FPS: {fps}"
        (w_txt,_),_ = cv.getTextSize(txt_fps, font, fs, th)
        x_txt = dw - w_txt - pad
        cv.rectangle(arena_view,
                     (x_txt-pad//2, pad),
                     (x_txt+w_txt+pad//2, pad+box_h//2),
                     (255,255,255), -1)
        cv.putText(arena_view, txt_fps,
                   (x_txt, pad+lh), font, fs, (0,0,0), th, cv.LINE_AA)

        # tampilkan
        cv.imshow("Camera View", display_frame)
        cv.imshow("Arena Tracking", arena_view)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            arena_corners_image   = None
            transformation_matrix = None
            print("Arena corners reset!")

cap.release()
cv.destroyAllWindows()