import cv2 as cv
from cv2 import aruco
import numpy as np
import math
import time

# SESUAIKAN PATH INI DENGAN LOKASI FILE KALIBRASI ANDA
calib_data_path = r"C:\\Users\\Windows\\Downloads\\skripsi\\kalibrasi_kamera_baru\\calibration_images\\MultiMatrix.npz"

# Load calibration data
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Konstanta arena & marker
MARKER_SIZE = 15    # cm
ARENA_WIDTH = 240   # cm
ARENA_HEIGHT = 240  # cm
ARENA_CORNERS_WORLD = np.array([
    [0, 0],
    [ARENA_WIDTH, 0],
    [ARENA_WIDTH, ARENA_HEIGHT],
    [0, ARENA_HEIGHT]
], dtype=np.float32)

# Setup ArUco
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
param_markers = aruco.DetectorParameters()
param_markers.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
param_markers.cornerRefinementWinSize = 5
param_markers.cornerRefinementMaxIterations = 30
param_markers.cornerRefinementMinAccuracy = 0.1

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
arena_corners_image = None
transformation_matrix = None
grid_step_cm = 40

# Variabel untuk corner selection
corner_selection_mode = False
selected_corners = []

# Helper functions
def calculate_distance(tVec):
    return float(np.linalg.norm(tVec))

def compute_transformation():
    global transformation_matrix, arena_corners_image
    if arena_corners_image is not None and len(arena_corners_image) == 4:
        pts = arena_corners_image.astype(np.float32)
        transformation_matrix = cv.getPerspectiveTransform(pts, ARENA_CORNERS_WORLD)
        print("Transformation matrix computed successfully!")
    else:
        print("Warning: Arena corners not fully set, transformation matrix remains None")

def camera_to_world_coords(pixel_pt):
    global transformation_matrix
    if transformation_matrix is None:
        print("Warning: Transformation matrix not initialized, returning (0, 0)")
        return np.array([0., 0.], dtype=np.float32)
    pt = np.array([[pixel_pt]], dtype=np.float32)
    wp = cv.perspectiveTransform(pt, transformation_matrix)[0][0]
    return np.round(wp, 1)

def refine_corner(img, corner, win_size=11):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c = np.array([corner], dtype=np.float32)
    r = cv.cornerSubPix(gray, c, (win_size, win_size), (-1, -1), crit)
    return r[0]

def set_arena_corners_manual(corners):
    global arena_corners_image
    arena_corners_image = np.array(corners, dtype=np.float32)
    compute_transformation()
    print("Arena corners set manually")

def mouse_callback_simple(evt, u, v, flags인가, param):
    global arena_corners_image, selected_corners, corner_selection_mode
    
    if evt != cv.EVENT_LBUTTONDOWN or not corner_selection_mode:
        return
    
    u0 = int(u / scale_factor)
    v0 = int(v / scale_factor)
    
    if not (0 <= u0 < w_roi and 0 <= v0 < h_roi):
        return
    
    print(f"[Klik] Posisi pixel (ROI): ({u0}, {v0})")
    
    corner = np.array([u0, v0], dtype=np.float32)
    
    ret, frame = cap.read()
    if ret:
        und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
        undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        refined = refine_corner(undistorted_roi, corner)
        selected_corners.append(refined)
        
        print(f"[Refined] Corner {len(selected_corners)}: ({refined[0]:.1f}, {refined[1]:.1f})")
        
        if len(selected_corners) == 4:
            pts = np.array(selected_corners)
            idx = np.lexsort((pts[:,0], pts[:,1]))
            s = pts[idx]
            top2, bot2 = s[:2], s[2:]
            tl, tr = top2[np.argsort(top2[:,0])]
            bl, br = bot2[np.argsort(bot2[:,0])]
            
            arena_corners_image = np.array([tl, tr, br, bl], dtype=np.float32)
            compute_transformation()
            
            print("✓ Arena corners set successfully!")
            print(f"  TL: ({tl[0]:.1f}, {tl[1]:.1f})")
            print(f"  TR: ({tr[0]:.1f}, {tr[1]:.1f})")
            print(f"  BR: ({br[0]:.1f}, {br[1]:.1f})")
            print(f"  BL: ({bl[0]:.1f}, {bl[1]:.1f})")
            corner_selection_mode = False

def setup_arena_corners_with_camera():
    global corner_selection_mode, selected_corners, arena_corners_image
    
    print("Klik arena corners: TL,TR,BR,BL (4x)")
    print("Tekan 'r' untuk reset, 'q' untuk keluar")
    
    corner_selection_mode = True
    selected_corners = []
    arena_corners_image = None
    
    cv.namedWindow("Camera View")
    cv.setMouseCallback("Camera View", mouse_callback_simple)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca dari kamera")
            break
        
        und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
        undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        display_frame = cv.resize(
            undistorted_roi, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv.INTER_LINEAR
        )
        
        instruksi = "Klik 4 titik sudut arena (TL,TR,BR,BL)"
        cv.putText(display_frame, instruksi,
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2, cv.LINE_AA)
        
        if len(selected_corners) > 0:
            txt_count = f"Corner ke-{len(selected_corners)}/4"
            cv.putText(display_frame, txt_count,
                       (10, 55), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 200, 0), 1, cv.LINE_AA)
        
        for i, corner in enumerate(selected_corners):
            corner_display = (corner * scale_factor).astype(int)
            cv.circle(display_frame, tuple(corner_display), 8, (0, 255, 0), -1)
            cv.circle(display_frame, tuple(corner_display), 12, (255, 255, 255), 2)
            corner_names = ["TL", "TR", "BR", "BL"]
            if i < len(corner_names):
                cv.putText(display_frame, corner_names[i],
                          (corner_display[0] + 15, corner_display[1] - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        
        if len(selected_corners) >= 2:
            for i in range(len(selected_corners) - 1):
                pt1 = (selected_corners[i] * scale_factor).astype(int)
                pt2 = (selected_corners[i+1] * scale_factor).astype(int)
                cv.line(display_frame, tuple(pt1), tuple(pt2), (255, 255, 0), 2)
        
        if len(selected_corners) == 4:
            pt1 = (selected_corners[3] * scale_factor).astype(int)
            pt2 = (selected_corners[0] * scale_factor).astype(int)
            cv.line(display_frame, tuple(pt1), tuple(pt2), (255, 255, 0), 2)
            cv.putText(display_frame, "Arena setup complete! Press any key to continue",
                       (10, 85), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2, cv.LINE_AA)
        
        cv.imshow("Camera View", display_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Setup dibatalkan oleh user")
            corner_selection_mode = False
            cv.destroyWindow("Camera View")
            return False
        elif key == ord('r'):
            print("Arena corners reset!")
            selected_corners = []
            arena_corners_image = None
            compute_transformation()
        elif len(selected_corners) == 4 and key != 255:
            break
    
    corner_selection_mode = False
    cv.destroyWindow("Camera View")
    
    if len(selected_corners) == 4:
        print("✓ Arena corners setup berhasil!")
        return True
    else:
        print("✗ Arena corners setup gagal")
        return False

def get_camera_frame_with_markers(robot_marker_id=2):
    ret, frame = cap.read()
    if not ret:
        return {'success': False, 'frame': None, 'error': 'Camera read failed'}

    # Undistort frame
    und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
    undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    display_frame = cv.resize(
        undistorted_roi, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv.INTER_LINEAR
    )

    # Detect markers
    gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray_full, marker_dict, parameters=param_markers)

    robot_found = False
    if ids is not None:
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
        for i, mid in enumerate(ids.flatten()):
            pts = corners[i].reshape(-1, 1, 2).astype(np.float32)
            pts_ud = cv.undistortPoints(pts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
            pts_ud -= np.array([x_roi, y_roi])
            refined = np.array([refine_corner(undistorted_roi, c, win_size=5) for c in pts_ud])
            center_px = refined.mean(axis=0)

            world_xy = camera_to_world_coords(center_px)
            dist3d = calculate_distance(tVecs[i][0])

            if mid == robot_marker_id:
                robot_found = True

            poly = (pts_ud * scale_factor).astype(int)
            color = (0, 255, 0) if mid == robot_marker_id else (0, 255, 255)
            thickness = 3 if mid == robot_marker_id else 2
            cv.polylines(display_frame, [poly], True, color, thickness, cv.LINE_AA)

            axis_len = MARKER_SIZE * 0.75
            axis3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0]]).reshape(-1, 3)
            imgpts, _ = cv.projectPoints(axis3d, rVecs[i], tVecs[i], cam_mat, dist_coef)
            imgpts = cv.undistortPoints(imgpts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
            imgpts = ((imgpts - np.array([x_roi, y_roi])) * scale_factor).astype(int)
            org = tuple(imgpts[0])
            cv.line(display_frame, org, tuple(imgpts[1]), (0, 0, 255), 2)
            cv.line(display_frame, org, tuple(imgpts[2]), (0, 255, 0), 2)
            cv.putText(display_frame, "X", tuple(imgpts[1]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 0, 255), 1)
            cv.putText(display_frame, "Y", tuple(imgpts[2]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 255, 0), 1)

            tr = tuple(poly[1])
            font = cv.FONT_HERSHEY_SIMPLEX
            scale_txt = 0.5 * scale_factor
            thk = 1
            txt1 = f"ID:{mid} {'(ROBOT)' if mid == robot_marker_id else ''} D:{dist3d:.1f}cm"
            txt2 = f"x:{tVecs[i][0][0]:.1f} y:{tVecs[i][0][1]:.1f}"
            (w1, h1), _ = cv.getTextSize(txt1, font, scale_txt, thk)
            (w2, h2), _ = cv.getTextSize(txt2, font, scale_txt, thk)
            pad = 4
            bx = tr[0]
            by = tr[1] - (h1 + h2 + pad * 3)
            box_color = (0, 150, 0) if mid == robot_marker_id else (0, 0, 0)
            cv.rectangle(display_frame, (bx, by), (bx + max(w1, w2) + pad * 2, by + h1 + h2 + pad * 2), box_color, cv.FILLED)
            cv.putText(display_frame, txt1, (bx + pad, by + h1 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)
            cv.putText(display_frame, txt2, (bx + pad, by + h1 + h2 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)

    status_text = f"Robot ID {robot_marker_id}: {'FOUND' if robot_found else 'NOT FOUND'}"
    status_color = (0, 255, 0) if robot_found else (0, 0, 255)
    cv.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv.LINE_AA)

    return {'success': True, 'frame': display_frame, 'robot_found': robot_found}

def show_camera_with_arena_tracking(robot_marker_id=2, duration_seconds=10):
    print(f"\nCamera View dengan Arena Tracking")
    print(f"Target Robot Marker ID: {robot_marker_id}")
    print("Tekan 'q' untuk keluar")
    
    cv.namedWindow("Camera View")
    cv.namedWindow("Arena Tracking")
    start_time = time.time()
    
    robot_found = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
        undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        display_frame = cv.resize(
            undistorted_roi, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv.INTER_LINEAR
        )
        
        dh, dw = display_frame.shape[:2]
        arena_view = 255 * np.ones((dh, dw, 3), dtype=np.uint8)
        a_scale = min(dw/ARENA_WIDTH, dh/ARENA_HEIGHT)
        ox = int((dw - ARENA_WIDTH * a_scale) / 2)
        oy = int((dh - ARENA_HEIGHT * a_scale) / 2)
        
        if transformation_matrix is not None:
            cv.rectangle(
                arena_view,
                (ox, oy),
                (ox + int(ARENA_WIDTH * a_scale), oy + int(ARENA_HEIGHT * a_scale)),
                (0, 0, 0), 2
            )
            step = int(grid_step_cm * a_scale)
            for yy in range(0, int(ARENA_HEIGHT * a_scale) + 1, step):
                cv.line(arena_view, (ox, oy + yy), (ox + int(ARENA_WIDTH * a_scale), oy + yy), (200, 200, 200), 1)
            for xx in range(0, int(ARENA_WIDTH * a_scale) + 1, step):
                cv.line(arena_view, (ox + xx, oy), (ox + xx, oy + int(ARENA_HEIGHT * a_scale)), (200, 200, 200), 1)
        
        gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_full, marker_dict, parameters=param_markers)
        
        robot_found = False
        if ids is not None:
            rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1, 1, 2).astype(np.float32)
                pts_ud = cv.undistortPoints(pts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
                pts_ud -= np.array([x_roi, y_roi])
                refined = np.array([refine_corner(undistorted_roi, c, win_size=5) for c in pts_ud])
                center_px = refined.mean(axis=0)
                
                world_xy = camera_to_world_coords(center_px)
                dist3d = calculate_distance(tVecs[i][0])
                
                if mid == robot_marker_id:
                    robot_found = True
                
                poly = (pts_ud * scale_factor).astype(int)
                color = (0, 255, 0) if mid == robot_marker_id else (0, 255, 255)
                thickness = 3 if mid == robot_marker_id else 2
                cv.polylines(display_frame, [poly], True, color, thickness, cv.LINE_AA)
                
                axis_len = MARKER_SIZE * 0.75
                axis3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0]]).reshape(-1, 3)
                imgpts, _ = cv.projectPoints(axis3d, rVecs[i], tVecs[i], cam_mat, dist_coef)
                imgpts = cv.undistortPoints(imgpts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
                imgpts = ((imgpts - np.array([x_roi, y_roi])) * scale_factor).astype(int)
                org = tuple(imgpts[0])
                cv.line(display_frame, org, tuple(imgpts[1]), (0, 0, 255), 2)
                cv.line(display_frame, org, tuple(imgpts[2]), (0, 255, 0), 2)
                cv.putText(display_frame, "X", tuple(imgpts[1]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 0, 255), 1)
                cv.putText(display_frame, "Y", tuple(imgpts[2]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 255, 0), 1)
                
                tr = tuple(poly[1])
                font = cv.FONT_HERSHEY_SIMPLEX
                scale_txt = 0.5 * scale_factor
                thk = 1
                txt1 = f"ID:{mid} {'(ROBOT)' if mid == robot_marker_id else ''} D:{dist3d:.1f}cm"
                txt2 = f"x:{tVecs[i][0][0]:.1f} y:{tVecs[i][0][1]:.1f}"
                (w1, h1), _ = cv.getTextSize(txt1, font, scale_txt, thk)
                (w2, h2), _ = cv.getTextSize(txt2, font, scale_txt, thk)
                pad = 4
                bx = tr[0]
                by = tr[1] - (h1 + h2 + pad * 3)
                
                box_color = (0, 150, 0) if mid == robot_marker_id else (0, 0, 0)
                cv.rectangle(display_frame, (bx, by), (bx + max(w1, w2) + pad * 2, by + h1 + h2 + pad * 2), box_color, cv.FILLED)
                cv.putText(display_frame, txt1, (bx + pad, by + h1 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)
                cv.putText(display_frame, txt2, (bx + pad, by + h1 + h2 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)
                
                if transformation_matrix is not None:
                    ax = int(world_xy[0] * a_scale) + ox
                    ay = int(world_xy[1] * a_scale) + oy
                    circle_color = (0, 255, 0) if mid == robot_marker_id else (255, 0, 0)
                    cv.circle(arena_view, (ax, ay), 6, circle_color, -1)
                    cv.putText(arena_view, f"id: {mid}", (ax + 8, ay - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)
                    txt_w = f"world: ({world_xy[0]:.1f},{world_xy[1]:.1f})cm"
                    cv.putText(arena_view, txt_w, (ax + 8, ay + 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)
        
        fps = int(cap.get(cv.CAP_PROP_FPS))
        pad = 10
        font = cv.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        th = 1
        lh = int(20 * fs)
        overlay = arena_view.copy()
        box_h = lh * 2 + pad * 2
        box_w = 200
        cv.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (255, 255, 255), -1)
        arena_view = cv.addWeighted(overlay, 0.6, arena_view, 0.4, 0)
        cv.putText(arena_view, f"Arena: {ARENA_WIDTH}x{ARENA_HEIGHT} cm", (pad * 2, pad + lh), font, fs, (0, 0, 0), th, cv.LINE_AA)
        
        status_text = f"Robot ID {robot_marker_id}: {'FOUND' if robot_found else 'NOT FOUND'}"
        status_color = (0, 255, 0) if robot_found else (0, 0, 255)
        cv.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv.LINE_AA)
        
        elapsed = time.time() - start_time
        if duration_seconds > 0:
            remaining = max(0, duration_seconds - elapsed)
            cv.putText(arena_view, f"Time: {remaining:.1f}s", (dw - 100, pad + lh), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            if elapsed >= duration_seconds:
                break
        
        cv.imshow("Camera View", display_frame)
        cv.imshow("Arena Tracking", arena_view)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            arena_corners_image = None
            compute_transformation()
            print("Arena corners reset!")
    
    cv.destroyWindow("Camera View")
    cv.destroyWindow("Arena Tracking")
    print("Camera view closed")
    
    return robot_found

def show_camera_with_arena_tracking_with_log(robot_marker_id=2, duration_seconds=30, logfile=None):
    print(f"\nCamera View dengan Arena Tracking + Logging")
    print(f"Target Robot Marker ID: {robot_marker_id}")
    print("Tekan 'q' untuk keluar")
    
    cv.namedWindow("Camera View")
    cv.namedWindow("Arena Tracking")
    start_time = time.time()
    
    robot_found = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
        undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        display_frame = cv.resize(
            undistorted_roi, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv.INTER_LINEAR
        )
        
        dh, dw = display_frame.shape[:2]
        arena_view = 255 * np.ones((dh, dw, 3), dtype=np.uint8)
        a_scale = min(dw/ARENA_WIDTH, dh/ARENA_HEIGHT)
        ox = int((dw - ARENA_WIDTH * a_scale) / 2)
        oy = int((dh - ARENA_HEIGHT * a_scale) / 2)
        
        if transformation_matrix is not None:
            cv.rectangle(
                arena_view,
                (ox, oy),
                (ox + int(ARENA_WIDTH * a_scale), oy + int(ARENA_HEIGHT * a_scale)),
                (0, 0, 0), 2
            )
            step = int(grid_step_cm * a_scale)
            for yy in range(0, int(ARENA_HEIGHT * a_scale) + 1, step):
                cv.line(arena_view, (ox, oy + yy), (ox + int(ARENA_WIDTH * a_scale), oy + yy), (200, 200, 200), 1)
            for xx in range(0, int(ARENA_WIDTH * a_scale) + 1, step):
                cv.line(arena_view, (ox + xx, oy), (ox + xx, oy + int(ARENA_HEIGHT * a_scale)), (200, 200, 200), 1)
        
        gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_full, marker_dict, parameters=param_markers)
        
        robot_found = False
        if ids is not None:
            rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1, 1, 2).astype(np.float32)
                pts_ud = cv.undistortPoints(pts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
                pts_ud -= np.array([x_roi, y_roi])
                refined = np.array([refine_corner(undistorted_roi, c, win_size=5) for c in pts_ud])
                center_px = refined.mean(axis=0)
                
                world_xy = camera_to_world_coords(center_px)
                dist3d = calculate_distance(tVecs[i][0])
                
                if mid == robot_marker_id:
                    robot_found = True
                
                poly = (pts_ud * scale_factor).astype(int)
                color = (0, 255, 0) if mid == robot_marker_id else (0, 255, 255)
                thickness = 3 if mid == robot_marker_id else 2
                cv.polylines(display_frame, [poly], True, color, thickness, cv.LINE_AA)
                
                axis_len = MARKER_SIZE * 0.75
                axis3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0]]).reshape(-1, 3)
                imgpts, _ = cv.projectPoints(axis3d, rVecs[i], tVecs[i], cam_mat, dist_coef)
                imgpts = cv.undistortPoints(imgpts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
                imgpts = ((imgpts - np.array([x_roi, y_roi])) * scale_factor).astype(int)
                org = tuple(imgpts[0])
                cv.line(display_frame, org, tuple(imgpts[1]), (0, 0, 255), 2)
                cv.line(display_frame, org, tuple(imgpts[2]), (0, 255, 0), 2)
                cv.putText(display_frame, "X", tuple(imgpts[1]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 0, 255), 1)
                cv.putText(display_frame, "Y", tuple(imgpts[2]), cv.FONT_HERSHEY_PLAIN, 1 * scale_factor, (0, 255, 0), 1)
                
                tr = tuple(poly[1])
                font = cv.FONT_HERSHEY_SIMPLEX
                scale_txt = 0.5 * scale_factor
                thk = 1
                txt1 = f"ID:{mid} {'(ROBOT)' if mid == robot_marker_id else ''} D:{dist3d:.1f}cm"
                txt2 = f"x:{tVecs[i][0][0]:.1f} y:{tVecs[i][0][1]:.1f}"
                (w1, h1), _ = cv.getTextSize(txt1, font, scale_txt, thk)
                (w2, h2), _ = cv.getTextSize(txt2, font, scale_txt, thk)
                pad = 4
                bx = tr[0]
                by = tr[1] - (h1 + h2 + pad * 3)
                
                box_color = (0, 150, 0) if mid == robot_marker_id else (0, 0, 0)
                cv.rectangle(display_frame, (bx, by), (bx + max(w1, w2) + pad * 2, by + h1 + h2 + pad * 2), box_color, cv.FILLED)
                cv.putText(display_frame, txt1, (bx + pad, by + h1 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)
                cv.putText(display_frame, txt2, (bx + pad, by + h1 + h2 + pad // 2), font, scale_txt, (0, 255, 255), thk, cv.LINE_AA)
                
                if logfile:
                    logfile.write(f"ID:{mid}\n")
                    logfile.write(f"tVec: {tVecs[i][0]}  Dist: {dist3d:.2f}cm  World: {world_xy}\n")
                    logfile.write("---\n")
                    logfile.flush()
                
                if transformation_matrix is not None:
                    ax = int(world_xy[0] * a_scale) + ox
                    ay = int(world_xy[1] * a_scale) + oy
                    circle_color = (0, 255, 0) if mid == robot_marker_id else (255, 0, 0)
                    cv.circle(arena_view, (ax, ay), 6, circle_color, -1)
                    cv.putText(arena_view, f"id: {mid}", (ax + 8, ay - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)
                    txt_w = f"world: ({world_xy[0]:.1f},{world_xy[1]:.1f})cm"
                    cv.putText(arena_view, txt_w, (ax + 8, ay + 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)
        
        fps = int(cap.get(cv.CAP_PROP_FPS))
        pad = 10
        font = cv.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        th = 1
        lh = int(20 * fs)
        overlay = arena_view.copy()
        box_h = lh * 2 + pad * 2
        box_w = 200
        cv.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (255, 255, 255), -1)
        arena_view = cv.addWeighted(overlay, 0.6, arena_view, 0.4, 0)
        cv.putText(arena_view, f"Arena: {ARENA_WIDTH}x{ARENA_HEIGHT} cm", (pad * 2, pad + lh), font, fs, (0, 0, 0), th, cv.LINE_AA)
        txt_fps = f"FPS: {fps}"
        (w_txt, _), _ = cv.getTextSize(txt_fps, font, fs, th)
        x_txt = dw - w_txt - pad
        cv.rectangle(arena_view, (x_txt - pad // 2, pad), (x_txt + w_txt + pad // 2, pad + box_h // 2), (255, 255, 255), -1)
        cv.putText(arena_view, txt_fps, (x_txt, pad + lh), font, fs, (0, 0, 0), th, cv.LINE_AA)
        
        status_text = f"Robot ID {robot_marker_id}: {'FOUND' if robot_found else 'NOT FOUND'}"
        status_color = (0, 255, 0) if robot_found else (0, 0, 255)
        cv.putText(display_frame, status_text, (10, display_frame.shape[0] - 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv.LINE_AA)
        
        elapsed = time.time() - start_time
        if duration_seconds > 0:
            remaining = max(0, duration_seconds - elapsed)
            cv.putText(display_frame, f"Time: {remaining:.1f}s", (10, display_frame.shape[0] - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            if elapsed >= duration_seconds:
                break
        
        cv.imshow("Camera View", display_frame)
        cv.imshow("Arena Tracking", arena_view)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            arena_corners_image = None
            compute_transformation()
            print("Arena corners reset!")
    
    cv.destroyWindow("Camera View")
    cv.destroyWindow("Arena Tracking")
    print("Camera view closed")
    
    return robot_found

def get_robot_marker_position(robot_marker_id=2):
    ret, frame = cap.read()
    if not ret:
        return {'success': False, 'error': 'Camera read failed'}

    und = cv.undistort(frame, cam_mat, dist_coef, None, new_cam_mtx)
    undistorted_roi = und[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray_full, marker_dict, parameters=param_markers)

    if ids is not None:
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
        for i, mid in enumerate(ids.flatten()):
            if mid == robot_marker_id:
                pts = corners[i].reshape(-1, 1, 2).astype(np.float32)
                pts_ud = cv.undistortPoints(pts, cam_mat, dist_coef, P=new_cam_mtx).reshape(-1, 2)
                pts_ud -= np.array([x_roi, y_roi])
                refined = np.array([refine_corner(undistorted_roi, c, win_size=5) for c in pts_ud])
                center_px = refined.mean(axis=0)

                world_xy = camera_to_world_coords(center_px)
                dist3d = calculate_distance(tVecs[i][0])
                
                if len(refined) >= 2:
                    dx = refined[1][0] - refined[0][0]
                    dy = refined[1][1] - refined[0][1]
                    theta_marker = math.atan2(dy, dx)
                else:
                    theta_marker = 0.0

                return {
                    'success': True,
                    'x': float(world_xy[0]),
                    'y': float(world_xy[1]),
                    'theta': float(theta_marker),
                    'distance_3d': dist3d,
                    'marker_id': mid
                }
    
    return {'success': False, 'error': f'Marker ID {robot_marker_id} not found'}

# koreksi x y marker
def get_position_correction(current_x_cm, current_y_cm, current_theta, robot_marker_id=2):
    marker_data = get_robot_marker_position(robot_marker_id)
    
    if not marker_data['success']:
        return marker_data
    
    correction_x = marker_data['x'] - current_x_cm
    correction_y = marker_data['y'] - current_y_cm
    correction_theta = marker_data['theta'] - current_theta
    correction_theta = (correction_theta + math.pi) % (2 * math.pi) - math.pi #normalisasi
    error_magnitude = math.sqrt(correction_x**2 + correction_y**2)
    
    return {
        'success': True,
        'correction_x': correction_x,
        'correction_y': correction_y,
        'correction_theta': correction_theta,
        'error_magnitude': error_magnitude,
        'marker_data': marker_data
    }

def close_camera():
    global cap
    if cap:
        cap.release()
    cv.destroyAllWindows()