import numpy as np
import cv2 as cv
import glob
import os

CALIB_PATH = "testimages"   
CHESS_SIZE = (7, 5)         
SQUARE_SIZE = 24            

def calibrate():
    """Calibrates the camera using chessboard images and saves the calibration data."""
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESS_SIZE[0] * CHESS_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_SIZE[0], 0:CHESS_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
        image_files.extend(glob.glob(f"{CALIB_PATH}/{ext}"))

    if not image_files:
        print(f"No images found in {CALIB_PATH}")
        exit()

    for fname in image_files:
        img = cv.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, CHESS_SIZE, None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            window_name = f'Chessboard Detection - {fname}'
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(window_name, 1280, 720)
            
            cv.drawChessboardCorners(img, CHESS_SIZE, corners2, ret)
            cv.imshow(window_name, img)
            cv.waitKey(3000)
        else:
            print(f"Chessboard pattern not found in {fname}")

    cv.destroyAllWindows()

    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("Camera Matrix (Intrinsic Parameters):")
            print(mtx)

            np.savez(f'{CALIB_PATH}/calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"Calibration data saved to {CALIB_PATH}/calibration.npz")
        else:
            print("Calibration failed.")
    else:
        print("No valid chessboard patterns found in the images.")

    return ret, mtx, dist, rvecs, tvecs

def print_calibration_results(ret, mtx, dist, rvecs, tvecs):
    """Print the results of the camera calibration process."""
    print("\n=== Camera Calibration Results ===")
    print(f"Calibration Success: {ret}\n")
    
    print("Camera Matrix (Intrinsics):")
    print("[fx  0  cx]")
    print("[0   fy cy]")
    print("[0   0   1]")
    print(mtx, "\n")
    
    print("Distortion Coefficients:")
    print("(k1, k2, p1, p2, k3)")
    print(dist, "\n")
    
    print("Rotation Vectors (for each image):")
    for i, rvec in enumerate(rvecs):
        print(f"Image {i+1}:", rvec.flatten())
    print()
    
    print("Translation Vectors (for each image):")
    for i, tvec in enumerate(tvecs):
        print(f"Image {i+1}:", tvec.flatten())

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate()
    print_calibration_results(ret, mtx, dist, rvecs, tvecs)