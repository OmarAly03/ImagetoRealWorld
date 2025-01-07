import numpy as np
import cv2 as cv
import glob
import os

# Constants
CALIB_PATH = "testimages"   # Path to folder with calibration images
CHESS_SIZE = (7, 5)         # Number of internal corners in the chessboard pattern
SQUARE_SIZE = 24            # Size of one square on the chessboard in millimeters

def calibrate():
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points for the chessboard (real-world coordinates)
    objp = np.zeros((CHESS_SIZE[0] * CHESS_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_SIZE[0], 0:CHESS_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Initialize arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Load all images in the directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(f"{CALIB_PATH}/{ext}"))

    if not image_files:
        print(f"No images found in {CALIB_PATH}")
        exit()

    # Process each image
    for fname in image_files:
        img = cv.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, CHESS_SIZE, None)

        if ret:
            # Refine the corner locations
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Append object and image points
            objpoints.append(objp)
            imgpoints.append(corners2)

            window_name = f'Chessboard Detection - {fname}'
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(window_name, 1280, 720)  # windows size --> 1280 = 1920, Height = 720
            
            # Draw and show the detected corners
            cv.drawChessboardCorners(img, CHESS_SIZE, corners2, ret)
            cv.imshow(window_name, img)
            cv.waitKey(3000)  # Show image for 3 seconds
        else:
            print(f"Chessboard pattern not found in {fname}")

    cv.destroyAllWindows()

    # Perform camera calibration if enough patterns were detected
    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            # Print the camera matrix
            print("Camera Matrix (Intrinsic Parameters):")
            print(mtx)

            # Save calibration results for future use
            np.savez(f'{CALIB_PATH}/calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"Calibration data saved to {CALIB_PATH}/calibration.npz")
        else:
            print("Calibration failed.")
    else:
        print("No valid chessboard patterns found in the images.")

    return ret, mtx, dist, rvecs, tvecs

def print_calibration_results(ret, mtx, dist, rvecs, tvecs):
    """Print all camera calibration parameters in a formatted way"""
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
