import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_camera_params():
    """Return camera intrinsic and distortion parameters"""
    mtx = np.array([
        [3.86982417e+03, 0.00000000e+00, 2.57473721e+03],
        [0.00000000e+00, 3.94926316e+03, 1.53485806e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    dist = np.array([0.63263233, -1.99705216, 0.00576365, 0.09056981, 3.61538196])
    rvec = np.array([-0.48879785, -0.29172352, -0.18250088])
    tvec = np.array([-165.8908412, -88.68793195, 411.4511886])
    return mtx, dist, rvec, tvec

def project_3d_to_2d(points_3d, rvec, tvec, mtx, dist):
    """Project 3D points to 2D image coordinates"""
    points_3d = np.array(points_3d, dtype=np.float32)
    img_points_distorted, _ = cv.projectPoints(points_3d, rvec, tvec, mtx, dist)
    img_points_undistorted, _ = cv.projectPoints(points_3d, rvec, tvec, mtx, None)
    return img_points_distorted.reshape(-1, 2), img_points_undistorted.reshape(-1, 2)

def generate_test_points(nx=8, ny=6, square_size=24):
    """Generate grid of 3D test points"""
    x, y = np.meshgrid(np.linspace(0, (nx-1)*square_size, nx), 
                      np.linspace(0, (ny-1)*square_size, ny))
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

def visualize_projections(points_2d_distorted, points_2d_undistorted, image_size=(4032, 3024)):
    """Visualize distorted and undistorted projections"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.scatter(points_2d_distorted[:, 0], points_2d_distorted[:, 1], c='r', marker='.')
    plt.title('Distorted Projection')
    plt.xlim(0, image_size[0])
    plt.ylim(image_size[1], 0)
    plt.grid(True)
    
    plt.subplot(122)
    plt.scatter(points_2d_undistorted[:, 0], points_2d_undistorted[:, 1], c='b', marker='.')
    plt.title('Undistorted Projection')
    plt.xlim(0, image_size[0])
    plt.ylim(image_size[1], 0)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_test_results(test_points, distorted, undistorted):
    """Print comparison of test point projections"""
    print("\nTest Point Projections:")
    print("3D Point (mm) | 2D Distorted (pixels) | 2D Undistorted (pixels)")
    print("-" * 65)
    for i in range(len(test_points)):
        print(f"{test_points[i]} | {distorted[i]} | {undistorted[i]}")

def run_distortion_analysis():
    """Main function to run the distortion analysis"""
    # Get camera parameters
    mtx, dist, rvec, tvec = get_camera_params()
    
    # Generate and project grid points
    points_3d = generate_test_points()
    points_2d_distorted, points_2d_undistorted = project_3d_to_2d(
        points_3d, rvec, tvec, mtx, dist)
    
    # Visualize results
    visualize_projections(points_2d_distorted, points_2d_undistorted)
    
    # Test specific points
    test_points = np.array([
        [0, 0, 0],
        [24, 0, 0],
        [0, 24, 0],
        [24, 24, 0],
        [120, 96, 0],
    ])
    
    distorted, undistorted = project_3d_to_2d(test_points, rvec, tvec, mtx, dist)
    print_test_results(test_points, distorted, undistorted)

if __name__ == "__main__":
    run_distortion_analysis()

