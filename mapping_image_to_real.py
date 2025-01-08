import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_camera_params():
    """Get camera intrinsic and extrinsic parameters"""
    K = np.array([
        [3.86982417e+03, 0.00000000e+00, 2.57473721e+03],
        [0.00000000e+00, 3.94926316e+03, 1.53485806e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    rvec = np.array([-0.48879785, -0.29172352, -0.18250088])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[-165.8908412], [-88.68793195], [411.4511886]])
    return K, R, t

def world_to_image(X, Y, Z, K, R, t):
    """Transform world coordinates to image coordinates"""
    world_point = np.array([[X], [Y], [Z], [1]])
    Rt = np.hstack((R, t))
    camera_point = K @ Rt @ world_point
    u = camera_point[0] / camera_point[2]
    v = camera_point[1] / camera_point[2]
    return u, v

def load_image(path):
    """Load and convert image to RGB"""
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def visualize_projection(x, y, image_size=(5000, 3000)):
    """Visualize projected point on coordinate system"""
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.scatter(x, y, color='red', s=100, label='Projected Point')
    
    plt.xlim(0, image_size[0])
    plt.ylim(0, image_size[1])
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Projected Point in Image Coordinates')
    plt.legend()
    plt.show()

def run_projection(world_point=(100, 100, 0)):
    """Main function to run the projection"""
    # Get camera parameters
    K, R, t = get_camera_params()
    
    # Project point
    x, y = world_to_image(world_point[0], world_point[1], world_point[2], K, R, t)
    print(f"Point at Z={world_point[2]}mm: ({x}, {y})")
    
    # Visualize results
    visualize_projection(x, y)
    
    return x, y
