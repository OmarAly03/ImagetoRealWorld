import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_robot_object(x, y, z, length, width, height):
    """Create box vertices and faces"""
    vertices = np.array([
        [x, y, z],
        [x + length, y, z],
        [x + length, y + width, z],
        [x, y + width, z],
        [x, y, z + height],
        [x + length, y, z + height],
        [x + length, y + width, z + height],
        [x, y + width, z + height]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[3], vertices[0], vertices[4], vertices[7]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]
    return faces

def setup_simulation(start_point, end_point, box_dims):
    """Setup simulation parameters"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits and labels
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 300)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Robot Motion')
    
    # Calculate path steps
    frames = 100
    x_steps = np.linspace(start_point[0], end_point[0], frames)
    y_steps = np.linspace(start_point[1], end_point[1], frames)
    
    return fig, ax, x_steps, y_steps

def run_simulation(start_point=(0,0,0), end_point=(100,100,0), 
                  box_dims=(10,7,1), frames=100):
    """Main simulation function"""
    # Setup
    fig, ax, x_steps, y_steps = setup_simulation(start_point, end_point, box_dims)
    
    # Create initial box
    box_faces = create_robot_object(start_point[0], start_point[1], start_point[2], 
                          box_dims[0], box_dims[1], box_dims[2])
    box = Poly3DCollection(box_faces, alpha=0.7, edgecolor='k', facecolor='blue')
    ax.add_collection3d(box)
    
    # Add markers
    ax.scatter(*start_point, color='green', s=50, label='Start', marker='o')
    ax.scatter(end_point[0], end_point[1], 0, color='red', s=50, label='End', marker='o')
    
    def update(frame):
        new_faces = create_robot_object(x_steps[frame], y_steps[frame], start_point[2], 
                             box_dims[0], box_dims[1], box_dims[2])
        box.set_verts(new_faces)
        return box,
    
    # Create and run animation
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    plt.legend()
    plt.show()
    
    return ani

if __name__ == "__main__":
    # Example usage
    start_point = (0, 0, 0)
    end_point = (100, 100, 0)
    box_dims = (10, 7, 1)
    
    ani = run_simulation(start_point, end_point, box_dims)