from simulation3D import run_simulation 
from camera_calibration import calibrate, print_calibration_results
from mapping_image_to_real import run_projection
from distortion_display import run_distortion_analysis

def main():
    # Configuration parameters
    config = {
        'start_point': (0, 0, 0),
        'end_point': (100, 100, 0),
        'box_dims': (10, 7, 1)
    }
    
    try:
        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = calibrate()
        if not ret:
            raise Exception("Camera calibration failed")
        print_calibration_results(ret, mtx, dist, rvecs, tvecs)

        run_distortion_analysis()

        image_coords = run_projection(config['end_point'])
        if image_coords is None:
            raise Exception("Projection failed")

        animation = run_simulation(
            config['start_point'], 
            config['end_point'], 
            config['box_dims']
        )
        
        return animation
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return None

if __name__ == "__main__":
    main()