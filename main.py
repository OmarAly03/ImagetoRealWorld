from simulation3D import run_simulation 
from camera_calibration import calibrate, print_calibration_results
from mapping_image_to_real import run_projection
from distortion_display import run_distortion_analysis

start_point = (0, 0, 0)
end_point = (100, 100, 0)
box_dims = (10, 7, 1)

ret, mtx, dist, rvecs, tvecs = calibrate()
print_calibration_results(ret, mtx, dist, rvecs, tvecs)

run_distortion_analysis()

image_coords = run_projection(end_point)

ani = run_simulation(start_point, end_point, box_dims)