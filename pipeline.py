import os
import raw_to_dat
import dat_to_frames
import recorder_demo
import camera_calibrate
import shutil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
 
def test(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, main_Folder="Calibrate_Alternative"):
    # Take an image used for stereo calibration from camera 1
    test_img_path1 = f'{main_Folder}/LEFT/15_rgb.png'
    test_img1 = cv.imread(test_img_path1)
    gray_test_img1 = cv.cvtColor(test_img1, cv.COLOR_BGR2GRAY)

    # Take the same image from camera 2
    test_img_path2 = f'{main_Folder}/RIGHT/15_event.png'
    test_img2 = cv.imread(test_img_path2)
    gray_test_img2 = cv.cvtColor(test_img2, cv.COLOR_BGR2GRAY)

    # Draw a random point on the left image
    x = np.random.randint(0, gray_test_img1.shape[1])
    y = np.random.randint(0, gray_test_img1.shape[0])
    cv.circle(test_img1, (x, y), 5, (0, 255, 0), -1)

    # Map the point to the right image
    left_point = np.array([[x, y]], dtype=np.float32)
    left_point = np.array([left_point])
    right_point = cv.undistortPoints(left_point, camera_matrix_left, dist_coeffs_left, R=R, P=camera_matrix_left)
    right_point = cv.projectPoints(right_point, R, T, camera_matrix_right, dist_coeffs_right)
    right_point = np.squeeze(right_point[0], axis=1)
    right_point = tuple(right_point.astype(int))
    cv.circle(test_img2, right_point, 5, (0, 255, 0), -1)



    # Show the image
    cv.imshow('Left Image', test_img1)
    cv.imshow('Right Image', test_img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
   
 
 




def record(main_Folder="Calibrate_Alternative"):
    # Create a new folder to store the data
    os.makedirs(f"{main_Folder}", exist_ok=True)
    os.makedirs(f"{main_Folder}/Records", exist_ok=True)
    
    # Search last created folder in the Records folder
    folders = [f for f in os.listdir(f"{main_Folder}/Records") if os.path.isdir(os.path.join(f"{main_Folder}/Records", f))]
    folder_nums = [int(f.split('_')[-1]) for f in folders]
    if folder_nums == []:
        new_folder_num = 0
    else:
        new_folder_num = max(folder_nums) + 1 

    # Create new folder
    new_folder = f"{main_Folder}/Records/record_{new_folder_num}"
    os.makedirs(new_folder, exist_ok=True)

    # Start recording RGB and event data
    recorder_demo.main(new_folder)

    # Convert the raw data to .dat files
    raw_to_dat.main(new_folder)

    # Convert the .dat files to frames
    dat_to_frames.main(new_folder)

    os.makedirs(f"{main_Folder}/LEFT", exist_ok=True)
    os.makedirs(f"{main_Folder}/RIGHT", exist_ok=True)

    # Take first element of rgb frames and save it as left image (move and rename)
    os.rename(f"{main_Folder}/Records/record_{new_folder_num}/frames_rgb/frame_0.png", f"{main_Folder}/LEFT/{new_folder_num}_rgb.png")
    
    # Take te event frames and save it as right image (move and rename)
    os.rename(f"{main_Folder}/Records/record_{new_folder_num}/event_frames/0.png", f"{main_Folder}/RIGHT/{new_folder_num}_event.png")

    # Remove the rest of the frames
    shutil.rmtree(f"{main_Folder}/Records/record_{new_folder_num}/frames_rgb")
    shutil.rmtree(f"{main_Folder}/Records/record_{new_folder_num}/event_frames")
    

def calibration(main_Folder="Calibrate_Alternative"):
    # Calibrate the camera
    calibration_data = camera_calibrate.main(f"{main_Folder}/")
    return calibration_data


def main(main_Folder="Calibrate_Alternative"):
    # Collect images
    num_records = int(input("How many records do you want to make? "))
    for i in range(num_records):
        record()
        print(f"Record {i} done!\n Now move the camera to a new position")
        input("Press Enter to continue...")


    # First calibrate each camera separately
    left_images = f"{main_Folder}/LEFT_SINGLE/*.png"
    right_images = f"{main_Folder}/RIGHT_SINGLE/*.png"
    cmtx0, dist0 = camera_calibrate.calibrate_camera_for_intrinsic_parameters(left_images)
    cmtx1, dist1 = camera_calibrate.calibrate_camera_for_intrinsic_parameters(right_images)

    # Then calibrate the stereo camera
    R, T  = camera_calibrate.stereo_calibrate(cmtx0, dist0, cmtx1, dist1, f"{main_Folder}/LEFT/*.png", f"{main_Folder}/RIGHT/*.png")
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]

    # Also calibrate to get homography
    #H = camera_calibrate.get_homography(cmtx0, dist0, cmtx1, dist1, R, T)


    test(cmtx0, dist0, cmtx1, dist1, R, T, main_Folder)

if __name__=="__main__":
    main()




