import os
import raw_to_dat
import dat_to_frames
import recorder_demo
import camera_calibrate
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    num_records = int(input("How many records do you want to make? "))
    for i in range(num_records):
        record()
        print(f"Record {i} done!\n Now move the camera to a new position")
        input("Press Enter to continue...")
    calibration_data = calibration()
    print("Calibration done!")
    print("Calibration data:")
    print(calibration_data)




if __name__=="__main__":
    main()




