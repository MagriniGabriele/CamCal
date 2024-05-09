import os
import json
import numpy as np
import cv2 as cv
from load_atis_data import load_atis_data

def generate_frames_from_events(events, delta_time, max_frames=1):
    """
    Generate frames from events at every delta time steps.

    Args:
    - events: Numpy array containing the events
    - delta_time: Time interval (in microseconds) for each frame
    - max_frames: Maximum number of frames to generate

    Returns:
    - frames: List of frames (numpy arrays)
    """
    frames = []
    current_time = events[2][0] + delta_time
    current_frame = np.zeros((720, 1280), dtype=np.uint8)

    for i, ts in enumerate(events[2]):
        if ts >= current_time:
            frames.append(current_frame.copy())
            current_frame.fill(0)  # Resetting frame
            current_time += delta_time

        x = int(events[0][i])
        y = int(events[1][i])
        current_frame[y, x] = 255  # Setting pixel intensity to 255 for events
        print(f"Frame {len(frames)}: {i}/{len(events[2])}")
        if len(frames) >= max_frames:
            break

    return frames

def main(dir = './'):
    delta_time =50000  # Time interval (in microseconds) for each frame
    # Now load data from nested dat files. each file is a different example. each file is inside a folder repredenting its class label.
    # I need to convert each example in a different folder to frames and save them in a new folder.
    

    # Now recursively go through all the folders and files and convert them to frames. each .dat files will make a new folder with the frames inside, and all of the .dat files 
    # Inside the same folder will be put insidie the same folder with the frames. Save all of this in a new folder called frames.
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.dat'):
                # Load data
                print(f"Processing {file}...")
                data = load_atis_data(os.path.join(root, file))
                frames = generate_frames_from_events(data, delta_time)
                # Create a new folder for the frames
                new_folder = "event_frames"
                os.makedirs(os.path.join(root,new_folder), exist_ok=True)
                # Save frames
                print(f"Saving frames to {new_folder}...")
                for i, frame in enumerate(frames):
                    print(f"Saving frame {i}...")
                    cv.imwrite(os.path.join(os.path.join(root,new_folder), f"{i}.png"), frame)
                print("Done!")


    
if __name__ == "__main__":
    main()
    