from recorder_api import Recorder
import time

def main(recording_folder):
    rec = Recorder(recording_folder)

    time.sleep(3)

    rec.start_recording_rgb_and_event()

    # wait 5 seconds
    start_time = time.time()

    while time.time() - start_time < 0.3:
        print('waiting...')
        pass
    print('done waiting')

    rec.stop_recording_rgb_and_event()

    time.sleep(3)


if __name__ == "__main__":
    main()