# %%
from pathlib import Path
file_list = "../syncnet_trainer/data/dev_old.txt"
f = open(file_list, "r")
filenames = f.readlines()[:10]

# %%
from headpose.detect import PoseEstimator
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('AGG')
est = PoseEstimator()

yaw_max_val, pitch_max_val, roll_max_val = 45, 20, 60

def get_max_head_pose(video_filename):
    video_capture = cv2.VideoCapture(video_filename)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    assert fps == 25
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    error_frame_counter = 0
    for i in range(frame_count):
        _, frame = video_capture.read()
        try:
            est.detect_landmarks(frame, plot=True)  # plot the result of landmark detection
            yaw, pitch, roll = est.pose_from_image(frame)  # estimate the head pose
        except Exception as e:
            print(f"Error detecting pose for a frame in:{video_filename} idx:{i}")
            error_frame_counter += 1
            if error_frame_counter > 10:
                return False
            continue
        if abs(yaw) >= yaw_max_val or abs(pitch) >= pitch_max_val:
            plt.text(0 ,0, f"yaw:{yaw:.2f} pitch:{pitch:.2f} roll:{roll:.2f}")
            plot_filename = f"{Path(video_filename).parent.parent.stem}_{Path(video_filename).parent.stem}_{Path(video_filename).stem}"
            plt.savefig(f"./{plot_filename}_{i}.png")
            return False
        matplotlib.pyplot.close()
    return True

# %%
video_files = [line.split(" ")[0] for line in filenames]
# %%
filtered_files = list(filter(get_max_head_pose, video_files))
print(filtered_files)
# %%
