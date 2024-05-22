# %%
from pathlib import Path
import shutil

filenames = "/home/diksha.meghwal/repos/syncnet_trainer/data/dev_old.txt"
target_dir = Path("./AVLip")
train_dir = target_dir / "train"
val_dir = target_dir / "val"
target_dir.mkdir(exist_ok=True, parents=True)
train_dir.mkdir(exist_ok=True, parents=True)
train_audio_dir = train_dir / "wav"
train_audio_dir.mkdir(exist_ok=True, parents=True)
val_audio_dir = val_dir / "wav"
val_audio_dir.mkdir(exist_ok=True, parents=True)

categories = ["0_real", "1_fake"]

for category in categories:
    (train_dir / category).mkdir(exist_ok=True, parents=True)
    (train_audio_dir/ category).mkdir(exist_ok=True, parents=True)
    (val_dir / category).mkdir(exist_ok=True, parents=True)
    (val_audio_dir/ category).mkdir(exist_ok=True, parents=True)


# %%
n = 40
video_files, audio_files = [], []

with open(filenames, "r") as f:
    counter = 0
    for line in f.readlines():
        print(line)
        video_file, audio_file, _, _ = line.split(" ")
        video_files.append(video_file)
        audio_files.append(audio_file)
        counter += 1
        if counter == n:
            break
# %%
def copy_files(filenames, start_idx, end_idx, target_dir):
    for f in filenames[start_idx:end_idx]:
        file = Path(f)
        new_file_id = f"{file.parent.parent.stem}_{file.parent.stem}_{file.stem}{file.suffix}"
        shutil.copy(f, f"{str(target_dir)}/{new_file_id}")


copy_files(video_files, 0, 10, train_dir/categories[0])
copy_files(audio_files, 0, 10, train_audio_dir/categories[0])
copy_files(video_files, 10, 20, train_dir/categories[1])
copy_files(audio_files, 10, 20, train_audio_dir/categories[1])
copy_files(video_files, 20, 30, val_dir/categories[0])
copy_files(audio_files, 20, 30, val_audio_dir/categories[0])
copy_files(video_files, 30, 40, val_dir/categories[1])
copy_files(audio_files, 30, 40, val_audio_dir/categories[1])

# %%

import subprocess
import platform
from pathlib import Path

# video_dir = "../AVLip/train/1_fake"
# wav_dir = "../AVLip/train/wav/1_fake"
video_dir = "../AVLip/val/1_fake"
wav_dir = "../AVLip/val/wav/1_fake"


for video_file in Path(video_dir).glob("*.mp4"):
    # command = 'ffmpeg -i {} -vn -acodec copy {}'.format(video_file, f"{wav_dir}/{Path(video_file).stem}.wav")
    command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video_file, f"{wav_dir}/{Path(video_file).stem}.wav")
    subprocess.call(command, shell=platform.system() != 'Windows')

# %%

# sample from voxceleb dev

voxceleb_dev_list = "/home/diksha.meghwal/repos/syncnet_trainer/data/dev_v1.txt"

from pathlib import Path
import random

n_train_subset = 50000

f = open(voxceleb_dev_list)
dev_lines = f.readlines()

# %%
indices = list(range(len(dev_lines)))
random.shuffle(indices)
real_train_indices = indices[:n_train_subset]
real_train_data = [dev_lines[idx] for idx in real_train_indices]

AVLip_train_real = Path("../AVLip/train/0_real")
AVLip_train_real.mkdir(exist_ok=True, parents=True)
file_list = AVLip_train_real / "fileList.txt"
with open(file_list, "w+") as f:
    for line in real_train_data:
        f.write(line)
f.close()
# %%
n_fake_Wav2Lip, n_fake_TalkLip, n_fake_MakeItTalk = 15000, 15000, 15000

AVLip_train_fake = Path("../AVLip/train/1_fake")
AVLip_train_fake.mkdir(exist_ok=True, parents=True)
file_list = AVLip_train_fake / "fileList.txt"
fake_train_indices = indices[n_train_subset:n_train_subset + 45000]
fake_train_data = [dev_lines[idx] for idx in fake_train_indices]
with open(file_list, "+w") as f:
    for line in fake_train_data:
        f.write(line)
f.close()

# %%
