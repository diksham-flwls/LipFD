# %%
from pathlib import Path
import shutil

filenames = "/home/diksha.meghwal/repos/syncnet_trainer/data/test.txt"
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
