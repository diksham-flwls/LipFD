# mount s3 bucket for video
# mount s3 bucket for landmarks
# zip through video files
# 

from numpy.typing import NDArray
import numpy as np
import torch
from torch import Tensor
from torch_tps import ThinPlateSpline

import cv2
import zipfile
import tempfile
from glob import glob
import shutil
from PIL import Image, ImageDraw
import random
import subprocess
import shortuuid
from tqdm import tqdm
from pathlib import Path


def get_warped_image(image: NDArray, input_ctrl: Tensor, output_ctrl: Tensor) -> NDArray: # 1 x C x H x W
    height, width, C = image.shape
    tps = ThinPlateSpline(0.5)
    tps.fit(torch.flip(torch.tensor(output_ctrl, dtype=torch.float32), (-1,)), torch.flip(torch.tensor(input_ctrl, dtype=torch.float32), (-1,)))  # because landmarks are width, height
    i = torch.arange(height, dtype=torch.float32)
    j = torch.arange(width, dtype=torch.float32)

    ii, jj = torch.meshgrid(i, j, indexing="ij")
    output_indices = torch.cat((ii[..., None], jj[..., None]), dim=-1)  # Shape (H, W, 2)

    input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(height, width, 2)

    size = torch.tensor((height, width))
    grid = 2 * input_indices / size - 1  # Into [-1, 1]
    grid = torch.flip(grid, (-1,))  # Grid sample works with x,y coordinates, not i, j
    torch_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None, ...]
    warped = torch.nn.functional.grid_sample(torch_image, grid[None, ...], align_corners=False)[0]
    return warped.numpy().transpose(1,2,0)


def read_video(video_file):
    video_capture = cv2.VideoCapture(video_file)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    for _ in range(0,frame_count):
        try:
            ret, image = video_capture.read()
            images.append(image)
        except Exception as e:
            print(f"Found an error while reading frames for file:{video_file}")
            print(str(e))
            return None
    video_capture.release()
    images = np.stack(images,axis=0).astype(np.float32)
    return images, fps

def get_landmarks(landmark_file):
    archive = zipfile.ZipFile(landmark_file, 'r')
    tmpdir = tempfile.TemporaryDirectory()
    archive.extractall(tmpdir.name)
    landmarks = [np.load(file) for file in glob(f"{tmpdir.name}/*.npy")]
    return np.stack(landmarks, axis=0).astype(np.float32)

def draw_landmarks(lms, height, width, image=None, color=None, size=None):
    if image is None:
        image = Image.new('RGB', (height, width))
    draw_big_points(image, lms, color, size)
    return image

def draw_big_points(image, points, color="yellow", size=1):
    for x,y in points:
        ImageDraw.Draw(image).ellipse([x-size/2,y-size/2,x+size//2,y+size//2], fill=color)

mouth_landmark_ids = list(range(76, 96))

corners = np.array(  # Add corners ctrl points
    [
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ], dtype=np.float32
)

def build_control_points(landmarks, size, gamma, interpolate=False):
    modified_landmarks = landmarks.copy()
    if interpolate:
        num_frames = len(landmarks)
        keyframes = random.sample(range(num_frames), int(0.25 * num_frames))
        keyframes += [0, num_frames-1]
        for lm in modified_landmarks[keyframes]:
            lm[mouth_landmark_ids] = lm[mouth_landmark_ids] + np.random.rand(lm[mouth_landmark_ids].shape[0], 2) * gamma

        from scipy.interpolate import interp1d as interp1d
        f = interp1d(keyframes, modified_landmarks[keyframes][:,mouth_landmark_ids], axis=0)
        for idx, lm in enumerate(modified_landmarks):
            lm[mouth_landmark_ids] = f(idx)
    else:
        for lm in modified_landmarks:
            lm[mouth_landmark_ids] = lm[mouth_landmark_ids] + np.random.rand(lm[mouth_landmark_ids].shape[0], 2) * gamma
    corner_landmarks = corners * size
    landmarks = np.concatenate((landmarks, corner_landmarks[None].repeat(len(landmarks), axis=0)), axis=1)
    modified_landmarks = np.concatenate((modified_landmarks, corner_landmarks[None].repeat(len(landmarks), axis=0)), axis=1)
    return landmarks, modified_landmarks


def write_video(frames, fps, save_path=None):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    if save_path is None:
        save_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    out = cv2.VideoWriter(save_path, fourcc, fps, (height,width))
    for frame in frames:
        out.write(np.uint8(frame))
    out.release()
    return save_path

def add_audio_to_video(video_path, audio_path, target_path):
    temp_file = f"/tmp/{shortuuid.uuid()}.mp4"
    cmd = f"ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac -shortest {temp_file}"
    subprocess.call(cmd, shell=True, stdout=None)
    shutil.copyfile(temp_file, target_path)

def warp_video(video_path, landmarks_path, target_path, interpolate=False):
    frames, fps = read_video(str(video_path))
    _, height, width, C = frames.shape
    size = np.array([height, width])

    landmarks = get_landmarks(landmarks_path)
    frames = frames[:len(landmarks)]

    input_ctrl_points, output_ctrl_points = build_control_points(landmarks, size, gamma=3, interpolate=interpolate)

    warped_frames = [get_warped_image(frame, input_ctrl_points[idx], output_ctrl_points[idx]) for idx,frame in enumerate(frames)]
    output_video_path = write_video(warped_frames, fps)
    add_audio_to_video(output_video_path, str(video_path), str(target_path))

# def generate_warped_videos(video_path, interpolate=False):
#     video_id = video_path.stem
#     shot_id = video_path.parent.stem
#     speaker_id = video_path.parent.parent.stem
#     landmarks_path = voxceleb_landmarks_repo_dir / speaker_id / voxceleb_landmarks_subdir / f"{shot_id}_{video_id}_track_000.zip"
#     target_dir = voxceleb_target_dir / speaker_id / shot_id
#     target_dir.mkdir(parents=True, exist_ok=True)
#     target_path = target_dir / f"{video_path.stem}.mp4"
#     try:
#         warp_video(video_path, landmarks_path, target_path, interpolate)
#     except Exception as e:
#         print(f"Found an error while trying to generate warped video for: {video_path}")
#         return False, e
#     return True, None

# list files from S3 bucket
voxceleb_landmarks_repo_dir = Path("/home/diksha.meghwal/staging-centipede/centipede/diksha/shared_datasets/talking_heads/voxceleb/repo/")
voxceleb_landmarks_subdir = "train_keypoints_2023-07-10"
voxceleb_videos_dir = Path("/home/diksha.meghwal/sdp-external-restricted/voxceleb/v2/test/mp4/mp4")
voxceleb_target_dir = Path("/home/diksha.meghwal/research-eu-west/centipede/diksha/shared_datasets/talking_heads/voxceleb")
S3_config = {
    "video_landmarks_store_bucket": "flwls-staging-eu-west-1-research-centipede",
    "video_landmarks_store_path": "centipede/diksha/shared_datasets/talking_heads/voxceleb/repo/",
    "video_source_bucket": "flwls-sdp-eu-west-1-external-restricted",
    "video_source_path": "voxceleb/v2/test/mp4/mp4",
    "video_target_bucket": "flwls-research-eu-west-1",
    "video_target_path": "centipede/diksha/shared_datasets/talking_heads/voxceleb/"
}

# from multiprocessing import Pool, cpu_count
# from functools import partial

# video_files = list(voxceleb_videos_dir.glob("*/*/*.mp4"))

# pool = Pool(processes=(cpu_count()))

# error_file = "./warp_error.log"
# error_log = open(error_file, "a", buffering=1)

# with tqdm(total=len(video_files)) as pbar:
#     for result in pool.imap_unordered(generate_warped_videos, video_files):
#     # for video_file in video_files:
#         # result = generate_warped_videos(video_file)
#         pbar.update()
#         success_status, error_msg = result
#         if not success_status:
#             error_log.write(f"{error_msg}\n")

#     # input_lms = draw_landmarks(input_ctrl_points[0], height, width, None, "blue", 3)
#     # output_lms = draw_landmarks(output_ctrl_points[0], height, width, input_lms, "yellow", 2)
#     # input_lms.save("input_lms.png","PNG")
#     # output_lms.save("output_lms.png","PNG")

from flytekit import Resources, map_task, task, workflow
from scratch.s3_boto import list_subdir, list_files, download_file, upload_file
from typing import List, Tuple

@task
def get_file_list(s3_config: dict[str, str] = S3_config) -> List:
    results = []
    for speaker_path in list_subdir(s3_config["video_source_bucket"], s3_config["video_source_path"]):
        for shot_path in list_subdir(s3_config["video_source_bucket"], speaker_path):
            for video_path in list_files(s3_config["video_source_bucket"], shot_path):
                video_s3_uri = f"{s3_config["video_source_bucket"]}/{video_path}"
                speaker_id = speaker_path.split("/")[-2]
                shot_id = shot_path.split("/")[-2]
                video_id, video_suffix = video_path.split("/")[-1].split(".")
                landmark_s3_uri = (
                    f"{s3_config["video_landmarks_store_bucket"]}/"
                    f"{s3_config["video_landmarks_store_path"]}"
                    f"{speaker_id}/"
                    f"{voxceleb_landmarks_subdir}/"
                    f"{shot_id}_{video_id}_track_000.zip"
                )
                target_video_s3_uri = (
                    f"{s3_config["video_target_bucket"]}/"
                    f"{s3_config["video_target_path"]}"
                    f"{speaker_id}/"
                    f"{shot_id}/"
                    f"{video_id}.{video_suffix}"
                )
                results.append((video_s3_uri, landmark_s3_uri, target_video_s3_uri))
    return results

# @workflow
def workflow_runner(s3_config):
    s3_uris = get_file_list(s3_config=s3_config)
    map_task(task_function=generate_warped_videos, min_success_ratio=0.5)(
        s3_uris=s3_uris
    )

# @task
def generate_warped_videos(s3_uris):
    video_path, landmark_path, target_path = s3_uris
    source_video_file = tempfile.NamedTemporaryFile(suffix=".mp4").name
    landmark_file = tempfile.NamedTemporaryFile(suffix=".mp4").name
    target_file = tempfile.NamedTemporaryFile(suffix=".mp4").name
    download_file(video_path, source_video_file)
    download_file(landmark_path, landmark_file)
    download_file(target_path, target_file)
    warp_video(source_video_file, landmark_file, target_file)
    upload_file(target_file, target_path)


breakpoint()
# for source_video_uri, landmark_uri, target_video_uri in get_file_list():
#     print(source_video_uri)
#     print(landmark_uri)
#     print(target_video_uri)

workflow_runner(S3_config)