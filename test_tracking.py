import haiku as hk
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree

import torch
import torch.nn.functional as F

from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

import cv2

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using cuda")
else:
    device = torch.device('cpu')
  
def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles


def inference(frames, query_points, model):
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = outputs['tracks'][0], outputs['occlusion'][0], outputs['expected_dist'][0]

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles


# video = media.read_video('/home/kasm-user/tapnet/examplar_videos/horsejump-high.mp4')
video = media.read_video('/home/kasm-user/SimplerEnv-OpenVLA/results_simple_full_eval/google_robot_open_middle_drawer/rt1/rt_1_x_tf_trained_for_002272480_step/episode_1_success_True.mp4')
height, width = video.shape[1:3]
video_path = "myvid.mp4"
media.write_video(video_path, video, fps=10)

model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load('/home/kasm-user/tapnet/checkpoints/bootstapir_checkpoint_v2.pt'))
model = model.to(device)

model = model.eval()
torch.set_grad_enabled(False)

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}
num_points = 1000  # @param {type: "integer"}

frames = media.resize_video(video, (resize_height, resize_width))
query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

# Convert frames and query points to PyTorch tensors 
# - frames: Tensor of shape [num_frames, resize_height, resize_width, 3]
# - query_points: Tensor of shape [num_points, 3] (time, y, x)
frames = torch.tensor(frames).to(device)
query_points = torch.tensor(query_points).to(device)


# Run inference on the preprocessed video frames and sampled query points.
# - tracks: [num_points, num_frames, 2], 2D coordinates of the tracked points across frames
# - visibles: [num_points, num_frames], boolean mask indicating point visibility in each frame
tracks, visibles = inference(frames, query_points, model)

tracks = tracks.cpu().detach().numpy()
visibles = visibles.cpu().detach().numpy()

# Visualize sparse point tracks

# Convert the coordinates of tracked points back to the original video dimensions.
# This ensures that the tracking visualization aligns with the original video size.
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
video_path_w_tracks = "vid_tracks3.mp4"
media.write_video(video_path_w_tracks, video_viz, fps=10)

print("here")