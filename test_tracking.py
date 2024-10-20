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

from pathlib import Path

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

def sample_grid_points(frame_max_idx, height, width, num_points):
    """Sample grid points with (time, height, width) order."""
    # Calculate the step size for each dimension
    t_step = max(1, frame_max_idx // num_points)
    y_step = max(1, height // num_points)
    x_step = max(1, width // num_points)

    # Generate grid indices
    t = np.arange(0, frame_max_idx + 1, t_step)
    y = np.arange(0, height, y_step)
    x = np.arange(0, width, x_step)

    # Create a grid of points
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')

    # Flatten the grids and stack them into point coordinates
    points = np.stack([tt.ravel(), yy.ravel(), xx.ravel()], axis=-1).astype(np.int32)

    # # Trim excess points if the number exceeds num_points
    # if len(points) > num_points:
    #     points = points[:num_points]

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

def save_tracking_results(tracks, visibles, video_path):
    """Save tracks and visibles to compressed .npz files."""
    video_path = Path(video_path)

    # Save tracks
    save_path_tracks = video_path.with_name(f"{video_path.stem}_tracks.npz")
    np.savez_compressed(save_path_tracks, tracks=tracks)
    print(f"Tracks saved to: {save_path_tracks}")

    # Save visibles
    save_path_visibles = video_path.with_name(f"{video_path.stem}_visibles.npz")
    np.savez_compressed(save_path_visibles, visibles=visibles)
    print(f"Visibles saved to: {save_path_visibles}")

def load_tracking_results(video_path, load_tracks=True, load_visibles=True):
    """Load tracks and visibles from compressed .npz files."""
    video_path = Path(video_path)
    results = {}

    # Load tracks
    if load_tracks:
        load_path_tracks = video_path.with_stem(f"{video_path.stem}_tracks").with_suffix('.npz')
        results['tracks'] = np.load(load_path_tracks)['tracks']
        print(f"Tracks loaded from: {load_path_tracks}")

    # Load visibles
    if load_visibles:
        load_path_visibles = video_path.with_stem(f"{video_path.stem}_visibles").with_suffix('.npz')
        results['visibles'] = np.load(load_path_visibles)['visibles']
        print(f"Visibles loaded from: {load_path_visibles}")

    return results



# video = media.read_video('/home/kasm-user/tapnet/examplar_videos/horsejump-high.mp4')
# video = media.read_video('/home/kasm-user/SimplerEnv-OpenVLA/results_simple_full_eval/google_robot_open_middle_drawer/rt1/rt_1_x_tf_trained_for_002272480_step/episode_1_success_True.mp4')
# video_path = '/home/kasm-user/Uploads/Hand_picks_up_the_can.mp4'
video_path = '/home/kasm-user/SimplerEnv-OpenVLA/octo_policy_video2.mp4'
video = media.read_video(video_path)
height, width = video.shape[1:3]

model = tapir_model.TAPIR(pyramid_level=1)
model.load_state_dict(torch.load('/home/kasm-user/tapnet/checkpoints/bootstapir_checkpoint_v2.pt'))
model = model.to(device)

model = model.eval()
torch.set_grad_enabled(False)

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}
num_points = 30  # @param {type: "integer"}

frames = media.resize_video(video, (resize_height, resize_width))
# query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)
query_points = sample_grid_points(0, frames.shape[1], frames.shape[2], num_points)

# Convert frames and query points to PyTorch tensors 
# - frames: Tensor of shape [num_frames, resize_height, resize_width, 3]
# - query_points: Tensor of shape [num_points, 3] (time, y, x)
frames = torch.tensor(frames).to(device)
query_points = torch.tensor(query_points).to(device)


# Run inference on the preprocessed video frames and sampled query points.
# - tracks: [num_points, num_frames, 2], 2D coordinates of the tracked points across frames
# - visibles: [num_points, num_frames], boolean mask indicating point visibility in each frame
print("starting inf")
tracks, visibles = inference(frames, query_points, model)
print("done inf")
tracks = tracks.cpu().detach().numpy()
visibles = visibles.cpu().detach().numpy()



# Visualize sparse point tracks
# Convert the coordinates of tracked points back to the original video dimensions.
# This ensures that the tracking visualization aligns with the original video size.
print("convert to grid coord now")
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
save_tracking_results(tracks, visibles, video_path) # should save tracks w respect to og video resoultion
print("painting point tracks now ")
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
video_path_w_tracks = f"{Path(video_path).with_suffix('')}_w_tracks_viz_from_test_tracking.mp4"
print("writing video")
media.write_video(video_path_w_tracks, video_viz, fps=10)
print(f"saved video w viz to {video_path_w_tracks}")
print("here")