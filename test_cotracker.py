import torch
# Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'
url = '/home/kasm-user/SimplerEnv-OpenVLA/results_simple_full_eval/google_robot_open_middle_drawer/rt1/rt_1_x_tf_trained_for_002272480_step/episode_1_success_True.mp4'
url = '/home/kasm-user/Uploads/Hand_picks_up_the_can.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 40
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1

from cotracker.utils.visualizer import Visualizer

vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)