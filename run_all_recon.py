import os
import subprocess
import re
from glob import glob

# Define the root directory of the 7-Scenes dataset
ROOT = "/mnt/d/NTU/Course/Second2/Computer_Vision/final/src/SLAM3R/data/7SCENES"

# Get all scene directories (e.g., 'chess', 'office', etc.)
scenes = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
print(f"Found scenes: {scenes}")

# Loop through each scene
for scene in scenes:
    test_dir = os.path.join(ROOT, scene, "test")
    if not os.path.isdir(test_dir):
        print(f"No 'test' directory found for scene {scene}, skipping...")
        continue
    seq_dirs = [
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and re.match(r'^seq-\d+$', d)
    ]
    print(f"Found sequences in {scene}/test: {seq_dirs}")
    for seq_dir in seq_dirs:
        seq_id = int(seq_dir.split('-')[1])
        seq_path = os.path.join(test_dir, seq_dir)
        image_files = sorted(glob(os.path.join(seq_path, "*.color.png")))
        print(f"Sequence {seq_dir} has {len(image_files)} images")
        if len(image_files) < 50:  # Skip sequences with too few images
            print(f"Skipping {scene}/seq-{seq_id:02d}: Too few images ({len(image_files)})")
            continue
        test_name = f"{scene}-seq{seq_id:02d}"
        cmd = [
            "python", "recon.py",
            "--dataset", "seven_scenes",
            "--test_name", test_name,
            "--i2p_weights", "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
            # "--l2w_weights", "./checkpoints/slam3r_7scenes_l2w_one_each/checkpoint-best.pth",
            "--keyframe_stride", "5",
            "--conf_thres_i2p", "3",
            "--initial_winsize", "7",
            "--buffer_size", "50",
            "--num_scene_frame", "5",
            "--save_preds",
            "--device", "cuda",
            "--scene_id", scene,
            "--seq_id", str(seq_id)
        ]
        print(f"Running command: {' '.join(cmd)}")
        log_dir = os.path.join("results", test_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "log.txt"), "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=f)
            if result.returncode != 0:
                print(f"Error running {scene} seq-{seq_id}: Check {os.path.join(log_dir, 'log.txt')} for details")