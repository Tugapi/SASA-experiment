import argparse
import os
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from visual_utils import visualize_utils as V


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pcd_path', type=str, default='demo_data',
                        help='specify the point cloud data directory')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction result storing directory')
    parser.add_argument('--vis_path', type=str, required=True, help='visualization result storing directory')

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    files = os.listdir(args.pcd_path)
    for idx, filename in enumerate(files):
        assert filename.endswith('.bin'), "point cloud files should end with .bin"
        file_path = os.path.join(args.pcd_path, filename)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        pred_filename = str(idx + 1) + '.txt'
        combined_array = np.loadtxt(os.path.join(args.pred_path, pred_filename), delimiter=',')
        ref_boxes = combined_array[:, :7]  # n * 7
        ref_scores = np.squeeze(combined_array[:, 7])  # n
        ref_labels = np.squeeze(combined_array[:, 8]).astype(np.int32)  # n

        ref_boxes = torch.from_numpy(ref_boxes)
        ref_scores = torch.from_numpy(ref_scores)
        ref_labels = torch.from_numpy(ref_labels)

        V.draw_scenes(
            points=points[:, 1:], ref_boxes=ref_boxes,
            ref_scores=ref_scores, ref_labels=ref_labels
        )

        mlab.show(stop=True)
        vis_filename = str(idx + 1) + '.png'
        mlab.savefig(os.path.join(args.vis_path), vis_filename)


if __name__ == '__main__':
    main()
