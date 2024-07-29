import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--pcd_path', type=str, default='demo_data',
                        help='specify the point cloud data directory')
    parser.add_argument('--pred_path', type=str, required=True, help='prediction result storing directory')
    parser.add_argument('--vis_path', type=str, required=True, help='visualization result storing directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        pred_filename = str(idx + 1) + '.txt'
        combined_array = np.loadtxt(os.path.join(args.pred_path, pred_filename), delimiter=',')
        ref_boxes = combined_array[:, :7]  # n * 7
        ref_scores = np.squeeze(combined_array[:, 7])  # n
        ref_labels = np.squeeze(combined_array[:, 8])  # n

        ref_boxes = torch.from_numpy(ref_boxes)
        ref_scores = torch.from_numpy(ref_scores)
        ref_labels = torch.from_numpy(ref_labels)


        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=ref_boxes,
            ref_scores=ref_scores, ref_labels=ref_labels
        )

        mlab.show(stop=True)
        vis_filename = str(idx + 1) + '.png'
        mlab.savefig(os.path.join(args.vis_path), vis_filename)
    logger.info('Done.')

if __name__ == '__main__':
    main()