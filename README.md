# SASA

This repository is the application of Semantics-Augmented Set Abstraction (SASA) in self-collected datasets.
## Getting Started

### Requirements
* Linux
* Python = 3.8
* PyTorch = 1.6.0
* CUDA = 10.2
* CMake >= 3.13.2
* [`spconv v1.2`](https://github.com/traveller59/spconv/tree/v1.2.1)

### Installation
a. Install `spconv` library.
```shell
git clone https://github.com/traveller59/spconv.git
cd spconv
git checkout v1.2.1
git submodule update --init --recursive
python setup.py bdist_wheel
pip install ./dist/spconv-1.2.1-cp36-cp36m-linux_x86_64.whl   # wheel file name may be different
cd ..
```

b. Install `pcdet` toolbox.
```shell
cd ${your path to SASA}
pip install -r requirements.txt
python setup.py develop
```

### Data Preparation
a. Prepare datasets. \
Organise your datasets as follows (in format of kitti or nuscenes).
```
SASA
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   ├── testing
│   │   ├── calib & velodyne & image_2
│   ├── nuscenes
│   │   ├── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   ├── samples
│   │   │   ├── sweeps
│   │   │   ├── maps
│   │   │   ├── v1.0-trainval  
├── pcdet
├── tools
```

b. Generate data infos.
```shell
# KITTI dataset
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

# nuScenes dataset
pip install nuscenes-devkit==1.0.5
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \ 
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

### Training
* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

* Train with multiple GPUs:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

### Testing
* Test a pretrained model with a single GPU:
```shell script
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```