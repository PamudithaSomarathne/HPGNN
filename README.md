# HPGNN - A Hierarchical Graph Neural Network for Semantic Segmentation of Large Scale Outdoor Point Clouds

Official implementation of the paper ['HPGNN - A Hierarchical Graph Neural Network for Semantic Segmentation of Large Scale Outdoor Point Clouds'](https://www.computer.org/csdl/proceedings-article/icpr/2022/09956238/1IHpg8unl4c) (ICPR 2022).

:heavy_exclamation_mark: It is not possible to retrain this model using this repo with later versions of tensorflow due to data loader differences. The inference can be used with provided checkpoints. We're expecting to port the code to a later version of tensorflow. Sorry for any inconvenience caused. :heavy_exclamation_mark:

## Requirements:
    - tensorflow-gpu 2.5.0
    - scikit-learn 0.24.2
    - matplotlib 3.4.1
    - tensorboard

## Dataset folder structure
	|-dataset
		|-sequences
            |-00
                |-velodyne
                    |-000000.bin
                    |-...
                |labels
                    |-000000.labels
                    |-...
            |-... 
		|-nuscenes
			|-velo
				|-n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin
				|-...
			|-lidarseg
				|-0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin
				|-...

Combine all 'pcd.bin' files in 'samples' folders of NuScenes dataset into the 'velo' folder. Training, validation, and testing without overlap is handled by the split files.

## Training

Initialize the environment
```
conda create -n hpgnn python=3.8
conda activate hpgnn
pip install tensorflow-gpu==2.5.0 scikit-learn==0.24.2 matplotlib==3.4.1 tensorboard
```

KITTI dataset
```
python train_kitti.py --config hpgnn_kitti
```

NuScenes dataset
```
python train_nuscenes.py --config hpgnn_nuscenes
```

## Inference

KITTI dataset
```
python inference_kitti.py --config hpgnn_kitti
```

NuScenes dataset
```
python inference_nuscenes.py --config hpgnn_nuscenes
```
