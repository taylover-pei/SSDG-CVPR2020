# SSDG
The implementation of [**Single-Side Domain Generalization for Face Anti-Spoofing**](https://arxiv.org/abs/2004.14043).

The motivation of the proposed SSDG method:
<div align=center>
<img src="https://github.com/taylover-pei/SSDG-CVPR2020/blob/master/article/motivation.png" width="400" height="296" />
</div>

An overview of the proposed SSDG method:

<div align=center>
<img src="https://github.com/taylover-pei/SSDG-CVPR2020/blob/master/article/architecture.png" width="700" height="345" />
</div>

## Congifuration Environment
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-training

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment. All the detected faces are normlaize to 256$\times$256$\times$3, where only RGB channels are utilized for training. 

To be specific, we process every frame of each video and then utilize the `sample_frames` function in the `utils/utils.py` to sample frames during training.

Put the processed frames in the path `$root/data/dataset_name`.

**Data Label Generation.** 

Move to the `$root/data_label` and generate the data label list:
```python
python generate_label.py
```

## Training

Move to the folder `$root/experiment/testing_scenarios/` and just run like this:
```python
python train_ssdg_full.py
```

The file `config.py` contains all the hype-parameters used during training.

## Testing

Run like this:
```python
python dg_test.py
```

## Citation
Please cite our paper if the code is helpful to your research.
```
@InProceedings{Jia_2020_CVPR_SSDG,
    author = {Yunpei Jia and Jie Zhang and Shiguang Shan and Xilin Chen},
    title = {Single-Side Domain Generalization for Face Anti-Spoofing},
    booktitle = {Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020}
}
```




