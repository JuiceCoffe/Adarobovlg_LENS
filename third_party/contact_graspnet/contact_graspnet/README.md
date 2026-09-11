# GraspClutter6D Contact-GraspNet

Vendored from [gc6d-contact-graspnet](https://github.com/SeungBack/gc6d-contact-graspnet),
which is based on
[contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch).

## Installation
Tested on `python == 3.10`, `torch ==2.5.1`, `CUDA 11.8`.

```
conda create -n cgnet python=3.10
conda activate cgnet
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install ext/graspnetAPI
pip install graspclutter6dAPI
```


## Evaluation

If you want to visualize the results use `--vis` flag.

Test: GraspClutter6D, Train: GraspNet-1B
```
python test.py \
    cfgs/GC6D/Contact_GraspNet.yaml \
    data/models/gc6d.pth \
    --camera realsense-d435 \
    --dataset graspclutter6d \
    --grasp_root "$GC6D_ROOT" \
    --split test 
```

Test: GraspClutter6D, Train: GraspNet-1B
```
python test.py \
    cfgs/G1B/Contact_GraspNet.yaml \
    data/models/g1b.pth \
    --camera realsense-d435 \
    --dataset graspclutter6d \
    --grasp_root "$GC6D_ROOT" \
    --split test 
```
