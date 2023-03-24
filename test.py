import torch
print(torch.cuda.is_available())  #返回True则说明已经安装了cuda
from torch.backends import  cudnn 
print(cudnn.is_available())  #返回True则说明已经安装了cuDNN conda activate Yolo_dnf
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())
import cv2 as cv
print(cv.__version__)

# 注：
# * 不要开vpn
# * 用anaconda 运行命令
# * 安装包用conda 来装，快
# * 安装anaconda 时 记得设置镜像源
# cd/d E:\Web\yolo\yolov5Master
# 参考连接
# https://www.cnblogs.com/beifangcc/p/16213038.html
# https://pytorch.org/


# n卡 安装 conda
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# 训练
# python train.py --data hat.yaml --weights yolov5s.pt --cfg models/yolov5s_hat.yaml --epochs 20

# 测试训练
# python train.py --img 640 --batch 1 --epochs 3 --data coco128.yaml --weights yolov5s.pt
# python train.py --img 640 --batch 1 --epochs 2 --data hat.yaml --cfg models/yolov5s_hat.yaml --weights yolov5s.pt --device cpu

# cd/d D:\webDeme\yolo\demo\yolov5-master
# conda activate Yolo_dnf

# python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt

# 测试
# python detect.py --weights weights/best.pt --img 640 --conf 0.55 --source VOCdevkit/images/QQ截图20230221220806.png











# 更新 pip （一般不用。用conda）
# python -m pip install --upgrade pip

# 缺少yaml
# pip install PyYAML

# 缺少tqdm
# conda install tqdm

# No module named 'cv2'
# conda install opencv

# No module named 'pandas'
# conda install pandas

# No module named 'IPython'
# conda install ipython

# No module named 'psutil'
# conda install psutil

# No module named 'matplotlib'
# conda install matplotlib

# No module named 'seaborn'
# conda install seaborn

# No module named 'tensorboard'
# conda install tb-nightly
# conda install tensorboard

# 报错
# *** WARNING: Ignore distutils configs in setup.cfg due to encoding errors.
# https://blog.csdn.net/weixin_37989267/article/details/128326603



