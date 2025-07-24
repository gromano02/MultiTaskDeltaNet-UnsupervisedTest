# MultiTaskDeltaNe
Change Detection-based Image Segmentation for operando ETEM with Application to Carbon Gasification Kinetics
## Network
<img src="https://github.com/niuyushuo/MultiTaskDeltaNet/blob/7f950176eab5de6c4512032d140f1fda05d82265/Results/image_github/Model_arch.png" width="500" height="400">

## Installition
Create a conda environment:
```
conda create -n python3.10_pytorch2.0 python=3.10
conda activate python3.10_pytorch2.0
```

Install pytorch based on your cuda version:
```
nvidia-smi
```

<img src='https://github.com/niuyushuo/MultiTaskDeltaNet/blob/cf35d0590df76a88bdd18f4d15cc7f719b90e6bd/Results/image_github/smi.png' width="500" height="400">

<img src='https://github.com/niuyushuo/MultiTaskDeltaNet/blob/cf35d0590df76a88bdd18f4d15cc7f719b90e6bd/Results/image_github/pytorch.png' width="400" height="200">

Install pytorch:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the rest packages:
```
conda install matplotlib
conda install esri::einops
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install anaconda::seaborn
conda install anaconda::openpyxl
pip install "ray[tune]"  #### if hyper-parameter searching needed (find best hyper-parameter for your case)

```

## Dataset

