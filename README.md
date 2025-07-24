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
In the Dataset file, it contains folder for each video ID. It also contain 'kinetic_curves.xlsx' files which includes all the detailed information for each video ID, such as diameter, start time, and end time and so on. 

To help you better understand the dataset, there are few points need to be mentioned:
First, there are video ID: 101, 102, 103, 201, 203, 301, and 302. There is a relationship between video ID and filament ID in the paper. Training dataset is video ID of 102_R1, 102_R2, and 302 which is related to filament ID 1, 2 and 3. Validation dataset is video ID of 103 and 301 which is related to filament ID 4 and 5. Test dataset video ID of 201 and 203 which is related to filament ID 6 and 7. For the image quality concern, we did not use 101 dataset. 

