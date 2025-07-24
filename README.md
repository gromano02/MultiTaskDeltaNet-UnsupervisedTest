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
The dataset file contains a separate folder for each video ID and includes 'kinetic_curves.xlsx' files. These files provide detailed information for each video ID, such as diameter, start time, end time, and more.

To help you better understand the dataset, here are a few key points:

1. **Video IDs:** The available video IDs are 101, 102, 103, 201, 203, 301, and 302. There is a relationship between video IDs and filament IDs mentioned in the paper. The training dataset consists of video IDs 102_R1, 102_R2, and 302, which correspond to filament IDs 1, 2, and 3. The validation dataset includes video IDs 103 and 301, related to filament IDs 4 and 5. The test dataset contains video IDs 201 and 203, corresponding to filament IDs 6 and 7. For quality reasons, dataset 101 has not been used.

2. **Folder Structure:** Each video ID folder contains two subfolders: "original" and "results." 
   - The "original" folder holds all the original images without cropping. Within this folder, the "area1" subfolder contains images labeled A1, the "area2" subfolder contains images labeled A2, and the "img" subfolder has the original filament images. 
   - The "results" folder contains images after preprocessing, which will have filament images without overlapping structures. Similarly, the "area1," "area2," and "area1-2" subfolders contain the A1 labels, A2 labels, and the difference between A1 and A2 (A1 - A2), respectively. The "img" subfolder in the results folder also contains filament images. It’s important to note that the 102 folder is a special case; it only contains the original images, while the result images are located in the 102_R1 and 102_R2 folders.

3. **Testing the Code:** To test the code, please download the entire dataset, as the code is designed to read the dataset based on this specific structure.
