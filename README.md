# [Visual Relation Grounding in Videos](https://arxiv.org/pdf/2007.08814.pdf)

This is the pytorch implementation of our work at ECCV2020 (Spotlight). 
![teaser](https://github.com/doc-doc/vRGV/blob/master/introduction.png)
The repository mainly includes 3 parts: (1) Extract RoI feature; (2) Train and inference; and (3) Generate relation-aware trajectories.
![](https://github.com/doc-doc/vRGV/blob/master/model.png)
## Notes
Fix issue on unstable result [2021/10/07].

## Environment

Anaconda 3, python 3.6.5, pytorch 0.4.1 (Higher version is OK once feature is ready) and cuda >= 9.0. For others libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda3 (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n envname python=3.6.5 # Create
>conda activate envname # Enter
>pip install -r requirements.txt # Install the provided libs
>sh vRGV/lib/make.sh # Set the environment for detection, make sure you have nvcc
```
## Data Preparation
Please download the data [here](https://drive.google.com/file/d/1qNJ3jBPPoi0BPkvLqooS66czvCxsib1M/view?usp=sharing). The folder ```ground_data``` should be at the same directory as ```vRGV```. Please merge the downloaded vRGV folder with this repo. 

Please download the videos [here](https://xdshang.github.io/docs/imagenet-vidvrd.html) and extract the frames into ground_data. 
The directory should be like: ground_data/vidvrd/JPEGImages/ILSVRC2015_train_xxx/000000.JPEG.

## Usage
Feature Extraction. (need about 100G storage! Because I dumped all the detected bboxes along with their features. It can be greatly reduced by changing detect_frame.py to return the top-40 bboxes and save them with .npz file.)
```
./detection.sh 0 val #(or train)
```
Sample video features:
```
cd tools
python sample_video_feature.py
```
Test. You can use our provided model to verify the feature and environment:
```
./ground.sh 0 val # Output the relation-aware spatio-temporal attention
python generate_track_link.py # Generate relation-aware trajectories with Viterbi algorithm.
python eval_ground.py # Evaluate the performance
```
You will get accuracy Acc_R: 24.58%.

Train. If you want to train the model from scratch. Please apply a two-stage training scheme: 1) train a basic model without relation attendance, and 2) load the reconstruction part of the pre-trained model to learn the whole model (with the same lr_rate). For implementation, please turn off/on ```[pretrain] in line 52 of ground.py```, and switch between ```line 6 & 7 in ground_relation.py```  for 1st & 2nd stage training respectively. Also, you need to change the model files in ```line 69 & 70 of ground_relation.py``` to the best model obtained at the first stage for 2nd-stage training. 
```
./ground.sh 0 train # Train the model with GPU id 0
```
The results maybe slightly different (+/-0.5%), For comparison, please follow the results reported in our paper.
## Result Visualization
|Query| bicycle-jump_beneath-person       | person-feed-elephant          | person-stand_above-bicycle       | dog-watch-turtle|
|:---| --------------------------------- | ----------------------------- | ---------------------------------------- | ---------------------------------------- | 
|Result| ![](https://media.giphy.com/media/htciIcJZ2q7pb06zoI/giphy.gif) | ![](https://media.giphy.com/media/dX34r2BJNjVCNCuFNy/giphy.gif)   | ![](https://media.giphy.com/media/ln7xmvrkjcX47W9Kax/giphy.gif)|![](https://media.giphy.com/media/h5uiVR9ukJLVRgT9yC/giphy.gif)|
|Query| person-ride-horse       | person-ride-bicycle          |   person-drive-car     |  bicycle-move_toward-car|
|Result| ![](https://media.giphy.com/media/J5jSa7lJxwFXorWYbx/giphy.gif) | ![](https://media.giphy.com/media/lSsztYWamp6gLfHSfg/giphy.gif)   | ![](https://media.giphy.com/media/S5Kp8KaApxrazkVmcd/giphy.gif)|![](https://media.giphy.com/media/ZE4vFIjfm1BHXP7w0R/giphy.gif)|

## Citation

```
@inproceedings{xiao2020visual,
  title={Visual Relation Grounding in Videos},
  author={Xiao, Junbin and Shang, Xindi and Yang, Xun and Tang, Sheng and Chua, Tat-Seng},
  booktitle={European Conference on Computer Vision},
  pages={447--464},
  year={2020},
  organization={Springer}
}
```

## License

NUS © [NExT++](https://nextcenter.org/)
