# Visual Relation Grounding in Videos

This is the pytorch implementation of our paper at ECCV2020 (Spotlight). 
![teaser](https://github.com/doc-doc/vRGV/blob/master/introduction.png)
The repository mainly includes 3 parts: (1) Extract RoI feature; (2) Train and inference; and (3) Generate relation-aware trajectories.
![](https://github.com/doc-doc/vRGV/blob/master/model.png)

## Environment

Anaconda 3, python 3.6.5, pytorch 0.4.1 and cuda >= 9.0. For others libs, please refer to the file requirements.txt.

## Install
Please create an envs for this project using anaconda3 (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n envname python=3.6.5 #create
>conda activate envname #enter
>pip install -r requirements.txt #blindly install the provided libs
>sh vRGV/lib/make.sh #install the needed libs for detection
```
## Data Preparation
Please download the data [here](https://drive.google.com/file/d/1__asF5lrOj7091TyLQkkebiRYLbrqYBA/view?usp=sharing). The folder [ground_data] should be at the same directory as vRGV [this project]. Please merge the downloaded vRGV folder with this repo. 

Please download the raw videos [here](https://xdshang.github.io/docs/imagenet-vidvrd.html), and extract them into ground_data/vidvrd/JPEGImages/. The directory should be like: JPEGImages/ILSVRC2015_train_xxx/000000.JPEG

## Usage
Feature Extraction (need about 100G storage!)
```
>./detection.sh
```
Train
```
>./ground.sh 0 --mode train # To train the model with GPU id 0
```
Inference
```
>./ground.sh 0 --mode val # To output the relation-aware spatio-temporal attention
>python generate_track_lick.py # To generate relation-aware trajectories with Viterbi algorithm
>python eval_ground.py # To evaluate the performance
```
Visualization
```
>python visualize.py
```
## Note  

If you find the codes useful in your research, please kindly cite:

```
@inproceedings{junbin2020visual,
  title={Visual Relation Grounding in Videos},
  author={Junbin, Xiao and Xindi, Shang and Xun, Yang and Sheng, Tang and Tat-Seng, Chua},
  booktitle={Proceedings of the 16th European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## License

NUS © [NExT++](https://nextcenter.org/)
