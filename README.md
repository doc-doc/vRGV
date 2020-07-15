# vRGV
Pytorch Implementation of our paper at ECCV2020: Visual Relation Grounding in Videos.
# Environment
Python 3.6.5  
Pytorch 0.4.1  
For other libs, please refer to the requirements.txt.
# Usage
Inference:  
step1: Output attention.  
./ground.sh 0 --mode val  
step2: Generate relation-aware trajectories.  
python generate_link.py  
step3: Evaluate  
python eval_ground.py  
Train:  
./ground.sh 0 --mode train (0 is GPU id)  
