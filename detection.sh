#########################################################################
# File Name: detection.sh
# Author: Xiao Junbin
# mail: xiaojunbin@u.nus.edu
# Created Time: Wed 20 Nov 2019 11:10:44 AM +08
#########################################################################
#!/bin/bash
GPU=$1
MODE=$2
CUDA_VISIBLE_DEVICES=$GPU python detection.py --mode $MODE
