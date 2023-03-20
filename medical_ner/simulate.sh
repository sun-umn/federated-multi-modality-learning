#/usr/bin/env bash

# rm -rf ./$1
# nvflare simulator -w ./$1 -n 2 -t 1 -gpu 0,1 . > log.txt

<<<<<<< HEAD
readonly workspace=workspace_2018_n2c2
=======
readonly workspace=workspace_2018_n2c2_trail2
>>>>>>> v1.0


rm -rf ./$workspace
nvflare simulator -w ./$workspace -n 2 -t 1 -gpu 0,1 . > log.txt

# CUDA_VISIBLE_DEVICES=$2 nvflare simulator -w ./$1 -n 2 -t 1 . > log.txt

## default example
## nvflare simulator -w workspace/ -n 2 -t 2 .
        