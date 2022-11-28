#/usr/bin/env bash

nvflare simulator -w ./$1 -n 2 -t 1 -gpu 0,1 . > log.txt

## default example
## nvflare simulator -w workspace/ -n 2 -t 2 .
