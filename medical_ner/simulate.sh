#/usr/bin/env bash

nvflare simulator -w ./$1 -n 2 -t 8 -gpu 0,1 . > log.txt