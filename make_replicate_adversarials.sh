#!/bin/bash
declare -a eps=("0.45" "0.55" "0.6") 
declare -a mode=("deep_fool")
for e in "${eps[@]}"
do
        for m in "${mode[@]}"
	do
		cmd="python evaluation.py -r 1 -m ${m} --eps=${e}"
		echo $cmd
		eval $cmd
	done
done

