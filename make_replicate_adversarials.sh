#!/bin/bash
declare -a eps=("2.0") 
#declare -a mode=("deep_fool")
declare -a mode=("carlini_wagner")
for e in "${eps[@]}"
do
        for m in "${mode[@]}"
	do
		cmd="python evaluation.py -r 1 -m ${m} --eps=${e}"
		echo $cmd
		eval $cmd
	done
done

