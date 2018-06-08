#!/bin/bash
declare -a eps=("0.01") 
#declare -a mode=("deep_fool")
declare -a mode=("carlini_wagner")
for e in "${eps[@]}"
do
        for m in "${mode[@]}"
	do
		cmd="python evaluation.py -m ${m} --eps=${e}"
		echo $cmd
		eval $cmd
	done
done

