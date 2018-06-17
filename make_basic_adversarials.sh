#!/bin/bash
declare -a eps=("0" "0.005" "0.01" "0.015" "0.02" "0.025" "0.04" "0.06" "0.08" "0.1") 
declare -a mode=("fgsm" "ifgsm")
for e in "${eps[@]}"
do
        for m in "${mode[@]}"
	do
		cmd="python evaluation.py -m ${m} --eps=${e}"
		echo $cmd
		eval $cmd
	done
done

