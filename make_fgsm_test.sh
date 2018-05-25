#!/bin/bash
#declare -a eps=("0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007") 
declare -a eps=("0" "0.0015" "0.0025" "0.0035") 
for i in "${eps[@]}"
do
	cmd="python evaluation.py -m fgsm --eps=${i}"
	echo $cmd
	eval $cmd
done

