#!/bin/bash
declare -a eps=("0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007") 
for i in "${eps[@]}"
do
	cmd="python models.py -a fgsm --eps=${i}"
	echo $cmd
	#eval $cmd
done

