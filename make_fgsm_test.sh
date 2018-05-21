#!/bin/bash
declare -a eps=("0.0" "0.01") 
for i in "${eps[@]}"
do
	cmd="python models.py -a fgsm --eps=${i}"
	echo $cmd
	eval $cmd
done

