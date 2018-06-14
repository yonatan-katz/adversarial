#!/bin/bash

for index in {0..999};
do
	cmd="python models.py -m carlini --index ${index}"
	echo ${cmd}
        eval ${cmd}
done
