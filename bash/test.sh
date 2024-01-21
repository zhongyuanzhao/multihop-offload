#!/bin/bash

# size=200
# seed=100
size=100
seed=500

gtype="ba"
datapath="../data/aco_data_${gtype}_${size}"
# for gtype in 'ba' 'er' 'grp' 'ws'; do
for scale in 0.15 ; do
	echo "submit task ${gtype}";
	python3.9 src/AdHoc_test.py --datapath=${datapath} --arrival_scale=${scale} --training_set=BAT800 ;
done


echo "Done"