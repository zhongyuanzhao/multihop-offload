#!/bin/bash

size=200
seed=100
# size=100
# seed=500

training_set="BAT800"
T=800
# for gtype in 'ba' 'er' 'grp' 'ws' 'poisson'; do
for gtype in 'ba' ; do
	datapath="../data/aco_data_${gtype}_${size}"
	echo "submit task ${gtype}";
	python3.9 src/AdHoc_train.py --datapath=${datapath} --arrival_scale=0.15 --learning_rate=0.000001 --training_set=${training_set} --T=${T};
done


echo "Done"