#!/bin/bash

# Training dataset
size=200
seed=100

# for gtype in 'ba' 'er' 'grp' 'ws'; do
for gtype in 'ba' ; do
	datapath="./data/aco_data_${gtype}_${size}"
	echo "submit task ${datapath} for training dataset";
	python3.9 src/data_generation_offloading.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed};
done

# Test dataset
size=100
seed=500

# for gtype in 'ba' 'er' 'grp' 'ws'; do
for gtype in 'ba' ; do
	datapath="./data/aco_data_${gtype}_${size}"
	echo "submit task ${datapath} for test dataset";
	python3.9 src/data_generation_offloading.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed};
done



echo "Done"