#!/bin/bash

source ~/.profile;

array=( 7new 8 9 6 5 4 3 2 1 0 )
for NUM in "${array[@]}"
do
	time th train.lua -input_h5 data/c$NUM.h5 -input_json data/c$NUM.json -checkpoint_name cv/c$NUM/c$NUM -checkpoint_every 10000> output_ionic_c$NUM;
done
