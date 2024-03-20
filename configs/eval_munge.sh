#!/bin/bash

for i in $(ls *32i_eval.yaml); do
	embdir=$(echo $i | sed 's/_eval.yaml//g')
	sed 's/12e/32e/g' -i $i
	sed 's/mlp/linear/gI' -i $i
	sed "s/embedding_dir:/embedding_dir: \"$embdir\" #/g" -i $i
	
done
