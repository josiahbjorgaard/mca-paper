#!/bin/bash

for i in $(ls *bm*4i.yaml); do
	sed "s/output_dir:/output_dir: \"$(echo $i | sed 's/\.yaml//g')\"  #/g" -i  $i
done
