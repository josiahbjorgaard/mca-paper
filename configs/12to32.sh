#!/bin/bash

for yamlfile in $(ls *bm*12i*); do
	newfile=$(echo $yamlfile | sed 's/12i/4i/g')
	echo $newfile
	cp $yamlfile $newfile
	sed 's/\/12/\/3/g' -i $newfile
done
