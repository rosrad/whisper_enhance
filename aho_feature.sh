#!/bin/sh
# ahocoder feature extraction Set low and high f0 range 
mkdir -p feat 
mkdir -p convert
for entry in `ls data/*.wav`; do
#echo $entry
fname=`basename $entry .wav`
echo $entry $fname
# feaure extraction : f0, mcc, fv
sox -r 16000 -b 16 -c 1 $entry convert/$fname.wav 
./ahocoder16_64 convert/$fname.wav feat/$fname.f0 feat/$fname.mcc feat/$fname.fv
done
