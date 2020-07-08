#!/bin/sh
# ahocoder feature extraction Set low and high f0 range 

mkdir -p DNNGAN_wav
mkdir -p generate
for entry in `ls DNNGAN/*.mcc`; do
#echo $entry
fname=`basename $entry .mcc`
echo $entry $fname
./ahodecoder16_64  feat/$fname.f0 DNNGAN/$fname.mcc feat/$fname.fv DNNGAN_wav/$fname.wav
done
