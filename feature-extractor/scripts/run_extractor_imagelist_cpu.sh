#!/bin/bash
#PBS -N Features_extraction
#PBS -q global
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=40gb
#PBS -l walltime=24:00:00

DATADIR=/storage/plzen1/home/anopheles

echo Scratch dir: $SCRATCHDIR
echo Working dir: `pwd`

IMAGELIST=#LST#
BASEDIR=#EXT_ESCAPED#

echo Image list: $IMAGELIST

cd $SCRATCHDIR

cp $IMAGELIST ./imagelist.txt

mkdir ./img
for file in $(cat ./imagelist.txt);
do
	cp $DATADIR/img/$file ./img &>/dev/null &
done

cp -r $DATADIR/feature-extractor ./feature-extractor

mkdir output

sed -i $'s|#EXTRACT#|#EXTRACTOR#|g' ./feature-extractor/scripts/run_extractor_singularity_cpu.sh

singularity run --bind $SCRATCHDIR:/scratch --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.12-tf2-py3.SIF /scratch/feature-extractor/scripts/run_extractor_singularity_cpu.sh

mkdir $DATADIR/extracted_clean_features/$BASEDIR 2>/dev/null

cp ./imagelist.txt ./output
mv ./output/* $DATADIR/extracted_clean_features/$BASEDIR

rm -rf *
