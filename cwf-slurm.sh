#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=1	# number of processors per task
#SBATCH -J "cwf"   # job name

## /SBATCH -p general # partition (queue)
#SBATCH -o cwf.%N.%j.out # STDOUT
#SBATCH -e cwf.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
import PyHipp as pyh; \
import DataProcessingTools as DPT; \
wfall = DPT.objects.processDirs(dirs=None, exclude=['*eye*', '*mountains*'], objtype=pyh.Waveform); \
wfall.save();”
