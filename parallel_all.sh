#!/bin/bash
# Run model_PreSigmoidal.py with 100 trials for each value of GAIN

NCPUS=64

doit()
{
	GAIN=$1
	REPETITION=$2
	# Pad label to 3 digits:
	LABEL=$(printf "%03d" $((${PARALLEL_SEQ}-1)))
	echo "Started simulation $LABEL (repetition $REPETITION for GAIN=$GAIN)"
	python model_PreSigmoidal.py $GAIN $LABEL
}

export -f doit
parallel --no-notice --nice 10 -j $NCPUS doit {1} {2} :::: <(cat points.txt) :::: <(seq 1 100)
