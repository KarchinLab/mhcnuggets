#!/bin/bash
set -e

#Converting boolean environment variables to lower case string for future comparison
EMBED_PEPTIDES=$( echo "$MHC_EMBED_PEPTIDES" | tr '[:upper:]'  '[:lower:]' )
BINARY_PREDICTIONS=$( echo "$MHC_BINARY_PREDICTIONS" | tr '[:upper:]'  '[:lower:]' )
BA_MODELS=$( echo "$MHC_BA_MODELS" | tr '[:upper:]'  '[:lower:]' )

#Setting filepath
PEPTIDES_FILEPATH="/mhcnuggets/mount/$MHC_PEPTIDES_FILENAME"

#Handling Environment Variable Defaults
if [ -z ${MHC_OUTPUT+x} ]; then MHC_OUTPUT=""; else MHC_OUTPUT="-o $MHC_OUTPUT"; fi
if [ -z ${MHC_MODEL+x} ]; then MHC_MODEL="lstm"; fi
if [ -z ${MHC_MODEL_WEIGHTS_PATH+x} ]; then MHC_MODEL_WEIGHTS_PATH="saves/production/"; fi
if [ -z ${MHC_PICKLE_PATH+x} ]; then MHC_PICKLE_PATH="data/production/examples_per_allele.pkl"; fi
if [ -z ${MHC_MASS_SPEC+x} ]; then MHC_MASS_SPEC=false; fi
if [ -z ${MHC_IC50_THRESHOLD+x} ]; then MHC_IC50_THRESHOLD=500; fi
if [ -z ${MHC_MAX_IC50+x} ]; then MHC_MAX_IC50=50000; fi

#Handling Environment Variable Booleans
if [[ $EMBED_PEPTIDES == "true" ]]; then EMBED_PEPTIDES="-q"; else EMBED_PEPTIDES=""; fi
if [[ $BINARY_PREDICTIONS == "true" ]]; then BINARY_PREDICTIONS="-B"; else BINARY_PREDICTIONS=""; fi
if [[ $BA_MODELS == "true" ]]; then BA_MODELS="-M"; else MHC_BA_MODELS=""; fi

#If using command line arguments (based on if MHC_CLASS is set or not)
if [[ -z "$MHC_CLASS" ]]; then
  exec python /usr/local/lib/python3.7/site-packages/mhcnuggets/src/predict.py "$@"

#Else, assuming that environment variables are in use.
else
  exec python /usr/local/lib/python3.7/site-packages/mhcnuggets/src/predict.py -c $MHC_CLASS -p $PEPTIDES_FILEPATH -a $MHC_ALLELE -m $MHC_MODEL -s $MHC_MODEL_WEIGHTS_PATH -k $MHC_PICKLE_PATH -e $MHC_MASS_SPEC -l $MHC_IC50_THRESHOLD -x $MHC_MAX_IC50 $EMBED_PEPTIDES $BINARY_PREDICTIONS $BA_MODELS $MHC_OUTPUT
fi