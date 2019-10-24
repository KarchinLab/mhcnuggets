FROM python:3

COPY . mhcnuggets

WORKDIR /mhcnuggets

RUN python3 setup.py sdist bdist_wheel

RUN pip install dist/mhcnuggets-2.3.2.tar.gz

#ENV MHC_MODEL=lstm \
#	MHC_MODEL_WEIGHTS_PATH='saves/production/' \
#	MHC_PICKLE_PATH='data/production/examples_per_allele.pkl' \
#	MHC_MASS_SPEC=false \
#	MHC_OUTPUT='/mhcnuggets/mount/output.txt' \
#	MHC_IC50_THRESHOLD=500 \
#	MHC_MAX_IC50=50000 \
#	MHC_EMBED_PEPTIDES=false \
#	MHC_BINARY_PREDICTIONS=false \
#	MHC_BA_MODELS=false \
#	MHC_PREDICT_PATH='/usr/local/lib/python3.7/site-packages/mhcnuggets/src/predict.py' \
#	MHC_PEPTIDES_PATH='/mhcnuggets/mount/'

#CMD python $MHC_PREDICT_PATH -c $MHC_CLASS -p $MHC_PEPTIDES_PATH$MHC_PEPTIDES_FILENAME -a $MHC_ALLELE -o $MHC_OUTPUT -m $MHC_MODEL -s $MHC_MODEL_WEIGHTS_PATH -k $MHC_PICKLE_PATH -e $MHC_MASS_SPEC -l $MHC_IC50_THRESHOLD -x $MHC_MAX_IC50 -q $MHC_EMBED_PEPTIDES -B $MHC_BINARY_PREDICTIONS -M $MHC_BA_MODELS

ENTRYPOINT [ "./docker-entrypoint.sh" ]