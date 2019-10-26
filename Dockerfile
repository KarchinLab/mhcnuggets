FROM python:3

COPY . mhcnuggets

WORKDIR /mhcnuggets

RUN python3 setup.py sdist bdist_wheel

RUN pip install dist/mhcnuggets-2.3.2.tar.gz

ENTRYPOINT [ "./docker-entrypoint.sh" ]