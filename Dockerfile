FROM python:3.6

WORKDIR /nbc
ADD . /nbc

RUN make -C /nbc install
