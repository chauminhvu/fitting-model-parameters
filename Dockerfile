FROM python:3.10.8

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip3 install --upgrade jaxlib==0.3.22 jaxopt==0.5.5 matplotlib pandas==1.5.1
