FROM python:3.7.3-stretch

RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl numpy

RUN apt-get update && apt-get install -y patchelf

RUN mkdir /optorch
WORKDIR /optorch
COPY dist/optorch-0.0.2-cp37-cp37m-manylinux1_x86_64.whl .
RUN pip install optorch-0.0.2-cp37-cp37m-manylinux1_x86_64.whl

COPY examples/rosenbrock.py .

CMD ["echo", "hi"]
