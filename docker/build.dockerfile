FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y git cmake libsuitesparse-dev
RUN apt-get install -y build-essential
RUN apt-get install -y libeigen3-dev

RUN mkdir /ceres
WORKDIR /ceres
RUN git clone https://github.com/ceres-solver/ceres-solver
# RUN echo "set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0\")" >>ceres-solver/CMakeLists.txt
RUN sed -i '34iadd_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)' ceres-solver/CMakeLists.txt

RUN mkdir -p /ceres/ceres-solver/build
WORKDIR /ceres/ceres-solver/build
RUN cmake -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=ON -DCXSPARSE=OFF -DLAPACK=OFF ..
RUN VERBOSE=1 make

RUN mkdir /optorch
WORKDIR /optorch

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7 curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl

RUN mkdir /ep
WORKDIR /ep
RUN git clone https://github.com/yse/easy_profiler
RUN mkdir /ep/easy_profiler/build
WORKDIR /ep/easy_profiler/build
RUN cmake ..
RUN make install

RUN apt-get install -y libgoogle-glog-dev

WORKDIR /optorch
COPY . .
RUN rm -rf build && mkdir build
RUN python3.7 setup.py build_ext
RUN python3.7 setup.py bdist_wheel

RUN mkdir /output
RUN cp dist/optorch-0.0.3-cp37-cp37m-manylinux1_x86_64.whl /output/optorch-0.0.3-cp37-cp37m-manylinux1_x86_64.whl
