#!/bin/bash
set -euox pipefail

cd $(dirname $0)
VERSION=$(cat VERSION)

docker build -f docker/Dockerfile.build -t optorch-build --build-arg VERSION=$VERSION .
container_id=$(docker create optorch-build)

mkdir -p dist
docker cp $container_id:/output/optorch-${VERSION}-cp37-cp37m-manylinux1_x86_64.whl dist
docker rm -v $container_id
