#!/bin/bash
set -euox pipefail

cd $(dirname $0)
VERSION=$(cat VERSION)

docker build -f docker/Dockerfile.test -t optorch-test --build-arg VERSION=$VERSION .
docker run --rm optorch-test
