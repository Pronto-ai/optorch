#!/bin/bash

cd docsrc
make html
rm -rf ../docs
cp -r _build/html ../docs
