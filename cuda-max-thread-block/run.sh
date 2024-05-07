#!/usr/bin/env bash
make

echo "1000 1000"
./incrementArray 1000 1000

echo
echo "1 1024"
./incrementArray 1 1024

echo
echo "1 1025"
./incrementArray 1 1025

echo
echo "2147483647 1"
./incrementArray 2147483647 1

echo
echo "2147483648 1"
./incrementArray 2147483648 1
