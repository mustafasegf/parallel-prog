#!/usr/bin/env bash
echo "1000 1000"
./incrementArray 1000 1000

echo "1 1024"
./incrementArray 1 1024

echo "1 1025"
./incrementArray 1 1025

echo "2147483647 1"
./incrementArray 2147483647 1

echo "2147483648 1"
./incrementArray 2147483648 1
