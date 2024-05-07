#!/usr/bin/env bash
make

echo "* * * * * grid:1 block:1"
./minimal 1 1

echo
echo "* * * * * grid:2 block:2"
./minimal 2 2

echo
echo "* * * * * grid:8 block:7"
./minimal 8 7

echo
echo "* * * * * grid:8 block:8"
./minimal 8 8

echo
echo "* * * * * grid:16 block:10"
./minimal 16 10
