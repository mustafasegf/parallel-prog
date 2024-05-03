#!/usr/bin/env bash
make

GRID=(
	1
	2
	8
	10
)

BLOCK=(
	1
	2
	7
	8
	10
)

for G in "${GRID[@]}"; do
	for B in "${BLOCK[@]}"; do
		echo
		echo "* * * * * grid:$G block:$B"
		./minimal "$G" "$B"
	done
done
