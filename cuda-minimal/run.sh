#!/usr/bin/env bash
# make

# GRID=(
# 	1
# 	2
# 	8
# 	10
#   16
# )
#
# BLOCK=(
# 	1
# 	2
# 	7
# 	8
# 	10
#   16
# )
#
# for G in "${GRID[@]}"; do
# 	for B in "${BLOCK[@]}"; do
# 		echo
# 		echo "* * * * * grid:$G block:$B"
# 		./minimal "$G" "$B"
# 	done
# done

GRID_BLOCK=(
	"1 1"
	"2 2"
	"8 7"
	"8 8"
	"16 10"
)

for i in "${GRID_BLOCK[@]}"; do
	G=${i% *}
	B=${i#* }
	if [ "$i" != "1 1" ]; then
		echo
	fi
	echo "* * * * * grid:$G block:$B"
	./minimal "$i"
done
