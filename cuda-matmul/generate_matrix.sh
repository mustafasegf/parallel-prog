#!/usr/bin/env bash
echo "generate test-matrices with python3 if no test data found"
echo

MATRIX_SIZES=(
	2
	128
	256
	512
	1024
	2048
	4096
	8192
	# 16384
)
for SZ in "${MATRIX_SIZES[@]}"; do
	FILE1="data/mat_${SZ}x${SZ}.txt"
	if [ ! -f "$FILE1" ]; then
		echo "generate ${SZ}x${SZ} matrix..."
		python3 random_float_matrix.py "$SZ" "$SZ" >"$FILE1"
	fi

	FILE2="data/mat_${SZ}x${SZ}b.txt"
	if [ ! -f "$FILE2" ]; then
		echo "generate ${SZ}x${SZ} matrix..."
		python3 random_float_matrix.py "$SZ" "$SZ" >"$FILE2"
	fi
done
