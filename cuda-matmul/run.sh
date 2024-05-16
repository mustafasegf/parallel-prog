#!/usr/bin/env bash
./generate_matrix.sh

make -j

MATRIX_SIZES=(
	128
	256
	512
	1024
	2048
	# 4096
	# 8192
	# 16384
)

GRID_BLOCK=(
	"1 1"
	"2 2"
	"4 4"
	"8 8"
	"16 16"
	"32 32"
	# "64 32"
	# "128 32"
	# "256 32"
	# "512 32"
)

# if 1nd is empty, skip gpu table
if [ -z "$1" ]; then
	echo "skip gpu table"
	GRID_BLOCK=("0 0")
fi
rm -f timecpu_old.txt
rm -f timegpu_old.txt
# rm -f table_old.csv

[ -f "timecpu.txt" ] && cp timecpu.txt timecpu_old.txt
[ -f "timegpu.txt" ] && cp timegpu.txt timegpu_old.txt
# [ -f "table.csv" ] && cp table.csv table_old.csv

echo -n >timecpu.txt
echo -n >timegpu.txt
# echo -n >table.csv

# if 2nd is not empty, skip cpu
if [ -z "$2" ]; then
	echo
	echo "* * * * * * * sequential"
	for SZ in "${MATRIX_SIZES[@]}"; do
		./sequential "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timecpu.txt
	done
else
	echo "skip cpu"
fi

if [ -z "$1" ]; then
	echo
	echo "* * * * * * * tiling"
	for SZ in "${MATRIX_SIZES[@]}"; do
		./tiling "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timecpu.txt
	done
else
	echo "skip cpu tiling"
fi

if [ -z "$1" ]; then
	# check if avx2 is available
	if grep -q avx2 /proc/cpuinfo; then
		echo
		echo "* * * * * * * avx2"
		for SZ in "${MATRIX_SIZES[@]}"; do
			./avx2 "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timecpu.txt
		done
	fi
else
	echo "skip cpu avx2"
fi

if [ -z "$1" ]; then
	# check if avx512 is available
	if grep -q avx512 /proc/cpuinfo; then
		echo
		echo "* * * * * * * avx512"
		for SZ in "${MATRIX_SIZES[@]}"; do
			./avx512 "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timecpu.txt
		done
	fi
else
	echo "skip cpu avx512"
fi

echo
echo "* * * * * * * naive"
for GB in "${GRID_BLOCK[@]}"; do
	GRID=${GB% *}
	BLOCK=${GB#* }
	echo
	echo "GRID: $GRID, BLOCK: $BLOCK"
	for SZ in "${MATRIX_SIZES[@]}"; do
		./naive "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" "$GRID" "$BLOCK" | tee -a timegpu.txt
	done
done

echo
echo "* * * * * * * shared"
for GB in "${GRID_BLOCK[@]}"; do
	GRID=${GB% *}
	BLOCK=${GB#* }
	echo
	echo "GRID: $GRID, BLOCK: $BLOCK"
	for SZ in "${MATRIX_SIZES[@]}"; do
		./shared "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" "$GRID" "$BLOCK" | tee -a timegpu.txt
	done
done

echo
echo "* * * * * * * cublas"
for SZ in "${MATRIX_SIZES[@]}"; do
	./cublas "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" 1 1 | tee -a timegpu.txt
done
