#!/usr/bin/env bash
./generate_matrix.sh

echo "compile..."
echo
make -j
echo
echo "calculate..."
echo

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

# GRID=

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
fi

if [ -z "$1" ]; then
	echo
	echo "* * * * * * * tiling"
	for SZ in "${MATRIX_SIZES[@]}"; do
		./tiling "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timecpu.txt
	done
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
fi

echo
echo "* * * * * * * naive"
for SZ in "${MATRIX_SIZES[@]}"; do
	./naive "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timegpu.txt
done

echo
echo "* * * * * * * shared"
for SZ in "${MATRIX_SIZES[@]}"; do
	./shared "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timegpu.txt
done
echo

echo "* * * * * * * cublas"
for SZ in "${MATRIX_SIZES[@]}"; do
	./cublas "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a timegpu.txt
done
