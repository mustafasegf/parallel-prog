#!/usr/bin/env bash
./generate_matrix.sh

echo "compile..."
echo
make -j 8
echo
echo "calculate..."
echo
MATRIX_SIZES=(
	128
	256
	512
	1024
	# 2048
	# 4096
	# 8192
	# 16384
)

rm -f time_old.txt
rm -f table.txt
# rm -f time_summary_old.txt

[ -f "time.txt" ] && cp time.txt time_old.txt
[ -f "table.txt" ] && cp table.txt table_old.txt
# [ -f "time_summary.txt" ] && cp time_summary.txt time_summary_old.txt

echo -n >time.txt
echo -n >table.txt
# echo -n >time_summary.txt

echo
echo "* * * * * * * sequential"
for SZ in "${MATRIX_SIZES[@]}"; do
	./sequential "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done

echo
echo "* * * * * * * tiling"
for SZ in "${MATRIX_SIZES[@]}"; do
	./tiling "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done

echo
echo "* * * * * * * avx"
for SZ in "${MATRIX_SIZES[@]}"; do
	./avx "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done

echo
echo "* * * * * * * naive"
for SZ in "${MATRIX_SIZES[@]}"; do
	./naive "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done

echo
echo "* * * * * * * shared"
for SZ in "${MATRIX_SIZES[@]}"; do
	./shared "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done
echo

echo "* * * * * * * cublas"
for SZ in "${MATRIX_SIZES[@]}"; do
	./cublas "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt" | tee -a time.txt
done
