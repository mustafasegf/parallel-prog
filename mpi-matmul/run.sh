#!/usr/bin/env bash
./generate_matrix.sh

echo "compile..."
echo
make
echo
echo "calculate..."
echo
NPS=(
	1
	2
	4
	8
)
MATRIX_SIZES=(
	128
	256
	512
	1024
	2048
)

echo -n > time.txt

for SZ in "${MATRIX_SIZES[@]}"; do
	echo "* * * * * * * ${SZ}x${SZ} Matrix"
	for PROC in "${NPS[@]}"; do
		mpirun -np "${PROC}" main "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt"
	done
  echo "" >> time.txt
done

for SZ in "${MATRIX_SIZES[@]}"; do
	echo "* * * * * * * ${SZ}x1 Matrix"
	for PROC in "${NPS[@]}"; do
		mpirun -np "${PROC}" main "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x1.txt"
	done
  echo "" >> time.txt
done
