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

for SZ in ${MATRIX_SIZES[@]}
do
    echo "* * * * * * * ${SZ}x${SZ} Matrix" | tee -a test-matrix.txt
    for PROC in ${NPS[@]}
    do
        cal_t=$((time mpirun -np ${PROC} main data/mat_${SZ}x${SZ}.txt data/mat_${SZ}x${SZ}b.txt)  2>&1 > /dev/null | grep real | awk '{print $2}')
        echo "with mpi(${PROC})     $cal_t"  | tee -a test-matrix.txt
        echo ""  | tee -a test-matrix.txt
    done
done

for SZ in ${MATRIX_SIZES[@]}
do
    echo "* * * * * * * ${SZ}x1 Matrix" | tee -a test-vector.txt
    for PROC in ${NPS[@]}
    do
        cal_t=$((time mpirun -np ${PROC} main data/mat_${SZ}x${SZ}.txt data/mat_${SZ}x1.txt)  2>&1 > /dev/null | grep real | awk '{print $2}')
        echo "with mpi(${PROC})     $cal_t" | tee -a test-vector.txt
        echo | tee -a test-vector.txt
    done
done

# for SZ in ${MATRIX_SIZES[@]}; do
# 	echo "* * * * * * * ${SZ}x${SZ} Matrix" | tee -a test-matrix.txt
# 	for PROC in ${NPS[@]}; do
# 		# Capture execution time while directing mpirun output to a separate file
# 		cal_t=$({ time mpirun -np ${PROC} main data/mat_${SZ}x${SZ}.txt data/mat_${SZ}x${SZ}b.txt >mpi_output_matrix_${SZ}_${PROC}.txt; } 3>&1 | grep real | awk '{print $2}')
# 		echo "with mpi(${PROC})     $cal_t" | tee -a test-matrix.txt
# 		echo "" | tee -a test-matrix.txt
#     sleep 1
# 	done
# done
#
# for SZ in ${MATRIX_SIZES[@]}; do
# 	echo "* * * * * * * ${SZ}x1 Matrix" | tee -a test-vector.txt
# 	for PROC in ${NPS[@]}; do
# 		# Capture execution time while directing mpirun output to a separate file
# 		cal_t=$({ time mpirun -np ${PROC} main data/mat_${SZ}x${SZ}.txt data/mat_${SZ}x1.txt >mpi_output_vector_${SZ}_${PROC}.txt; } 2>&1 | grep real | awk '{print $2}')
# 		echo "with mpi(${PROC})     $cal_t" | tee -a test-vector.txt
# 		echo | tee -a test-vector.txt
#     sleep 1
# 	done
# done
