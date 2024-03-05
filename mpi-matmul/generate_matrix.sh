#!/bin/bash
echo "generate test-matrices with python3 if no test data found"
echo

MATRIX_SIZES=(
    128
    256
    512
    1024
    2048
)
for SZ in "${MATRIX_SIZES[@]}"
do
    FILE="data/mat_${SZ}x1.txt"
    if [ ! -f $FILE ]; then
        echo "generate ${SZ}x1 matrix..."
        python3 random_float_matrix.py ${SZ} 1 > $FILE
    fi
    FILE1="data/mat_${SZ}x${SZ}.txt"
    FILE2="data/mat_${SZ}x${SZ}b.txt"

    if [ ! -f $FILE1 ]; then
        echo "generate ${SZ}x${SZ} matrix..."
        python3 random_float_matrix.py $SZ $SZ > $FILE1
    fi
    if [ ! -f $FILE2 ]; then
        echo "generate ${SZ}x${SZ} matrix..."
        python3 random_float_matrix.py $SZ $SZ > $FILE2
    fi
done