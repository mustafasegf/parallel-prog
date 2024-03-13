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

rm -f time_old.txt
rm -f time_summary_old.txt

[ -f "time.txt" ] && cp time.txt time_old.txt
[ -f "time_summary.txt" ] && cp time_summary.txt time_summary_old.txt

echo -n >time.txt
echo -n >time_summary.txt

for SZ in "${MATRIX_SIZES[@]}"; do
	echo "* * * * * * * ${SZ}x${SZ} Matrix"
	for PROC in "${NPS[@]}"; do
		OUTPUT=$(mpirun -np "${PROC}" main "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x${SZ}b.txt")
		echo "${OUTPUT}" >>time.txt
		echo "${OUTPUT}" | LC_ALL=id_ID.UTF-8 awk '
      {
        compute += $7; 
        comm += $10;
      } END  {  
        comm = int(comm / $2);
        compute = int(compute / $2);
        printf "np:%-3d %-16s compute: %-10'"'"'d comm: %-10'"'"'d (%'"'"'d/%'"'"'d)\n", $2, $4, compute, comm, compute, comm;
      }' | tee -a time_summary.txt
	done
	echo "" | tee -a time.txt
	echo "" | tee -a time_summary.txt

done

for SZ in "${MATRIX_SIZES[@]}"; do
	echo "* * * * * * * ${SZ}x1 Matrix"
	for PROC in "${NPS[@]}"; do
		OUTPUT=$(mpirun -np "${PROC}" main "data/mat_${SZ}x${SZ}.txt" "data/mat_${SZ}x1.txt")
		echo "${OUTPUT}" >>time.txt
		echo "${OUTPUT}" | LC_ALL=id_ID.UTF-8 awk '
      {
        compute += $7; 
        comm += $10;
      } END  {  
        comm = int(comm / $2);
        compute = int(compute / $2);
        printf "np:%-3d %-16s compute: %-10'"'"'d comm: %-10'"'"'d (%'"'"'d/%'"'"'d)\n", $2, $4, compute, comm, compute, comm;
      }' | tee -a time_summary.txt
	done
	echo "" | tee -a time.txt
	echo "" | tee -a time_summary.txt
done
