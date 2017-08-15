export LIBRARY_PATH=/usr/lib64/nvidia-304xx
if gcc cable.c -l:libOpenCL.so.1 -o run
then
	time ./run > data.csv
	echo "python plot.py data.csv 256 membrane.V"
fi
