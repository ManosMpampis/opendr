#!/bin/sh
#export DIR_NAME_0=./new_model_2k.txt
#export DIR_NAME_1=./new_model_1k.txt
export DIR_NAME_2_0=./new_model_big_ch_1k_0.txt
export DIR_NAME_2_1=./new_model_big_ch_1k_1.txt
export DIR_NAME_2_2=./new_model_big_ch_1k_2.txt
#export DIR_NAME_3=./new_model_big_1k.txt

#python3 ./benchmark_demo.py --model test --repetitions 1000 --warmup 10 > $DIR_NAME_0
#python3 ./benchmark_demo.py --model test_1k --repetitions 1000 --warmup 10 > $DIR_NAME_1
python3 ./benchmark_demo.py --model test_big_ch --repetitions 1000 --warmup 10 > $DIR_NAME_2_0
python3 ./benchmark_demo.py --model test_big_ch --repetitions 1000 --warmup 10 > $DIR_NAME_2_1
python3 ./benchmark_demo.py --model test_big_ch --repetitions 1000 --warmup 10 > $DIR_NAME_2_2
#python3 ./benchmark_demo.py --model test_big_1k --repetitions 1000 --warmup 10 > $DIR_NAME_3
