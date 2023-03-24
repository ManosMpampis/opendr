#!/bin/sh
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big > ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big_conv >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big_conv >> ./measures_pc_gpan_vgg.txt
#
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium_conv >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium_conv >> ./measures_pc_gpan_vgg.txt

#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small >> ./measures_pc_gpan_vgg.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small_conv >> ./measures_pc_gpan_vgg.txt
#python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small_conv >> ./measures_pc_gpan_vgg.txt
