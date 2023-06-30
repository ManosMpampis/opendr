#!/bin/sh
export DIR_NAME=./vgg_32_measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big > $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big_conv >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big_conv >> $DIR_NAME

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium_conv >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium_conv >> $DIR_NAME

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small_conv >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small_conv >> $DIR_NAME

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_very_small >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_very_small_conv >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_very_small >> $DIR_NAME
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_very_small_conv >> $DIR_NAME
