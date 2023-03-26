#!/bin/sh
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big > ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_big_conv >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_big_conv >> ./measures.txt

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_medium_conv >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_medium_conv >> ./measures.txt

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_small_conv >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_small_conv >> ./measures.txt

python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_very_small >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --model vgg_64_very_small_conv >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_very_small >> ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --mix --model vgg_64_very_small_conv >> ./measures.txt
