#!/bin/sh
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --conf-thresh 0.35 --iou-thresh 0.6 --nms 50 --model m_32 > ./measures.txt
python3 ./benchmark_demo.py --repetitions 1000 --warmup 100 --conf-thresh 0.35 --iou-thresh 0.6 --nms 50 --model vgg_64 >> ./measures.txt
