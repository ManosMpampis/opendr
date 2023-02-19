#!/bin/sh
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/optimized_model m ./data/object_detection_2d/nanodet/database/000000000036.jpg 100 1000 > ./measures.txt

