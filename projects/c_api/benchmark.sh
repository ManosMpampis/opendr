#!/bin/sh
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_m_32 m_32 ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 > ./measures.txt
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64 vgg_64 ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt
