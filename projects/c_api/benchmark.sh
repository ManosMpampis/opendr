#!/bin/sh
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_big vgg_64_big ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 > ./measures.txt
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_big_conv vgg_64_big_conv ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt

./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_medium vgg_64_medium ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_medium_conv vgg_64_medium_conv ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt

./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_small vgg_64_small ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt
./build/object_detection_2d/nanodet_jit_benchmark ./data/object_detection_2d/nanodet/nanodet_vgg_64_small_conv vgg_64_small_conv ./data/object_detection_2d/nanodet/database/000000000036.jpg 1000 100 >> ./measures.txt
