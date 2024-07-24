#!/bin/sh
sshpass -p "*********" scp -r /home/manos/Thesis/opendr/src/opendr/perception/object_detection_2d/nanodet/ manos@#######:/home/manos/thesis/opendr/src/opendr/perception/object_detection_2d/
sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/tx2/* mmanos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/
sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/trt manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/plus_weight_averager_augmented_data_bigger_pos_with_augmentations/model_best/* manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/plus_weight_averager_augmented_data_bigger_pos_with_augmentations/model_best/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/simple_big_ch_very_small_annots_big_1_6/model_best/* manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/simple_big_ch_very_small_annots_big_1_6/model_best/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/simple_big_ch/model_best/* manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/temp/simple_big_ch/model_best/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/include/ manos@#######:/home/manos/thesis/opendr/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/src/c_api/tx2/* manos@#######:/home/manos/thesis/opendr/src/c_api/
#sshpass -p "*********" scp -r /home/manos/Thesis/opendr/projects/c_api/tx2/* manos@#######:/home/manos/thesis/opendr/projects/c_api/

sshpass -p "*********" ssh cidl@########### << EOF

# In tx2
#sudo docker run -it --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY ********/opendr:tx2_v2 /bin/bash


echo "*************" | sudo -S docker run -it --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY *********/opendr:tx2_v2 /bin/bash
#sudo docker run --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY ********/opendr:tx2_v2 /run_bench.sh
sudo docker run --entrypoint /run_bench.sh ***********/opendr:tx2_v2
#sudo docker exec ------------ /run_bench.sh
#cd ./opendr/
#source bin/activate_nvidia.sh

export OPENDR_HOME=/opendr/
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
alias python=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda/bin:$PATH
export MXNET_HOME=$OPENDR_HOME/mxnet/
export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export LC_ALL="C.UTF-8"
export MPLBACKEND=TkAgg

sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/src/opendr/perception/object_detection_2d/nanodet /opendr/src/opendr/perception/object_detection_2d/
sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/ /opendr/projects/python/perception/object_detection_2d/




#sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/include/ /opendr/
#sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/src/c_api/ /opendr/src/
#sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/projects/c_api/ /opendr/projects/

#cd /opendr/projects/python/perception/object_detection_2d/nanodet
#./benchmark.sh

sshpass -p "*********" scp -r /opendr/projects/python/perception/object_detection_2d/nanodet/new_measures manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/


# C api
#mkdir -r /opendr/projects/c_api/data/object_detection_2d/nanodet
#cp -rp ./jit/* /opendr/projects/c_api/data/object_detection_2d/nanodet/

#cd /opendr/src/c_api
#make

#cd /opendr/projects/c_api
#make
#./benchmark.sh

#sshpass -p "*********" scp -r /opendr/projects/c_api/measures.txt manos@#######:/home/manos/thesis/opendr/projects/c_api/measures_tx2.txt
exit
# Out of docker

exit
EOF


# My PC

sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/projects/python/perception/object_detection_2d/nanodet/new_measures/* /home/manos/Thesis/opendr/projects/python/perception/object_detection_2d/nanodet/measures/final/tx2/


sshpass -p "*********" scp -r manos@#######:/home/manos/thesis/opendr/projects/c_api/measures_tx2.txt /home/manos/Thesis/opendr/projects/c_api/measures_tx2_gpan_vgg_full.txt

