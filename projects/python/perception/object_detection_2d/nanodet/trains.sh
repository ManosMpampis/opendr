#!/bin/sh
export OPENDR_HOME=/home/manos/Thesis/opendr
export PYTHONPATH=$OPENDR_HOME/src:$PYTHONPATH
export PYTHON=python3
export LD_LIBRARY_PATH=$OPENDR_HOME/lib:$LD_LIBRARY_PATH


#Allea Weed dataset
export DIR_PATH=/media/manos/hdd/allea_datasets/weedDataset
export DATA_FOLDER=1080p

python3 ./train.py --batch-size 128 --lr 0.001 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model g --dataset_path $DIR_PATH/$DATA_FOLDER
#python3 ./train.py --batch-size 256 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 60 --model plus_EMA_vgg_64_very_small_augmented --dataset_path $DIR_PATH/$DATA_FOLDER
#python3 ./train.py --batch-size 256 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_vgg_64_very_small --dataset_path $DIR_PATH/$DATA_FOLDER

#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_vgg_64_very_small_augmented --dataset_path $DIR_PATH/$DATA_FOLDER
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_vgg_64_very_small --dataset_path $DIR_PATH/$DATA_FOLDER





# Football dataset
export DIR_PATH=/media/manos/hdd/Binary_Datasets/Football/
export DATA_FOLDER=192x192_3pos_36neg_padded_augmented_size_0.9-2.0
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 405 --model plus_EMA_vgg_64_very_small_augmented_pre16 --dataset_path $DIR_PATH/$DATA_FOLDER
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 600 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_vgg_64_very_small_augmented_pre16_2 --dataset_path $DIR_PATH/$DATA_FOLDER
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model vgg_64_very_small --dataset_path $DIR_PATH/192x192_3pos_36neg_padded_augmented_size_0.9-2.0
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 110 --model plus_EMA_hinge_vgg_64_very_small_augmented --dataset_path $DIR_PATH/192x192_3pos_36neg_padded_augmented_size_0.9-2.0
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_hinge_loss_vgg_64_very_small_augmented --dataset_path $DIR_PATH/192x192_3pos_36neg_padded_augmented_size_0.9-2.0

#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model plus_EMA_hinge_labeling_vgg_64_very_small_augmented --dataset_path $DIR_PATH/192x192_3pos_36neg_padded_augmented_size_0.9-2.0

#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model g --dataset_path $DIR_PATH/192x192_3pos_36neg_padded_augmented_size_0.9-2.0

#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 600 --checkpoint-freq 5 --resume-from 395 --model vgg_64_very_small_norm_dataset --dataset_path $DIR_PATH/192x192_3pos_36neg_padded
#python3 ./train.py --batch-size 512 --lr 0.002 --n-epochs 600 --checkpoint-freq 5 --resume-from 295 --model vgg_64_very_small_norm_dataset_augmented --dataset_path $DIR_PATH/192x192_3pos_36neg_padded

#python3 ./train.py --batch-size 512 --lr 0.01 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model vgg_64_very_small --dataset_path $DIR_PATH/192x192_3pos_36neg_padded
#python3 ./train.py --batch-size 512 --lr 0.01 --n-epochs 500 --checkpoint-freq 5 --resume-from 0 --model vgg_64_very_small2 --dataset_path $DIR_PATH/192x192_3pos_36neg_padded
