#
# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash
CPP = g++
CC = gcc
AR = ar
WGET = wget -r -q
MV = mv

MKDIR_P = mkdir -p
BUILD_DIR = build
LIB_DIR = $(OPENDR_HOME)/lib
DATA_DIR = ./data

CFLAGS  = -fPIC
INC = -I/usr/local/include/onnxruntime/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)/lib -lopendr

all: clean demos download

demos: nanodet

nanodet_demo: $(BUILD_DIR)/object_detection_2d/nanodet_jit_demo
nanodet_benchmark: $(BUILD_DIR)/object_detection_2d/nanodet_jit_benchmark
nanodet: nanodet_demo nanodet_benchmark

download:
	echo Downloading resources for C API...; \
    $(MKDIR_P) $(DATA_DIR)/object_detection_2d/nanodet; \
    $(WGET) ftp://opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/images/*; \
    $(MV) opendrdata.csd.auth.gr/perception/object_detection_2d/nanodet/images  $(DATA_DIR)/object_detection_2d/nanodet/database; \
    $(RM) -r opendrdata.csd.auth.gr; \

$(BUILD_DIR)/object_detection_2d/nanodet_jit_demo:
	$(MKDIR_P) $(BUILD_DIR)/object_detection_2d
	@+echo "Building nanodet object detection demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/object_detection_2d/nanodet_jit_demo samples/object_detection_2d/nanodet/nanodet_jit_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

$(BUILD_DIR)/object_detection_2d/nanodet_jit_benchmark:
	$(MKDIR_P) $(BUILD_DIR)/object_detection_2d
	@+echo "Building nanodet object detection demo..."
	$(CPP)  $(CFLAGS) -o $(BUILD_DIR)/object_detection_2d/nanodet_jit_benchmark samples/object_detection_2d/nanodet/nanodet_jit_benchmark.cpp $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

clean:
	@+echo "Cleaning C API demo binaries and temporary files..."
	@+$(RM) -rf $(BUILD_DIR)/*
	@+echo "Done!"
