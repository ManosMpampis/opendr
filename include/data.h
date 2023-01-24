/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef C_API_DATA_H
#define C_API_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

/***
 * OpenDR data type for representing images
 */
struct OpendrImage {
  void *data;
};
typedef struct OpendrImage OpendrImageT;

/***
 * Opendr data type for representing tensors
 */
struct opendr_tensor {
  int batch_size;
  int frames;
  int channels;
  int width;
  int height;

  float *data;
};
typedef struct opendr_tensor opendr_tensor_t;

/***
 * Opendr data type for representing tensor vectors
 */
struct opendr_tensor_vector {
  int n_tensors;
  int *batch_sizes;
  int *frames;
  int *channels;
  int *widths;
  int *heights;

  float **memories;
};
typedef struct opendr_tensor_vector opendr_tensor_vector_t;

#ifdef __cplusplus
}
#endif

#endif  // C_API_DATA_H
