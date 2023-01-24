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
#include <assert.h>
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include "opendr_utils.h"

START_TEST(image_load_test) {
  // Load an image and performance inference
  OpendrImageT image;
  // An example of an image that exist
  loadImage("data/database/1/1.jpg", &image);
  ck_assert(image.data);
  // An example of an image that does not exist
  loadImage("images/not_existant/1.jpg", &image);
  ck_assert(image.data == 0);

  // Free the resources
  freeImage(&image);
}
END_TEST

START_TEST(detection_vector_init_load_test) {
  // Initialize a detection target vector
  opendr_detection_vector_target_t detection_vector;
  // init functions uses load internally
  init_detections_vector(&detection_vector);
  ck_assert(detection_vector.starting_pointer);
  // Free the resources
  free_detections_vector(&detection_vector);
  ck_assert(detection_vector.starting_pointer == NULL);
}
END_TEST

START_TEST(tensor_init_load_test) {
  // Initialize a detection target vector
  opendr_tensor_t opendr_tensor;
  // init functions uses load internally
  init_tensor(&opendr_tensor);
  ck_assert(opendr_tensor.data == NULL);

  void *tensor_data = malloc(1 * sizeof(float));
  load_tensor(&opendr_tensor, tensor_data, 1, 1, 1, 1, 1);
  ck_assert(opendr_tensor.data);
  // Free the resources
  free(tensor_data);
  free_tensor(&opendr_tensor);
  ck_assert(opendr_tensor.data == NULL);
}
END_TEST

START_TEST(tensor_vector_init_load_test) {
  // Initialize a detection target vector
  opendr_tensor_vector_t tensor_vector;
  // init functions uses load internally
  init_tensor_vector(&tensor_vector);

  ck_assert(tensor_vector.batch_sizes == NULL);
  ck_assert(tensor_vector.frames == NULL);
  ck_assert(tensor_vector.channels == NULL);
  ck_assert(tensor_vector.widths == NULL);
  ck_assert(tensor_vector.heights == NULL);
  ck_assert(tensor_vector.memories == NULL);

  opendr_tensor_t tensor[1];
  init_tensor(&(tensor[0]));

  void *tensor_data = malloc(1 * sizeof(float));
  load_tensor(&(tensor[0]), tensor_data, 1, 1, 1, 1, 1);

  load_tensor_vector(&tensor_vector, tensor, 1);
  ck_assert(tensor_vector.memories);
  // Free the resources
  free(tensor_data);
  free_tensor(&(tensor[0]));

  free_tensor_vector(&tensor_vector);
  ck_assert(tensor_vector.memories == NULL);
}
END_TEST

Suite *opendr_utilities_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("OpenDR Utilities");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, image_load_test);
  tcase_add_test(tc_core, detection_vector_init_load_test);
  tcase_add_test(tc_core, tensor_init_load_test);
  tcase_add_test(tc_core, tensor_vector_init_load_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = opendr_utilities_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}