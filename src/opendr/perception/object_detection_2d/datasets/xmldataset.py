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

import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import cv2

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.perception.object_detection_2d.datasets.detection_dataset import DetectionDataset, is_image_type, remove_extension


class XMLBasedDataset(DetectionDataset):
    """
    Reader class for datasets annotated with the LabelImg tool in Pascal VOC XML format.
    The dataset should be in the following structure:
    data_root
    |-- images
    |-- annotations
    The exact names of the folders can be passed as arguments (images_dir) and (annotations_dir).
    """
    def __init__(self, dataset_type, root, classes=None, image_transform=None,
                 target_transform=None, transform=None, splits='',
                 images_dir='images', annotations_dir='annotations', preload_anno=False):
        self.abs_images_dir = os.path.join(root, images_dir)
        self.abs_annot_dir = os.path.join(root, annotations_dir)
        image_names = [im_filename for im_filename in os.listdir(self.abs_images_dir)
                       if is_image_type(im_filename)]

        if classes is None:
            classes = []
        self.classes = classes
        super().__init__(classes, dataset_type, root, image_transform=image_transform, target_transform=target_transform,
                         transform=transform, image_paths=image_names, splits=splits)
        self.bboxes = []
        self.preload_anno = preload_anno
        if preload_anno:
            for image_name in image_names:
                annot_file = os.path.join(self.abs_annot_dir, remove_extension(image_name) + '.xml')
                if not os.path.exists(annot_file):
                    self.bboxes.append([])
                    continue
                bboxes = self._read_annotation_file(annot_file)
                self.bboxes.append(bboxes)

    def _read_annotation_file(self, filename):
        root = ET.parse(filename).getroot()
        bounding_boxes = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                self.classes.append(cls_name)
            cls_id = self.classes.index(cls_name)
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            bounding_box = BoundingBox(name=int(cls_id),
                                       left=float(xmin), top=float(ymin),
                                       width=float(xmax) - float(xmin),
                                       height=float(ymax) - float(ymin))
            bounding_boxes.append(bounding_box)
        return BoundingBoxList(boxes=bounding_boxes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_name = self.image_paths[item]
        image_path = os.path.join(self.abs_images_dir, image_name)
        img_np = cv2.imread(image_path)
        img = Image(img_np)

        if self.preload_anno:
            label = self.bboxes[item]
        else:
            annot_file = os.path.join(self.abs_annot_dir, remove_extension(image_name) + '.xml')
            if not os.path.exists(annot_file):
                label = []
            else:
                label = self._read_annotation_file(annot_file)

        if self._image_transform is not None:
            img = self._image_transform(img)

        if self._target_transform is not None:
            label = self._target_transform(label)

        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def get_image(self, item):
        image_name = self.image_paths[item]
        image_path = os.path.join(self.abs_images_dir, image_name)
        img_np = cv2.imread(image_path)
        if self._image_transform is not None:
            img = self._image_transform(img_np)
        return img

    def get_bboxes(self, item):
        boxes = self.bboxes[item]
        if self._target_transform is not None:
            boxes = self._target_transform(boxes)
        return boxes

if __name__ == '__main__':
    from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes

    # dataset_metadata = {
    #     "data_root": "/media/manos/hdd/Binary_Datasets/Football/192x192_3pos_36neg_padded_augmented",
    #     "classes": ["player"],
    #     "dataset_type": "BINARY_FOOTBALL",
    # }
    dataset_metadata = {
        "data_root": "/home/manos/data/weedDataset/original_dataset",
        "classes": ["poaceae", "brassicaceae"],
        "dataset_type": "",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]
    dataset = XMLBasedDataset(root=f'{data_root}/test', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)

    dataset = XMLBasedDataset(root=f'/home/manos/data/weedDataset/small_annots/big_annots/test', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='../../../original_dataset/test/annotations', classes=classes)
    # for i, ((img, targets), (img2, targets2)) in enumerate(zip(dataset, dataset2)):
    for i, (img, targets) in enumerate(dataset):
        sum = 0
        final_targets = []
        for target in targets:
            target_list = [target.left+1, target.top+1, target.width, target.height]
            for coordinate in target_list:
                if coordinate < 0:
                    sum += 1
                    print(target_list)
            area = target.width * target.height
            if area <= 4000 and target.width <= 320 and target.height <= 320:
                final_targets.append(target)
        final_targets = BoundingBoxList(final_targets)
        if sum > 0:
            print(f"found in: {dataset.image_paths[i]}")
        # continue

        if dataset.image_paths[i] in ["af7889ab6b42b57d7e34.jpg"]:
            print(dataset.image_paths[i])
            # img2 = img2.opencv()
            # img2 = draw_bounding_boxes(img2, targets2, class_names=dataset2.classes)
            # img2 = cv2.resize(img2, dsize=None, fx=0.3, fy=0.3)
            # cv2.imshow('img2', img2)

            img = img.opencv()
            img = draw_bounding_boxes(img, final_targets, class_names=dataset.classes)
            # img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('img', img)
            cv2.imwrite(f"/home/manos/Pictures/roobweeddataset/blacked_small_annot_example.png", img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
