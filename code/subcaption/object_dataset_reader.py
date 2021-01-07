import os
import json

import numpy as np
from PIL import Image
from torchvision import transforms

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance

@DatasetReader.register('object')
class ObjectDatasetReader(DatasetReader):
    def __init__(self, image_root: str, lazy: bool = False, pass_boxes: bool = True, start_line: int = 0, end_line: int = None):
        super().__init__(lazy)
        self.image_root = image_root
        self.pass_boxes = pass_boxes
        self.start_line = start_line
        self.end_line = end_line
        expected_img_size = 224
        self.image_transform = transforms.Compose([
                                    # transforms.Resize(expected_img_size),
                                    # transforms.CenterCrop(expected_img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def _read(self, fname: str):
        f = open(fname)
        lines = f.readlines()
        if self.end_line is None:
            self.end_line = len(lines)
        data = [json.loads(line) for line in lines[self.start_line:self.end_line]]
        for i, datum in enumerate(data):
            image_id = datum['pdf_hash']+'_'+datum['fig_uri']
            if datum['answer'] == 'ignore':
                continue
            if self.image_root == "":
                image_root = datum['image_root']
            else:
                image_root = self.image_root
            if not os.path.exists(os.path.join(image_root)):
                continue
            image = Image.open(os.path.join(image_root, image_id)).convert('RGB')
            if datum['answer'] == 'reject' or 'spans' not in datum or len(datum['spans']) == 0:
                continue
                # datum['spans'] = [{'points': [[0, 0], [0, datum['height']], [datum['width'], datum['height']], [datum['width'], 0]]}]
            for span in datum['spans']:
                if span['points'][2][0] < span['points'][2][0]:
                    print('A', span)
                if span['points'][2][1] < span['points'][2][1]:
                    print('B', span)
            yield self.text_to_instance(image, datum['spans'], image_id, datum['width'], datum['height'])

    def text_to_instance(self, image, boxes, image_id, width, height):
        image = self.image_transform(image)
        new_boxes = []
        for box in boxes:
            x_coords = [point[0] for point in box['points']]
            y_coords = [point[1] for point in box['points']]
            coords = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            if max(x_coords) > min(x_coords) and max(y_coords) > min(y_coords):
                new_boxes.append(coords)
        boxes = np.array(new_boxes)
        fields = {'images': MetadataField(image), # , dtype=np.float32),
                  'heights': MetadataField(height),
                  'widths': MetadataField(width),
                  'image_ids': MetadataField(image_id)}
        if self.pass_boxes:
            fields['boxes'] = ArrayField(boxes, dtype=np.float32)
            fields['num_boxes'] =  MetadataField(boxes.shape[0])
        return Instance(fields)
