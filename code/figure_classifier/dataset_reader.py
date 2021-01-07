import os
from overrides import overrides
import json

from PIL import Image
import numpy as np
from torchvision import transforms

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, MetadataField
from allennlp.data.instance import Instance

@DatasetReader.register("docfigure")
class DocFigureDatasetReader(DatasetReader):
    def __init__(self,
                 image_root: str,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.image_root = image_root
        expected_img_size = 224
        self.image_transform = transforms.Compose([
                                    transforms.Resize(expected_img_size),
                                    transforms.CenterCrop(expected_img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @overrides
    def _read(self, file_path: str):
        input_file = open(file_path)
        lines = input_file.readlines()
        if 'json' in file_path:
            data = [json.loads(line) for line in lines]
            for i, datum in enumerate(data):
                image_id = datum['pdf_hash']+'_'+datum['fig_uri']
                label = ""
                image_root = self.image_root
                if 'image_root' in datum and datum['image_root'] is not None:
                    image_root = datum['image_root']
                    instance = self.text_to_instance(image_id=image_id, label=label, image_root=image_root)
                    if instance is not None:
                        yield instance
        else:
            for index, line in enumerate(lines):
                parts = line.split(', ')
                image_id = parts[0].strip()
                label = parts[1].strip()
                yield self.text_to_instance(image_id, label)

    def text_to_instance(self,
                         image_id: str,
                         label: str,
                         image_root: str = None) -> Instance:
        if image_root is None:
            image_root = self.image_root
        try:
            image = Image.open(os.path.join(image_root, image_id)).convert('RGB')
        except:
            return None
        fields: Dict[str, Field] = {}
        fields['image'] = ArrayField(self.image_transform(image).numpy())
        if len(label) > 0:
            fields['label'] = LabelField(label)
        fields['image_id'] = MetadataField(image_id)
        return Instance(fields)
