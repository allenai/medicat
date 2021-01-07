from typing import List, Dict, Union
from overrides import overrides
import os
import sys
import json
import random
import re
import unicodedata
import string

from PIL import Image
import numpy as np
from torchvision import transforms

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ArrayField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

from transformers import BertConfig, BertTokenizer

from matching.token_indexer import BertFromConfigIndexer

@DatasetReader.register('image_text_matching')
class MatchingDatasetReader(DatasetReader):
    def __init__(self, image_root: str, scibert_path: str, lazy: bool = False, limit: int = None, max_sequence_length: int = 512, different_type_for_refs: bool = True, use_refs: bool = True):
        super().__init__(lazy)
        self.image_root = image_root
        config = BertConfig.from_json_file(os.path.join(scibert_path, 'config.json'))
        self.tokenizer = BertTokenizer(config=config, vocab_file=os.path.join(scibert_path, 'vocab.txt'))
        self.token_indexer = {'tokens': BertFromConfigIndexer(config=config, vocab_path=os.path.join(scibert_path, 'vocab.txt'), namespace='bert_tokens')}
        expected_img_size = 224
        self.image_transform = transforms.Compose([
                                    transforms.Resize(expected_img_size),
                                    transforms.CenterCrop(expected_img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.use_refs = use_refs
        self.different_type_for_refs = different_type_for_refs
        self.limit = limit
        self.max_sequence_length = max_sequence_length
        self.word_tokenizer = WordTokenizer()
        self.caption_field = "caption"

    @overrides
    def _read(self, fnames: Union[str, List[str]]):
        if isinstance(fnames, str):
            fnames = [fnames]
        for fname in fnames:
            f = open(fname)
            lines = f.readlines()
            examples = [json.loads(line) for line in lines]
            for i, ex in enumerate(examples):
                if self.limit is not None and i > self.limit:
                    break
                image_path = os.path.join(self.image_root, ex['image_id'])
                image_id = ex['image_id']
                image = Image.open(image_path).convert('RGB')
                caption = ex[self.caption_field]
                captions = [caption]
                text_types = [0]
                if self.use_refs and 'gorc_references' in ex and ex['gorc_references'] is not None:
                    captions += ex['gorc_references']
                    if self.different_type_for_refs:
                        typ = 1
                    else:
                        typ = 0
                    text_types += [typ for _ in ex['gorc_references']]
                for caption, text_type in zip(captions, text_types):
                    yield self.text_to_instance(image, image_id, caption, text_type, True)

    def text_to_instance(self, image, image_id, utterance, text_type, matching_label):
        input_image = self.image_transform(image)
        original_tokenized_utterance = self.tokenizer.tokenize(utterance)
        tokenized_utterance_strings = ['[CLS]']+original_tokenized_utterance+['[SEP]']
        tokenized_utterance = [Token(token) for token in tokenized_utterance_strings]
        tokenized_utterance = tokenized_utterance[:self.max_sequence_length]
        tokenized_utterance_strings = tokenized_utterance_strings[:self.max_sequence_length]
        utterance_field = TextField(tokenized_utterance, self.token_indexer)
        image_field = ArrayField(input_image.numpy())
        if text_type == 0:
            segment_ids = ArrayField(np.zeros((len(tokenized_utterance)), dtype=np.int64), dtype=np.int64)
        else:
            segment_ids = ArrayField(np.ones((len(tokenized_utterance)), dtype=np.int64)*text_type, dtype=np.int64)
        image_id_field = MetadataField(image_id)
        fields = {'images': image_field,
                  'token_ids': utterance_field,
                  'segment_ids': segment_ids,
                  'image_ids': image_id_field}
        return Instance(fields)
