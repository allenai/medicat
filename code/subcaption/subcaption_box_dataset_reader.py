from typing import Dict
from collections import defaultdict

import os
import json

from PIL import Image
import numpy as np
from transformers import BertConfig, BertTokenizer
import torch
from torchvision import transforms

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, ArrayField, MetadataField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer
from subcaption.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from subcaption.subcaption_ner_dataset_reader import convert_to_wordpieces

@DatasetReader.register('subcaption_box')
class SubcaptionNerDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer], lazy: bool = False, do_lowercase: bool = False, box_predictions_file: str = None, box_predictions_threshold: float = 0.7, max_seq_length: int = 512):
        super().__init__(lazy)
        self.token_indexers = token_indexers
        model_name = None
        for token_indexer in self.token_indexers.values():
            model_name = token_indexer._model_name
            break
        self.box_predictions_file = box_predictions_file
        self.box_predictions_threshold = box_predictions_threshold
        self.tokenizer = PretrainedTransformerTokenizer(model_name=model_name, do_lowercase=do_lowercase)
        # self.tag_map = {'B': 1, 'I': 2, 'L': 3, 'O': 4, 'U': 5}
        self.tag_map = {'B': 'B', 'I': 'I', 'L': 'L', 'O': 'O', 'U': 'U'}
        self.max_seq_length = max_seq_length

    def _read(self, fname: str):
        f = open(fname)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        box_predictions = None
        if self.box_predictions_file is not None:
            box_predictions = {}
            f = open(self.box_predictions_file)
            lines = f.readlines()
            box_predictions = [json.loads(line) for line in lines]
            box_predictions = {datum['image_id']: datum for datum in box_predictions}
        for i, datum in enumerate(data):
            # Don't include figures that were ignored or rejected in annotations
            if datum["answer"] != "accept":
                continue
            # Don't include figures that don't have subfigures
            if len(datum["spans"]) == 0:
                continue
            # Don't include figures that lack subcaptions
            if "subcaptions" not in datum or len(datum["subcaptions"]) == 0:
                continue
            subcaptions = {}
            image_id = datum["pdf_hash"]+"_"+datum["fig_uri"]
            # First, collect the tokens common to all subcaptions
            # These tokens will be removed from all subcaptions
            num_tokens = len(datum["tokens"])
            common_tokens = set(list(range(num_tokens)))
            excluded_tokens = set(list(range(num_tokens)))
            for subcaption_key in datum["subcaptions"]:
                datum["subcaptions"][subcaption_key] = set(datum["subcaptions"][subcaption_key])
            for subcaption_tokens in datum["subcaptions"].values():
                common_tokens = common_tokens.intersection(subcaption_tokens)
                excluded_tokens -= subcaption_tokens
            common_tokens = common_tokens.union(excluded_tokens)
            prev_span_indices = set()
            single_span_subcaptions = {}
            for subcaption_key in datum["subcaptions"]:
                subcaption_tokens = datum["subcaptions"][subcaption_key]
                spans = []
                span_lengths = []
                longest_span_index = None
                filtered_subcaption_tokens = subcaption_tokens-common_tokens
                if len(filtered_subcaption_tokens) == 0:
                    continue
                sorted_tokens = sorted(filtered_subcaption_tokens)
                curr_start = sorted_tokens[0]
                curr_end = sorted_tokens[0]
                for token in sorted_tokens[1:]:
                    if token > curr_end+1:
                        spans.append((curr_start, curr_end))
                        span_lengths.append(curr_end-curr_start+1)
                        if longest_span_index is None or span_lengths[-1] > span_lengths[longest_span_index]:
                            span_set = set(list(range(curr_start, curr_end+1)))
                            # We want our single-span subcaptions to be disjoint from one another
                            if len(span_set.intersection(prev_span_indices)) == 0:
                                longest_span_index = len(span_lengths)-1
                        curr_start = token
                        curr_end = token
                    else:
                        assert curr_end+1 == token
                        curr_end = token
                spans.append((curr_start, curr_end))
                span_lengths.append(curr_end-curr_start+1)
                if longest_span_index is None or span_lengths[-1] > span_lengths[longest_span_index]:
                    span_set = set(list(range(curr_start, curr_end+1)))
                    # We want our single-span subcaptions to be disjoint from one another
                    if len(span_set.intersection(prev_span_indices)) == 0:
                        longest_span_index = len(span_lengths)-1
                if longest_span_index is not None:
                    single_span_subcaptions[subcaption_key] = spans[longest_span_index]
                    for token in range(spans[longest_span_index][0], spans[longest_span_index][1]+1):
                        prev_span_indices.add(token)
                subcaptions[subcaption_key] = filtered_subcaption_tokens
            gold_subfigures = []
            pred_subfigures = []
            subcaptions_list = []
            for subfig in datum["spans"]:
                xcoords = [point[0] for point in subfig["points"]]
                ycoords = [point[1] for point in subfig["points"]]
                if subfig["label"] in subcaptions and len(subcaptions[subfig["label"]]) > 0:
                    if box_predictions is not None:
                        gold_subfigures.append([min(xcoords), min(ycoords), max(xcoords), max(ycoords)])
                        subcaptions_list.append(subcaptions[subfig["label"]])
                    else:
                        # span = single_span_subcaptions[subfig["label"]]
                        # subcaption = set([token for token in range(span[0], span[1]+1)])
                        instance = self.text_to_instance([token["text"] for token in datum["tokens"]], datum["pdf_hash"]+"_"+datum["fig_uri"], datum["height"], datum["width"], common_tokens, subcaptions[subfig["label"]], [min(xcoords), min(ycoords), max(xcoords), max(ycoords)])
                        if instance is not None:
                            yield instance
            if box_predictions is not None:
                first = False
                for box, score in zip(box_predictions[image_id]['predictions']['boxes'], box_predictions[image_id]['predictions']['scores']):
                    if score > self.box_predictions_threshold:
                        if not first:
                            gold_subfigs = gold_subfigures
                            gold_subcaps = subcaptions_list
                        else:
                            gold_subfigs = []
                            gold_subcaps = []
                        yield self.text_to_instance([token["text"] for token in datum["tokens"]], datum["pdf_hash"]+"_"+datum["fig_uri"], datum["height"], datum["width"], common_tokens, gold_subcaps, box, gold_subfigs)
                    first = True

    def text_to_instance(self, tokens, image_id, height, width, common_tokens=None, subcaption=None, box=None, gold_boxes=None):
        subcaption_input = [subcaption]
        if gold_boxes is not None:
            subcaption_input = subcaption
        wordpieces, common_wordpieces, _, wordpiece_subcaption = convert_to_wordpieces(self.tokenizer._tokenizer, tokens, common_tokens, None, subcaption_input)

        if len(wordpieces) > self.max_seq_length-2:
            wordpieces = wordpieces[:self.max_seq_length-2]
            # wordpiece_subcaptions = [new_subcaption]
        wordpieces = ["[CLS]"]+wordpieces+["[SEP]"]
        common_wordpieces = set([wordpiece_index+1 for wordpiece_index in common_wordpieces])
        wordpiece_tokens = [Token(wordpiece) for wordpiece in wordpieces]
        fields = {}
        tokens_field = TextField(wordpiece_tokens, token_indexers=self.token_indexers)
        fields["tokens"] = tokens_field
        if gold_boxes is None:
            # Account for CLS token
            wordpiece_subcaption = [wordpiece+1 for wordpiece in sorted(wordpiece_subcaption[0])]
            print([wordpieces[index] for index in wordpiece_subcaption])
            spans = []
            span_start = wordpiece_subcaption[0]
            span_end = wordpiece_subcaption[0]
            for token in wordpiece_subcaption[1:]:
                if token > span_end+1:
                    spans.append((span_start, span_end))
                    span_start = token
                    span_end = token
                else:
                    assert token == span_end+1
                    span_end = token
            spans.append((span_start, span_end))
            # assert len(spans) == 1
            tags = [self.tag_map['O'] for _ in wordpieces]
            for span in spans:
                if span[0] == span[1]:
                    tags[span[0]] = self.tag_map["U"]
                else:
                    tags[span[0]] = self.tag_map["B"]
                    tags[span[1]] = self.tag_map["L"]
                    for index in range(span[0]+1, span[1]):
                        tags[index] = self.tag_map["I"]
            print([word for word, tag in zip(wordpieces, tags) if tag != self.tag_map["O"]])
            print(tags)
            print(spans)
            fields["tags"] = SequenceLabelField(tags, tokens_field, "labels")
        else:
            wordpiece_subcaption = [[wordpiece+1 for wordpiece in sorted(subcap)] for subcap in wordpiece_subcaption]
        # fields["start"] = ArrayField(np.array(spans[0][0], dtype=np.int64), dtype=np.int64)
        # fields["end"] = ArrayField(np.array(spans[0][1], dtype=np.int64), dtype=np.int64)
        normalized_box = [box[0]/float(width), box[1]/float(height), box[2]/float(width), box[3]/float(height)]
        print(normalized_box)
        fields["box"] = ArrayField(np.array(normalized_box, dtype=np.float32), dtype=np.float32)
        fields["metadata"] = MetadataField({"image_id": image_id, "words": wordpieces, "common_wordpieces": common_wordpieces, "gold_subcaption": wordpiece_subcaption, "gold_subfigure": box if gold_boxes is None else gold_boxes, "predicted_subfigure": box})
        return Instance(fields)
