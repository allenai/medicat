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

def convert_to_wordpieces(tokenizer, tokens, common_tokens, spans, full_subcaptions):
    token_to_wordpiece_map = {}
    wordpieces = []
    for index, token in enumerate(tokens):
        token_to_wordpiece_map[index] = len(wordpieces)
        wordpieces += tokenizer.wordpiece_tokenizer.tokenize(token)
    token_to_wordpiece_map[len(tokens)] = len(wordpieces)
    new_spans = None
    if spans is not None:
        new_spans = [(token_to_wordpiece_map[span[0]], token_to_wordpiece_map[span[1]+1]-1) if span is not None else None for span in spans]
    new_full_subcaptions = None
    if full_subcaptions is not None:
        new_full_subcaptions = [set([wordpiece for token in subcaption for wordpiece in range(token_to_wordpiece_map[token], token_to_wordpiece_map[token+1])]) for subcaption in full_subcaptions]
    common_wordpieces = set()
    for token in common_tokens:
        for index in range(token_to_wordpiece_map[token], token_to_wordpiece_map[token+1]):
            common_wordpieces.add(index)
    return wordpieces, common_wordpieces, new_spans, new_full_subcaptions

@DatasetReader.register('subcaption_ner')
class SubcaptionNerDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False, do_lowercase: bool = False, box_predictions_file: str = None, box_predictions_threshold: float = 0.7, max_seq_length: int = 512):
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
            image_id = datum["pdf_hash"]+"_"+datum["fig_uri"]
            single_span_subcaptions = {}
            subcaptions = {}
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
            for subcaption_key in datum["subcaptions"]:
                subcaption_tokens = datum["subcaptions"][subcaption_key]
                spans = []
                span_lengths = []
                longest_span_index = None
                assert len(subcaption_tokens.intersection(excluded_tokens)) == 0
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
            single_span_subcaptions_list = []
            for subfig in datum["spans"]:
                xcoords = [point[0] for point in subfig["points"]]
                ycoords = [point[1] for point in subfig["points"]]
                if subfig["label"] in subcaptions and len(subcaptions[subfig["label"]]) > 0:
                    gold_subfigures.append([min(xcoords), min(ycoords), max(xcoords), max(ycoords)])
                    subcaptions_list.append(subcaptions[subfig["label"]])
                if subfig["label"] in single_span_subcaptions:
                    single_span_subcaptions_list.append(single_span_subcaptions[subfig["label"]])
                else:
                    single_span_subcaptions_list.append(None)
            if box_predictions is not None:
                for box, score in zip(box_predictions[image_id]['predictions']['boxes'], box_predictions[image_id]['predictions']['scores']):
                    if score > self.box_predictions_threshold:
                        pred_subfigures.append(box)
            else:
                for subfig in gold_subfigures:
                    pred_subfigures.append(subfig)
            yield self.text_to_instance([token["text"] for token in datum["tokens"]], image_id, pred_subfigures, common_tokens, single_span_subcaptions_list, subcaptions_list, gold_subfigures)

    def text_to_instance(self, tokens, image_id, predicted_boxes=None, common_tokens=None, single_span_subcaptions=None, subcaptions=None, gold_boxes=None):
        wordpieces, common_wordpieces, wordpiece_spans, wordpiece_subcaptions = convert_to_wordpieces(self.tokenizer._tokenizer, tokens, common_tokens, single_span_subcaptions, subcaptions)
        if len(wordpieces) > self.max_seq_length-2:
            new_common_wordpieces = set()
            for token in common_wordpieces:
                if token < self.max_seq_length-2:
                    new_common_wordpieces.add(token)
            new_wordpiece_spans = []
            for span in wordpiece_spans:
                if span is None:
                    new_wordpiece_spans.append(span)
                elif span[0] < self.max_seq_length-2:
                    new_wordpiece_spans.append((span[0], min(self.max_seq_length-3, span[1])))
                else:
                    new_wordpiece_spans.append(None)
            new_wordpiece_subcaptions = []
            for subcaption in wordpiece_subcaptions:
                new_subcaption = set()
                for token in subcaption:
                    if token < self.max_seq_length-2:
                        new_subcaption.add(token)
                new_wordpiece_subcaptions.append(new_subcaption)
            wordpieces = wordpieces[:self.max_seq_length-2]
            wordpiece_spans = new_wordpiece_spans
            # wordpiece_subcaptions = new_wordpiece_subcaptions
        wordpieces = ["[CLS]"]+wordpieces+["[SEP]"]
        # Account for CLS token
        wordpiece_spans = [(span[0]+1, span[1]+1) for span in wordpiece_spans if span is not None]
        wordpiece_subcaptions = [[wordpiece+1 for wordpiece in sorted(subcaption)] for subcaption in wordpiece_subcaptions]
        common_wordpieces = set([wordpiece_index+1 for wordpiece_index in common_wordpieces])
        wordpiece_tokens = [Token(wordpiece) for wordpiece in wordpieces]
        tokens_field = TextField(wordpiece_tokens, token_indexers=self.token_indexers)
        fields = {}
        tags = [self.tag_map['O'] for _ in wordpieces]
        prev_span_indices = set()
        for span in wordpiece_spans:
            span_set = set(list(range(span[0], span[1]+1)))
            if len(span_set.intersection(prev_span_indices)) > 0:
                continue
            prev_span_indices = prev_span_indices.union(span_set)
            if span[0] == span[1]:
                tags[span[0]] = self.tag_map["U"]
            else:
                tags[span[0]] = self.tag_map["B"]
                tags[span[1]] = self.tag_map["L"]
                for index in range(span[0]+1, span[1]):
                    tags[index] = self.tag_map["I"]
        fields["tokens"] = tokens_field
        fields["tags"] = SequenceLabelField(tags, tokens_field, "labels")
        fields["metadata"] = MetadataField({"image_id": image_id, "words": wordpieces, "common_wordpieces": common_wordpieces, "gold_subcaptions": wordpiece_subcaptions, "gold_subfigures": gold_boxes, "predicted_subfigures": predicted_boxes})
        return Instance(fields)
