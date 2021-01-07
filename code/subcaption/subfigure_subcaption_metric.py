from typing import List, Set

import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Metric

def iou(box1, box2):
    if max(box1[0], box2[0]) > min(box1[2], box2[2]) or max(box1[1], box2[1]) > min(box1[3], box2[3]):
        return 0
    intersect_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    intersect_area = (intersect_box[2]-intersect_box[0])*(intersect_box[3]-intersect_box[1])
    union_area = (box1[3]-box1[1])*(box1[2]-box1[0])+(box2[3]-box2[1])*(box2[2]-box2[0])-intersect_area
    if union_area == 0:
        return 0
    return intersect_area / union_area

class SubfigureSubcaptionAlignmentMetric(Metric):
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold
        self.reset()
    def __call__(self,
                 predicted_subfigures: List[List[List[float]]], # [batch_size, 4]
                 predicted_tokens: List[List[List[int]]], # [batch_size, sequence_length]
                 gold_subfigures: List[List[List[float]]], # [batch_size, 4]
                 gold_tokens: List[List[List[int]]],
                 wordpieces: List[List[str]],
                 common_wordpieces: List[Set[int]]): # [batch_size, sequence_length]
        batch_size = len(wordpieces)
        for i in range(batch_size):
            pred_filtered_tokens = []
            for subcaption in predicted_tokens[i]:
                filtered_subcaption = []
                for t in subcaption:
                    if wordpieces[i][t][0] != "#" and wordpieces[i][t].isalnum() and (len(wordpieces[i][t]) > 0 or not wordpieces[i][t].isalpha()):
                        # if t not in common_wordpieces[i]:
                        filtered_subcaption.append(t)
                pred_filtered_tokens.append(filtered_subcaption)
            for k in range(len(gold_subfigures[i])):
                max_iou = 0
                max_iou_index = None
                for p in range(len(predicted_subfigures[i])):
                    iou_value = iou(predicted_subfigures[i][p], gold_subfigures[i][k])
                    if iou_value > max_iou:
                        max_iou = iou_value
                        max_iou_index = p
                if max_iou < self.iou_threshold:
                    self.f1s.append(0)
                    continue
                gold_filtered_tokens = []
                for t in gold_tokens[i][k]:
                    if wordpieces[i][t][0] != "#" and wordpieces[i][t].isalnum() and (len(wordpieces[i][t]) > 0 or not wordpieces[i][t].isalpha()):
                        if t not in common_wordpieces[i]:
                            gold_filtered_tokens.append(t)
                if len(gold_filtered_tokens) == 0:
                    continue
                matching_pred_tokens = pred_filtered_tokens[max_iou_index]
                print([wordpieces[i][ind] for ind in matching_pred_tokens])
                intersection = set(gold_filtered_tokens).intersection(set(matching_pred_tokens))
                recall = float(len(intersection))/float(len(gold_filtered_tokens))
                if recall == 0:
                    self.f1s.append(0)
                    continue
                precision = float(len(intersection))/float(len(matching_pred_tokens))
                self.f1s.append(2.0*precision*recall/(precision+recall))

    def get_metric(self, reset: bool = False):
        if len(self.f1s) == 0:
            return 0.0
        avg = np.mean(self.f1s)
        if reset:
            self.reset()
        return avg

    def reset(self):
        self.f1s = []
