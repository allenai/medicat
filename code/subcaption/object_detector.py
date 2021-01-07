from typing import List
import sys

from allennlp.data import Vocabulary
from allennlp.models import Model

import torch
import torchvision
from pycocotools.coco import COCO

from subcaption.coco_eval import CocoEvaluator
from subcaption.utils import calculate_mAP

@Model.register('object')
class ObjectDetector(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=2)
        self.predicted_boxes = []
        self.predicted_labels = []
        self.predicted_scores = []
        self.true_boxes = []
        self.true_labels = []
        self.true_difficulties = []
        self.coco_ds = COCO()
        self.dataset = {'images': [], 'categories': [{'id': 0}, {'id': 1}], 'annotations': []}
        self.outputs = {}
        self.cpu_device = torch.device('cpu')

    def forward(self, images: List[torch.Tensor], boxes: torch.Tensor = None, num_boxes: List[int] = None, image_ids: List[str] = None, widths: List[int] = None, heights: List[int] = None):
        # print(images.shape)
        # image_list = [images[i] for i in range(images.shape[0])]
        image_list = images
        # image_list = list(image for image in images)
        if self.training:
            targets = []
            for i in range(len(image_list)):
                targets.append({'boxes': boxes[i,:num_boxes[i],:], 'labels': torch.tensor(1).long().view(1).repeat(num_boxes[i]).to(boxes.device)})
            loss_dict = self.model(image_list, targets)
            sys.stdout.flush()
            outputs = {'loss': sum(loss for loss in loss_dict.values() if not torch.isnan(loss).item())}
        else:
            predictions = []
            for image, image_id, h, w in zip(image_list, image_ids, heights, widths):
                predictions.append(self.model([image])[0])
                print(image_id, predictions[-1])
            # predictions = self.model(image_list)
            # predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions]
            predicted_boxes = [pred['boxes'] for pred in predictions]
            predicted_labels = [pred['labels'] for pred in predictions]
            predicted_scores = [pred['scores'] for pred in predictions]
            """threshold_masks = [scores > 0.7 for scores in predicted_scores]
            predicted_boxes = [pred[mask,:] for mask, pred in zip(threshold_masks, predicted_boxes)]
            predicted_labels = [pred[mask] for mask, pred in zip(threshold_masks, predicted_labels)]
            predicted_scores = [pred[mask] for mask, pred in zip(threshold_masks, predicted_scores)]"""
            # predictions = [{'boxes': b, 'labels': l, 'scores': s} for b, l, s in zip(predicted_boxes, predicted_labels, predicted_scores)]
            print([b.shape for b in predicted_boxes])
            true_boxes = [boxes[i,:num_boxes[i],:] for i in range(boxes.shape[0])]
            true_labels = [torch.ones_like(true_boxes[i][:,0].long()) for i in range(len(true_boxes))]
            true_difficulties = [torch.zeros_like(true_boxes[i][:,0]).long().to(boxes.device) for i in range(len(true_boxes))]
            self.predicted_boxes += predicted_boxes
            self.predicted_labels += predicted_labels
            self.predicted_scores += predicted_scores
            self.true_boxes += true_boxes
            self.true_labels += true_labels
            self.true_difficulties += true_difficulties
            for i in range(len(image_list)):
                img_dict = {}
                img_dict['id'] = len(self.dataset['images'])
                img_dict['height'] = heights[i]
                img_dict['width'] = widths[i]
                self.dataset['images'].append(img_dict)
                for j in range(num_boxes[i]):
                    ann = {}
                    ann['image_id'] = img_dict['id']
                    ann['bbox'] = boxes[i,j]
                    ann['bbox'][2:] -= ann['bbox'][:2]
                    ann['bbox'] = ann['bbox'].tolist()
                    ann['category_id'] = 1
                    ann['area'] = (ann['bbox'][2]-ann['bbox'][0])*(ann['bbox'][3]-ann['bbox'][1])
                    ann['id'] = len(self.dataset['annotations'])
                    ann['iscrowd'] = 0
                    self.dataset['annotations'].append(ann)
                self.outputs[img_dict['id']] = {k: v.to(self.cpu_device) for k, v in predictions[i].items()}
            outputs = {'predictions': predictions, 'loss': torch.tensor(0.).to(boxes.device), 'gold': boxes, 'gold_num_boxes': num_boxes, 'image_id': image_ids}
        return outputs
    def get_metrics(self, reset: bool = False):
        metrics = {}
        if not self.training:
            # average_precisions, mAP = calculate_mAP(self.predicted_boxes, self.predicted_labels, self.predicted_scores, self.true_boxes, self.true_labels, self.true_difficulties)
            self.coco_ds = COCO()
            self.coco_ds.dataset = self.dataset
            self.coco_ds.createIndex()
            coco_evaluator = CocoEvaluator(self.coco_ds, ['bbox'])
            coco_evaluator.update(self.outputs)
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            stats = coco_evaluator.coco_eval['bbox'].stats
            mAP = stats[0]
            metrics = {'mAP': mAP, 'mAp@0.5': stats[1]}
            if reset:
                self.predicted_boxes = []
                self.predicted_labels = []
                self.predicted_scores = []
                self.true_boxes = []
                self.true_labels = []
                self.true_difficulties = []
                self.dataset = {'images': [], 'categories': [{'id': 0}, {'id': 1}], 'annotations': []}
                self.outputs = {}
        return metrics
