from typing import Dict, List
import os
import random
import sys
import json

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from PIL import Image

from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_bert import BertLayerNorm, gelu

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from matching.visbert import VisBertModel

@Model.register('image_text_matching')
class ImageTextMatchingModel(Model):
    def __init__(self, vocab: Vocabulary, scibert_path: str, pretrained: bool = True, fusion_layer: int = 0, num_layers: int = None, image_root: str = None, full_matching: bool = False, retrieval_file: str = None, dropout: float = None, pretrained_bert: bool = False, tokens_namespace: str = "tokens", labels_namespace: str = "labels"):
        super().__init__(vocab)
        image_model = torchvision.models.resnet50(pretrained=pretrained)
        self.image_classifier_in_features = image_model.fc.in_features
        image_model.fc = torch.nn.Identity()
        self.image_feature_extractor = image_model
        config = BertConfig.from_json_file(os.path.join(scibert_path, 'config.json'))
        self.tokenizer = BertTokenizer(config=config, vocab_file=os.path.join(scibert_path, 'vocab.txt'))
        if dropout is not None:
            config.hidden_dropout_prob = dropout
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        num_visual_positions = 1
        self.bert = VisBertModel(config, self.image_classifier_in_features, fusion_layer, num_visual_positions)
        num_classifier_in_features = self.bert.config.hidden_size
        self.matching_classifier = torch.nn.Linear(num_classifier_in_features, 1)
        if pretrained_bert:
            state = torch.load(os.path.join(scibert_path, 'pytorch_model.bin'))
            filtered_state = {}
            for key in state:
                if key[:5] == 'bert.':
                    filtered_state[key[5:]] = state[key]
            self.bert.load_state_dict(filtered_state, strict=False)
        self.mode = "binary_matching"
        self.max_sequence_length = 512
        self.head_input_feature_dim = self.bert.config.hidden_size
        self._tokens_namespace = tokens_namespace

        if full_matching:
            expected_img_size = 224
            self.image_transform = transforms.Compose([
                                        transforms.Resize(expected_img_size),
                                        transforms.CenterCrop(expected_img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.mode = "full_matching"
            self.images = []
            self.image_id_index_map = {}
            f = open(retrieval_file)
            lines = f.readlines()
            for line in lines:
                fname = json.loads(line)['image_id']
                img = Image.open(os.path.join(image_root, fname)).convert('RGB')
                self.images.append(self.image_transform(img))
                self.image_id_index_map[fname] = len(self.images)-1
            self.images = torch.stack(self.images)
            self.top5_accuracy = CategoricalAccuracy(top_k=5)
            self.top10_accuracy = CategoricalAccuracy(top_k=10)
            self.top20_accuracy = CategoricalAccuracy(top_k=20)

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self,
            token_ids: Dict[str, torch.Tensor],
            segment_ids: torch.Tensor,
            images: torch.Tensor,
            image_ids: List[str],
            mask: torch.Tensor = None,
            matching_labels: torch.Tensor = None,
            labels: Dict[str, torch.Tensor] = None,
            labels_text: Dict[str, torch.Tensor] = None,
            words_tokens_map: torch.Tensor = None,
            words_to_mask: torch.Tensor = None,
            tokens_to_mask: torch.Tensor = None,
            question: List[str]= None,
            answer: List[str] = None,
            category: torch.Tensor = None,
            ):
        original_token_ids = token_ids
        if mask is not None:
            input_mask = mask
        else:
            input_mask = util.get_text_field_mask(token_ids)
        token_ids = token_ids[self._tokens_namespace]
        batch_size = min(token_ids.shape[0], images.shape[0])
        if images.shape[0] > token_ids.shape[0]:
            repeat_num = images.shape[0] // token_ids.shape[0]
            token_ids = token_ids.repeat(1, repeat_num).view(-1, token_ids.shape[-1])
            input_mask = input_mask.repeat(1, repeat_num).view(-1, input_mask.shape[-1])
            segment_ids = segment_ids.repeat(1, repeat_num).view(-1, segment_ids.shape[-1])
        elif token_ids.shape[0] > images.shape[0]:
            repeat_num = token_ids.shape[0] // images.shape[0]
            images = images.unsqueeze(1).repeat(1, repeat_num, 1, 1, 1).view(-1, images.shape[1], images.shape[2], images.shape[3])
        input_token_ids = token_ids
        visual_feats = self.image_feature_extractor(images)
        visual_feats = visual_feats.view(token_ids.shape[0], self.image_classifier_in_features)
        positions = torch.zeros((visual_feats.shape[0], 4)).to(visual_feats.device).float()
        visual_inputs = (positions, visual_feats)
        sequence_encodings, joint_representation = self.bert(input_ids=input_token_ids, attention_mask=input_mask, token_type_ids=segment_ids, visual_inputs=visual_inputs)
        outputs = {'loss': torch.tensor(0.)}
        if self.mode == "binary_matching":
            joint_representation = joint_representation.view(batch_size, -1, joint_representation.shape[-1])
            match_predictions = self.matching_classifier(joint_representation).view(batch_size, -1)
            if labels is not None:
                loss = self.loss(match_predictions, labels)
                outputs['loss'] = loss
                self.accuracy(match_predictions, labels)
        if self.mode == "full_matching":
            with torch.no_grad():
                predictions = torch.zeros((batch_size, self.images.shape[0])).to(images.device)
                batch = []
                batch_image_ids = []
                finished_images = set()
                sub_batch_size = 100
                for i in range(0, self.images.shape[0], sub_batch_size):
                    batch_images = self.images[i:i+sub_batch_size].to(images.device)
                    visual_feats = self.image_feature_extractor(batch_images)
                    visual_feats = visual_feats.view(-1, self.image_classifier_in_features).repeat(batch_size, 1)
                    positions = torch.zeros((visual_feats.shape[0], 4)).to(visual_feats.device).float()
                    visual_inputs = (positions, visual_feats)
                    batch_input_token_ids = input_token_ids.repeat(1, batch_images.shape[0]).view(-1, input_token_ids.shape[-1])
                    batch_input_mask = input_mask.repeat(1, batch_images.shape[0]).view(-1, input_mask.shape[-1])
                    batch_segment_ids = segment_ids.repeat(1, batch_images.shape[0]).view(-1, segment_ids.shape[-1])
                    sequence_encodings, joint_representation = self.bert(input_ids=batch_input_token_ids, attention_mask=batch_input_mask, token_type_ids=batch_segment_ids, visual_inputs=visual_inputs)
                    joint_representation = joint_representation.view(batch_size, batch_images.shape[0], joint_representation.shape[-1])
                    match_predictions = self.matching_classifier(joint_representation).view(batch_size, batch_images.shape[0])
                    predictions[:,i:i+batch_images.shape[0]] = match_predictions
                labels = torch.Tensor([self.image_id_index_map[image_id] for image_id in image_ids]).long().to(images.device)
                outputs['loss'] = self.loss(predictions, labels)
                outputs['predictions'] = predictions
                outputs['image_ids'] = image_ids
                self.accuracy(predictions, labels)
                self.top5_accuracy(predictions, labels)
                self.top10_accuracy(predictions, labels)
                self.top20_accuracy(predictions, labels)
        return outputs

    def get_metrics(self, reset: bool = False):
        metrics = {'accuracy': self.accuracy.get_metric(reset)}
        if "full_matching" in self.mode:
            metrics["top5_accuracy"] = self.top5_accuracy.get_metric(reset)
            metrics["top10_accuracy"] = self.top10_accuracy.get_metric(reset)
            metrics["top20_accuracy"] = self.top20_accuracy.get_metric(reset)
        return metrics
