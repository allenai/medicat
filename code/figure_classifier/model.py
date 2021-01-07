from typing import Dict, List

import torch
import torchvision

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.training.metrics import F1Measure, CategoricalAccuracy

@Model.register("image_classifier")
class ImageClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 backbone: str,
                 dropout_prob: float = 0.3,
                 pretrained: bool = True,
                 label_namespace: str = "labels") -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self._label_namespace = label_namespace
        self._dropout_prob = dropout_prob
        if backbone == 'resnet101':
            self.model = torchvision.models.resnet101(pretrained=True)
            self.feature_dim = 2048
            self.model.fc = torch.nn.Sequential(
                                    torch.nn.Dropout(p=self._dropout_prob, inplace=False),
                                    torch.nn.Linear(self.feature_dim, self.vocab.get_vocab_size(self._label_namespace))
                            )
        else:
            assert backbone == 'vgg16'
            self.model = torchvision.models.vgg16(pretrained=True)
            backbone_classifier = list(self.model.classifier.children())
            backbone_classifier.pop()
            self.feature_dim = 4096
            backbone_classifier.append(torch.nn.Linear(self.feature_dim, self.vocab.get_vocab_size(self._label_namespace)))
            self.model.classifier = torch.nn.Sequential(*backbone_classifier)
            self._dropout_prob = 0.0

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(self.vocab.get_token_index('Medical images', self._label_namespace))
    def forward(self,
                image: torch.Tensor,
                image_id: List[str],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        predictions = self.model(image)

        outputs = {'predictions': predictions, 'image_id': image_id}
        if label is not None:
            loss = self.loss(predictions, label)
            outputs['loss'] = loss
            self.f1(predictions, label)
            self.accuracy(predictions, label)
        return outputs
    def get_metrics(self,
                    reset: bool) -> Dict[str, float]:
        precision, recall, f1 = self.f1.get_metric(reset)
        return {'medical_f1': f1,
                'medical_precision': precision,
                'medical_recall': recall,
                'accuracy': self.accuracy.get_metric(reset)}
