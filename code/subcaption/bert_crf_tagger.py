from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from transformers import BertModel

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure

from subcaption.subfigure_subcaption_metric import SubfigureSubcaptionAlignmentMetric

@Model.register("bert_crf_tagger")
class BertCrfTagger(Model):
    """
    The ``BertCrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 label_namespace: str = "labels",
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 pretrained: bool = True,
                 block_pixel_width: float = 50,
                 span_labels: bool = False,
                 iou_threshold: float = 0.5,
                 show_oracle_f1: bool = True,
                 dropout: float = 0.1,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        # self.num_tags = 6
        self._verbose_metrics = verbose_metrics
        self.dropout = torch.nn.Dropout(dropout)
        self.tag_projection_layer = TimeDistributed(
            Linear(self.text_field_embedder.get_output_dim(), self.num_tags)
        )
        self.block_pixel_width = block_pixel_width
        self.span_labels = span_labels

        if not pretrained:
            self.text_field_embedder._token_embedders['bert'].transformer_model = BertModel(config=self.text_field_embedder._token_embedders['bert'].transformer_model.config)

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions
        )

        self.subcaption_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
        self.oracle_single_span_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
        self.oracle_full_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
        self.show_oracle_f1 = show_oracle_f1

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        mask = util.get_text_field_mask(tokens)
        batch_size = mask.shape[0]
        embedded_text_input = self.text_field_embedder(tokens)

        embedded_text_input = self.dropout(embedded_text_input)

        logits = self.tag_projection_layer(embedded_text_input)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            if self.span_labels:
                batch_predicted_boxes = [metadata[i]["predicted_subfigures"] for i in range(batch_size)]
            else:
                batch_predicted_boxes = []
                batch_oracle_single_span_boxes = []
                batch_oracle_full_boxes = []
                for i in range(batch_size):
                    blocks = []
                    for box in metadata[i]["predicted_subfigures"]:
                        placed = False
                        for j in range(len(blocks)):
                            if any([abs(b[1]-box[1]) < self.block_pixel_width for b in blocks[j]]):
                                blocks[j].append(box)
                                placed = True
                                break
                        if not placed:
                            index = 0
                            while index < len(blocks):
                                if all([box[1] < b[1] for b in blocks[index]]):
                                    break
                                index += 1
                            blocks = blocks[:index]+[[box]]+blocks[index:]
                    blocks = [sorted(block, key=lambda x: x[0]) for block in blocks]
                    predicted_boxes = [box for block in blocks for box in block]
                    batch_predicted_boxes.append(predicted_boxes)
                    batch_oracle_single_span_boxes.append(predicted_boxes)
                    batch_oracle_full_boxes.append(predicted_boxes)
            batch_subcaptions = []
            batch_oracle_single_span_subcaptions = []
            batch_oracle_full_subcaptions = []
            for i in range(batch_size):
                pred_spans = []
                pred_span_labels = []
                pred_start = None
                for j in range(len(predicted_tags[i])):
                    if mask[i,j].long().item() == 0:
                        print("here")
                        if pred_start is not None:
                            pred_spans.append((pred_start, j-1))
                            pred_tag = self.vocab.get_token_from_index(predicted_tags[i][j-1], namespace="labels")
                            pred_span_labels.append(pred_tag[1:])
                            pred_start = None
                        break
                    pred_tag = self.vocab.get_token_from_index(predicted_tags[i][j], namespace="labels")
                    if pred_tag[0] == "U":
                        pred_spans.append((j, j))
                        pred_span_labels.append(pred_tag[1:])
                    if pred_start is None and pred_tag[0] == "B":
                        pred_start = j
                    elif pred_start is not None and pred_tag[0] == "L":
                        pred_spans.append((pred_start, j))
                        pred_span_labels.append(pred_tag[1:])
                        pred_start = None
                subcaptions = []
                for span in pred_spans:
                    subcaption = list(range(span[0], span[1]+1))
                    subcaptions.append(subcaption)
                if len(subcaptions) < len(batch_predicted_boxes[i]):
                    if len(subcaptions) > 0:
                        subcaptions += [subcaptions[-1] for _ in range(len(batch_predicted_boxes[i])-len(subcaptions))]
                    else:
                        batch_predicted_boxes[i] = []
                batch_subcaptions.append(subcaptions)
                if self.show_oracle_f1:
                    gold_start = None
                    gold_spans = []
                    for j in range(tags.shape[1]):
                        if mask[i,j].long().item() > 0:
                            tag = self.vocab.get_token_from_index(tags[i,j].item(), namespace="labels")
                            if tag == "U":
                                gold_spans.append((j, j))
                            if gold_start is None and tag[0] == "B":
                                gold_start = j
                            elif gold_start is not None and tag[0] == "L":
                                gold_spans.append((gold_start, j))
                                gold_start = None
                    oracle_single_span_subcaptions = []
                    for span in gold_spans:
                        subcaption = list(range(span[0], span[1]+1))
                        oracle_single_span_subcaptions.append(subcaption)
                    if len(oracle_single_span_subcaptions) < len(batch_oracle_single_span_boxes[i]):
                        if len(oracle_single_span_subcaptions) > 0:
                            oracle_single_span_subcaptions += [oracle_single_span_subcaptions[-1] for _ in range(len(batch_oracle_single_span_boxes[i])-len(oracle_single_span_subcaptions))]
                    batch_oracle_single_span_subcaptions.append(oracle_single_span_subcaptions)
                    oracle_full_subcaptions = sorted([list(subcaption) for subcaption in metadata[i]["gold_subcaptions"]])
                    if len(oracle_full_subcaptions) < len(batch_oracle_full_boxes[i]):
                        if len(oracle_full_subcaptions) > 0:
                            oracle_full_subcaptions += [oracle_full_subcaptions[-1] for _ in range(len(batch_oracle_full_boxes[i])-len(oracle_full_subcaptions))]
                    batch_oracle_full_subcaptions.append(oracle_full_subcaptions)
            # self.subcaption_metric(gold_subfigures=[metadata[i]["gold_subfigures"] for i in range(batch_size)], predicted_subfigures=batch_predicted_boxes, gold_tokens=[metadata[i]["gold_subcaptions"] for i in range(batch_size)], predicted_tokens=batch_subcaptions, common_wordpieces=[metadata[i]["common_wordpieces"] for i in range(batch_size)], wordpieces=[metadata[i]["words"] for i in range(batch_size)])
            for i in range(batch_size):
                self.subcaption_metric(gold_subfigures=[metadata[i]["gold_subfigures"]], predicted_subfigures=batch_predicted_boxes[i:i+1], gold_tokens=[metadata[i]["gold_subcaptions"]], predicted_tokens=batch_subcaptions[i:i+1], common_wordpieces=[metadata[i]["common_wordpieces"]], wordpieces=[metadata[i]["words"]])
            if self.show_oracle_f1:
                self.oracle_single_span_metric(gold_subfigures=[metadata[i]["gold_subfigures"] for i in range(batch_size)], predicted_subfigures=batch_oracle_single_span_boxes, gold_tokens=[metadata[i]["gold_subcaptions"] for i in range(batch_size)], predicted_tokens=batch_oracle_single_span_subcaptions, common_wordpieces=[metadata[i]["common_wordpieces"] for i in range(batch_size)], wordpieces=[metadata[i]["words"] for i in range(batch_size)])
                self.oracle_full_metric(gold_subfigures=[metadata[i]["gold_subfigures"] for i in range(batch_size)], predicted_subfigures=batch_oracle_full_boxes, gold_tokens=[metadata[i]["gold_subcaptions"] for i in range(batch_size)], predicted_tokens=batch_oracle_full_subcaptions, common_wordpieces=[metadata[i]["common_wordpieces"] for i in range(batch_size)], wordpieces=[metadata[i]["words"] for i in range(batch_size)])

        output["predicted_subcaptions"] = batch_subcaptions
        if metadata is not None:
            # output["metadata"] = metadata
            output["gold_subcaptions"] = [[sorted(subcaption) for subcaption in metadata[i]["gold_subcaptions"]] for i in range(batch_size)]
            output["gold_subfigures"] = [metadata[i]["gold_subfigures"] for i in range(batch_size)]
            output["predicted_subfigures"] = batch_predicted_boxes
            output["common_wordpieces"] = [list(metadata[i]["common_wordpieces"]) for i in range(batch_size)]
            output["wordpieces"] = [metadata[i]["words"] for i in range(batch_size)]
            output["pdf_hash"] = [metadata[i]["image_id"].split("_")[0] for i in range(batch_size)]
            output["fig_uri"] = [metadata[i]["image_id"].split("_")[1] for i in range(batch_size)]
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {"subcaption_f1": self.subcaption_metric.get_metric(reset=reset)}
        if self.show_oracle_f1:
            metrics_to_return["oracle_single_span_f1"] = self.oracle_single_span_metric.get_metric(reset=reset)
            metrics_to_return["oracle_full_f1"] = self.oracle_full_metric.get_metric(reset=reset)
        return metrics_to_return
