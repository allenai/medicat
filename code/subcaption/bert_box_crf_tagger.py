from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from transformers import BertModel, BertConfig

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

@Model.register("bert_box_crf_tagger")
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
                 iou_threshold: float = 0.5,
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
        self.pos_layer = torch.nn.Linear(4, self.text_field_embedder.get_output_dim())

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions
        )
        self.binary_layer = torch.nn.Linear(self.text_field_embedder.get_output_dim(), 1)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(2*self.text_field_embedder.get_output_dim(), self.text_field_embedder.get_output_dim()),
                                              torch.nn.ReLU())

        self.subcaption_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                box: torch.Tensor,
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
        embedded_text_input = self.text_field_embedder(tokens)
        box_features = self.pos_layer(box)
        mask = util.get_text_field_mask(tokens)
        assert embedded_text_input.shape[1] == mask.shape[1]
        batch_size = mask.shape[0]

        embedded_text_input = self.output_layer(torch.cat((embedded_text_input, box_features.unsqueeze(1).repeat(1, embedded_text_input.shape[1], 1)), dim=2))
        embedded_text_input = self.dropout(embedded_text_input)

        logits = self.tag_projection_layer(embedded_text_input)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        batch_subcaptions = []
        for i in range(batch_size):
            pred_spans = []
            pred_start = None
            for j in range(len(predicted_tags[i])):
                pred_tag = self.vocab.get_token_from_index(predicted_tags[i][j], namespace="labels")
                if pred_tag[0] == "U":
                    pred_spans.append((j, j))
                if pred_start is None and pred_tag[0] == "B":
                    pred_start = j
                elif pred_start is not None and pred_tag[0] == "L":
                    pred_spans.append((pred_start, j))
                    pred_start = None
            subcaption = sorted([token for span in pred_spans for token in range(span[0], span[1]+1)])
            batch_subcaptions.append([subcaption])
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
            output["f1"] = []
            for i in range(batch_size):
                current_num_f1s = len(self.subcaption_metric.f1s)
                self.subcaption_metric(gold_subfigures=[[metadata[i]["gold_subfigure"]]], predicted_subfigures=[[metadata[i]["predicted_subfigure"]]] , gold_tokens=[[metadata[i]["gold_subcaption"]]], predicted_tokens=[batch_subcaptions[i]], common_wordpieces=[metadata[i]["common_wordpieces"]], wordpieces=[metadata[i]["words"]])
                assert len(self.subcaption_metric.f1s)-current_num_f1s == 1
                output["f1"].append(self.subcaption_metric.f1s[-1])

        if metadata is not None:
            output["predicted_subfigures"] = [[metadata[i]["predicted_subfigure"]] for i in range(batch_size)]
            output["predicted_subcaptions"] = batch_subcaptions
            output["common_wordpieces"] = [list(metadata[i]["common_wordpieces"]) for i in range(batch_size)]
            output["wordpieces"] = [metadata[i]["words"] for i in range(batch_size)]
            output["pdf_hash"] = [metadata[i]["image_id"].split("_")[0] for i in range(batch_size)]
            output["fig_uri"] = [metadata[i]["image_id"].split("_")[1] for i in range(batch_size)]
            if tags is None:
                output["gold_subfigures"] = [metadata[i]["gold_subfigure"] for i in range(batch_size)]
                output["gold_subcaptions"] = [metadata[i]["gold_subcaption"] for i in range(batch_size)]
            else:
                output["gold_subfigures"] = [[metadata[i]["gold_subfigure"]] for i in range(batch_size)]
                output["gold_subcaptions"] = [[metadata[i]["gold_subcaption"]] for i in range(batch_size)]
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
        return metrics_to_return
