import logging
from typing import List, Tuple, Optional

from overrides import overrides
# from pytorch_transformers.tokenization_auto import AutoTokenizer
from transformers import AutoTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@Tokenizer.register("pretrained_transformer2")
class PretrainedTransformerTokenizer(Tokenizer):
    """
    A ``PretrainedTransformerTokenizer`` uses a model from HuggingFace's
    ``pytorch_transformers`` library to tokenize some input text.  This often means wordpieces
    (where ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is',
    'awesome']``), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.

    We take a model name as an input parameter, which we will pass to
    ``AutoTokenizer.from_pretrained``.

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  We try
        to be a little bit smart about defaults here - e.g., if your model name contains ``bert``,
        we by default add ``[CLS]`` at the beginning and ``[SEP]`` at the end.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 model_name: str,
                 do_lowercase: bool,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        if model_name.endswith("-cased") and do_lowercase:
            logger.warning("Your pretrained model appears to be cased, "
                           "but your tokenizer is lowercasing tokens.")
        elif model_name.endswith("-uncased") and not do_lowercase:
            logger.warning("Your pretrained model appears to be uncased, "
                           "but your tokenizer is not lowercasing tokens.")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lowercase)
        default_start_tokens, default_end_tokens = _guess_start_and_end_token_defaults(model_name)
        self._start_tokens = start_tokens if start_tokens is not None else default_start_tokens
        self._end_tokens = end_tokens if end_tokens is not None else default_end_tokens
        self._tokenizer_lowercases = self.tokenizer_lowercases(self._tokenizer)

    def tokenizer_lowercases(self, tokenizer) -> bool:
        # Huggingface tokenizers have different ways of remembering whether they lowercase or not. Detecting it
        # this way seems like the least brittle way to do it.
        tokenized = tokenizer.tokenize(
                    "A"
        )  # Use a single character that won't be cut into word pieces.
        detokenized = " ".join(tokenized)
        return "a" in detokenized

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # pytorch-transformers is dealing with the whitespace...
        token_strings = self._start_tokens + self._tokenizer.tokenize(text) + self._end_tokens
        tokens = self._tokenizer.tokenize(text)
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        offsets = self._estimate_character_indices(text, token_ids)
        offsets = [(0, 0)]+offsets+[(len(text), len(text))]
        assert len(offsets) == len(token_strings)
        return [Token(t, idx=o) for t, o in zip(token_strings, offsets)]

    def _estimate_character_indices(
        self, text: str, token_ids: List[int]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        The huggingface tokenizers produce tokens that may or may not be slices from the
        original text.  Differences arise from lowercasing, Unicode normalization, and other
        kinds of normalization, as well as special characters that are included to denote
        various situations, such as "##" in BERT for word pieces from the middle of a word, or
        "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.

        This code attempts to calculate character offsets while being tolerant to these
        differences. It scans through the text and the tokens in parallel, trying to match up
        positions in both. If it gets out of sync, it backs off to not adding any token
        indices, and attempts to catch back up afterwards. This procedure is approximate.
        Don't rely on precise results, especially in non-English languages that are far more
        affected by Unicode normalization.
        """

        token_texts = [
            sanitize_wordpiece(t) for t in self._tokenizer.convert_ids_to_tokens(token_ids)
        ]
        token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)
        if self._tokenizer_lowercases:
            text = text.lower()
            token_texts = [t.lower() for t in token_texts]

        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)

            # Did we not find it at all?
            if token_start_index < 0:
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue

            # Did we jump too far?
            non_whitespace_chars_skipped = sum(
                1 for c in text[text_index:token_start_index] if not c.isspace()
            )
            if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                # Too many skipped characters. Something is wrong. Ignore this token.
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            token_offsets[token_index] = (
                token_start_index,
                token_start_index + len(token_text),
            )
            text_index = token_start_index + len(token_text)
            token_index += 1
        return token_offsets

def _guess_start_and_end_token_defaults(model_name: str) -> Tuple[List[str], List[str]]:
    if 'bert' in model_name:
        return (['[CLS]'], ['[SEP]'])
    else:
        return ([], [])

def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    else:
        return wordpiece

