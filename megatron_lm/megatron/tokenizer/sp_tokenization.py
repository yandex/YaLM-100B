import six
import sentencepiece as spm


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    return six.ensure_text(text, errors="ignore")


class SentencePieceTokenizer:
    NEW_LINE = "[NL]"
    UNK = 0
    BOS = 1
    EOS = 2
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "[MASK]"

    def __init__(self, vocab_file):
        self.name = "sp"
        self._tokenizer = spm.SentencePieceProcessor(model_file=vocab_file)
        self._vocab_words = self._get_vocab_words()
        self.encoder = {token: idx for idx, token in enumerate(self._vocab_words)}
        self.decoder = {idx: token for idx, token in enumerate(self._vocab_words)}

        mask_tokens = self.convert_tokens_to_ids([self.MASK_TOKEN])
        assert len(mask_tokens) == 1
        self.MASK = mask_tokens[0]

    def _encode(self, line, out_type=str):
        return self._tokenizer.encode(line, out_type=out_type)

    def tokenize(self, line, out_type=int):
        line = convert_to_unicode(line)
        line = line.replace("\n", SentencePieceTokenizer.NEW_LINE)
        return self._encode(line, out_type=out_type) # BOS will be added in another wrapper

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.piece_to_id(tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.decoder[idx] for idx in ids]

    def get_tokens(self):
        return self._vocab_words

    def _get_vocab_words(self):
        indices = list(range(self._tokenizer.GetPieceSize()))
        return self._tokenizer.id_to_piece(indices)

    @property
    def vocab(self):
        return self.encoder

    @property
    def inv_vocab(self):
        return self.decoder

    @property
    def vocab_size(self):
        return len(self.encoder)

    def detokenize(self, token_ids):
        tokens = [self.decoder[idx] for idx in token_ids]
        text = "".join(tokens).replace("\u2581", " ").replace(self.EOS_TOKEN, "").lstrip()
        return text

    @property
    def cls(self):
        return self.BOS

    @property
    def eod(self):
        return self.EOS

    @property
    def mask(self):
        return self.MASK
