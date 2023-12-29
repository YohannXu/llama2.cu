from sentencepiece import SentencePieceProcessor


class Tokenizer():
    def __init__(self, model_path):
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    def encode(self, s, bos, eos):
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t):
        return self.sp_model.decode(t)