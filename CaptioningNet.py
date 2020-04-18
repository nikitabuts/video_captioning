from torch import nn


class CaptModel(nn.Module):
    def __init__(self, n_tokens, pad_ix,
                 emb_size=512, lstm_units=980,
                 cnn_feature_size=2048):


        super(self.__class__, self).__init__()

        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size, padding_idx=pad_ix)

        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions_ix):

        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        captions_emb = self.emb(captions_ix)

        state = (initial_cell[None], initial_hid[None])
        lstm_out, state = self.lstm(captions_emb, state)

        logits = self.logits(lstm_out)

        return logits