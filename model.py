# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)


    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))

        return lstm_out

class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, n_targets, n_actions, embedding_dim, hidden_dim):

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.fcTarget = nn.Linear(hidden_dim, n_targets)
        self.fcAction = nn.Linear(hidden_dim, n_actions)


    def forward(self, x, h_0):
        outTarget = self.fcTarget().squeeze(1)
        outAction = self.fcAction().squeeze(1)

        hidden_state = self.lstm([np.argmax(outTarget), np.argmax(outAction)], h_0)

        return outTarget, outAction


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, vocab_size, n_targets, n_actions, embedding_dim, hidden_dim):
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(n_targets, n_actions, embedding_dim, hidden_dim)

    def forward(self, x):
        h_0 = self.encoder(x)
        outTarget, outAction = self.decoder(np.zeros(len(x)), h_0)

        return outTarget, outAction
