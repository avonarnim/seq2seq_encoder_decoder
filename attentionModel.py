# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn as nn
import numpy as np
import torch


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)


    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embeds.view(-1, len(x), self.embedding_dim))

        return h_n

class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, n_actions, n_targets, a_embed_dim, t_embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.targetEmbedding = nn.Embedding(n_targets, t_embed_dim)
        self.actionEmbedding = nn.Embedding(n_actions, a_embed_dim)

        self.lstm = nn.LSTM(t_embed_dim+a_embed_dim, hidden_dim)

        self.fcTarget = nn.Linear(hidden_dim, n_targets)
        self.fcAction = nn.Linear(hidden_dim, n_actions)


    def forward(self, seed, h_0, c_0):

        actionEmbedding = self.actionEmbedding(seed[0])
        targetEmbedding = self.targetEmbedding(seed[1])

        cat = torch.cat((actionEmbedding, targetEmbedding), 2)

        lstm_out, (hidden_state, cell_state) = self.lstm(cat, (h_0, c_0))

        outAction = self.fcAction(lstm_out)
        outTarget = self.fcTarget(lstm_out)

        # outTarget = self.fcTarget(targetEmbedding).squeeze(1)
        # print("OUT TARGET", outTarget)
        # outAction = self.fcAction(actionEmbedding).squeeze(1)
        # print("OUT ACTIOn", outAction)

        # outputPair = torch.from_numpy(np.array([np.argmax(outTarget), np.argmax(outAction)]))

        # lstm_out, (hidden_state, c_n) = self.lstm(outputPair, h_0)

        return outAction, outTarget, hidden_state, cell_state


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, vocab_size, t2i, a2i, embedding_dim, hidden_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(len(a2i), len(t2i), 5, 40, hidden_dim)
        self.t2i = t2i
        self.a2i = a2i

    def forward(self, batch, labels, training):

        # BATCH 32x6270 = B*seq_len
        # HIDDEN 1x32x128 = B*hidden_dim
        # LABELS[:,idx] = B*2 (action, target)
        # SEEDS 2*1*32 = num_labels*1*B
        # DECODER EMBEDDING 1*32*5 + 1*32*40 = 1*B*action_embed_dim + 1*B*target_embed_dim
        # ACTION OUTPUTS 32*10 = B*action_vocab
        # TARGET OUTPUTS 32*82 = B*target_vocab

        action_outputs = torch.empty((len(batch), len(labels[0]), len(self.a2i)))
        target_outputs = torch.empty((len(batch), len(labels[0]), len(self.t2i)))
        print(action_outputs.shape, target_outputs.shape)

        # for each instruction, this creates an encoded hidden state
        # then, it passes it to the deocder which outputs the target and action
        h_0 = self.encoder(batch)
        c_0 = torch.zeros(1, len(batch), 128)

        seeds = torch.zeros((len(batch),2))

        for seq_idx in range(len(labels)):
            if seq_idx == 0:
                seeds = [torch.full((1, len(batch)), self.a2i["<start>"]), torch.full((1, len(batch)), self.t2i["<start>"])]
            else:
                if training:
                    seeds = [labels[:,seq_idx, 0], labels[:,seq_idx, 1]]
                    seeds[0] = seeds[0].resize(1, len(batch))
                    seeds[1] = seeds[1].resize(1, len(batch))
                else:
                    print("got here 2")
                    seeds = [torch.argmax(action_outputs[seq_idx-1]), torch.argmax(target_outputs[seq_idx-1])]
                    # seeds = [torch.from_numpy(np.array([np.argmax(outTarget)])), torch.from_numpy(np.array([np.argmax(outAction)]))]

            outAction, outTarget, h_0, c_0 = self.decoder(seeds, h_0, c_0)
            action_outputs[:, seq_idx, :] = outAction.squeeze()
            target_outputs[:, seq_idx, :] = outTarget.squeeze()

        return action_outputs, target_outputs
