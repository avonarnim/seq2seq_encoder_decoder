# CSCI 499 HW 3: ALFRED Sequence-to-Sequence Encoder Decoder

## Implementation Choices

### Models

This assignment includes 3 model architectures: an LSTM-based encoder-decoder, an
attention-based encoder-decoder, and a transformer encoder-decoder. The first two models
use an EncoderDecoder wrapper class.

#### LSTM-based encoder-decoder

For the first model, the encoder embeds the input, then passes it through an LSTM. Only
the hidden state is returned to the EncoderDecoder wrapper.

The EncoderDecoder wrapper then iterates through the sequence length and performs decoding
for each of the expected labels in the batch. This process starts with creating a "seed"
set of "previous" labels, which are "<start>" for all of the batch's actions and
"<start>" for all of the batch's targets. The decoder embeds the "previous" labels, feeds
the embeddings through an LSTM, and runs the LSTM output through fully-connected layers
specific to actions and targets.

After this initial set of labels, the hidden state and cell state output of the LSTM is
used in the next iteration of the decoder. Additionally, the EncoderDecoder continues to
provide a set of "previous" labels by either feeding in the correct labels (in a
teacher-forcing mechanism) or by feeding in the most-recently predicted set of labels (in
a student-forcing mechanism).

#### Attention-based encoder-decoder

The second model was created by augmenting the first model's decoder with a linear-softmax
process. This is seen in the lines

```
attn_weights = F.softmax(self.attn(torch.cat((catEmbedding, h_0), 2)), dim=2)
attn_applied = torch.bmm(attn_weights, seed)

attn_output = torch.cat((catEmbedding, attn_applied), 2)
attn_output = self.attn_combination(attn_output)
```

which creates an input to feed into the final LSTM-FC layer process. The attention weights
calculation was based on the lecture notes, which indicate that a linear layer and softmax
are needed to operate on the hidden outputs of the encoder. The latter calculations were
based on the PyTorch tutorial for attention-based encoder-decoders.

#### Transformer-based encoder-decoder

The final model was created using the `simpletransformers` package, which has a
Seq2SeqModel class.

The evaluation process has the ability to generate metrics based on the input-output
pairs. I utilized a basic metric from the package tutorial that simply counted the number
of matches.

### Dataloader

The first two models take episodes of data and flatten the instruction-action/targets into
sequences and then form them into batches. This means that for a single episode, an input
sequence like `walk straight turn left to the counter pick up the knife on the table slice the green vegetable on the table` will map to output sequences `<start> GotoLocation PickupObject SliceObject <end` and `<start> diningtable knife lettuce <end>`. These
sequences are encoded to give a final batch data input shape of `batch_size*seq_len`.

The encoder-decoder pair of models in the first two architectures thus yield outputs for
actions of shape `batch_size*action_vocab_size` and for targets of shape
`batch_size*target_vocab_size`. These are then strung together over the sequence length to
yield final outputs from `EncoderDecoder` and `EncoderAttnDecoder` of shape
`batch_size*action_vocab_size*seq_len` and `batch_size*target_vocab_size*seq_len`.

The transformer model accepts pandas dataframes of shape

```
[
    ["input phrase", "target"],
    ["input phrase", "target"],
    ...
    ["input phrase", "target"]
]
```

for both training and evaluating. To predict for both actions and targets, I created
individual models and individual dataframes for actions and targets individually.

## Performance

The model and attention model indicate that learning is occuring, since the loss generally
decreases. That said, on occasion, the loss value will jump from single digits to a
massive value (several magnitudes larger than usual), which may indicate that the loss is
not very informative and is being over-corrected. Additionally, the values being
calculated for accuracy remain at 0, indicating that the model is bad at predicting the
prefix exactly.

The Transformer ran prohibitively slowly on my machine (likely due to memory constraints).
When limiting the number of episodes to ~400, the forward pass would occur in a few
minutes, but the loss propogation was taking ~40 seconds/iteration, for an estimated total
duration of 10+hours. For this reason, it was difficult to get an accurate estimate of how
it would compare to the basic encoder-decoder or attention encoder-decoder.
