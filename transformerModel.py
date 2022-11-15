from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd

class TransformerEncDec():
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        super(TransformerEncDec, self).__init__()
        model_args = Seq2SeqArgs()
        model_args.eval_batch_size = 64
        model_args.num_train_epochs = 2
        model_args.train_batch_size = 64
        self.model = Seq2SeqModel("roberta", "roberta-base", "bert-base-cased", use_cuda=False, args=model_args)

    def train(self, data):

        train_df = pd.DataFrame(data, columns=["input_text", "target_text"])
        self.model.train_model(train_df)
    

    def count_matches(labels, preds):
        return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

    def eval(self, data):

        eval_df = pd.DataFrame(data, columns=["input_text", "target_text"])
        result = self.model.eval_model(eval_df, matches=self.count_matches)
        print(result)

        return result
