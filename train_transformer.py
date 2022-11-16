import argparse
import json
from transformerModel import TransformerEncDec

from utils import (
    get_device,
)

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    args = vars(args)
    input_file = open(args['in_data_fn'])
    input_data = json.load(input_file)
    train_data = input_data['train']
    val_data = input_data['valid_seen']

    # Encode the training and validation set inputs/outputs as sequences
    train_action = []
    train_target = []
    for episode in train_data[:10]:
        for insts, outseq in episode:
            train_action.append([insts, outseq[0]])
            train_target.append([insts, outseq[1]])

    val_action = []
    val_target = []
    for episode in val_data[:10]:
        for insts, outseq in episode:
            val_action.append([insts, outseq[0]])
            val_target.append([insts, outseq[1]])
    return train_action, train_target, val_action, val_target

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_action, train_target, val_action, val_target = setup_dataloader(args)

    # build model
    actionModel = TransformerEncDec()

    actionModel.train(train_action)
    actionModel.eval(val_action)

    targetModel = TransformerEncDec()

    targetModel.train(train_target)
    targetModel.eval(val_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1, help="number of training epochs")

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
