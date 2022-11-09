import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
import json
import numpy as np
from model import EncoderDecoder

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)

def encode_data(data, vocab_to_index, seq_len, actions_to_index, targets_to_index, max_episode_len, max_instruction_len):
    n_episodes = len(data)
    n_actions = len(actions_to_index)
    n_targets = len(targets_to_index)

    # NOTE: need to have x include enough room for n_episodes*(len(episode)*seq_len)... not just n_episodes*seq_len
    # NOTE: len(episode) is around 40... maybe more like 33
    x = np.zeros((n_episodes, max_episode_len*max_instruction_len), dtype=np.int32)
    y = np.zeros((n_episodes, max_episode_len*2), dtype=np.int32)

    episode_idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for episode in data:
        inst_idx = 0
        action_target_idx = 0
        for i in range(len(episode)):
            command, actionTarget = episode[i]
            command = preprocess_string(command)
            action, target = actionTarget

            for word in command:
                if word not in vocab_to_index:
                    n_unks += 1
                    word = "<unk>"
                if inst_idx == 40*seq_len - 1:
                    n_early_cutoff += 1
                    break
                x[episode_idx, inst_idx] = vocab_to_index[word]
                inst_idx += 1
                n_tks += 1
            
            y[episode_idx, action_target_idx] = actions_to_index[action]
            action_target_idx += 1
            y[episode_idx, action_target_idx] = targets_to_index[target]
            action_target_idx += 1
        episode_idx += 1

    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(vocab_to_index))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d episodes without regard to order" % episode_idx)
    return x, y

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
    minibatch_size = 256

    args = vars(args)
    input_file = open(args['in_data_fn'])
    input_data = json.load(input_file)
    train_data = input_data['train']
    val_data = input_data['valid_seen']

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff, max_episode_len, max_instruction_len = build_tokenizer_table(train_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)

    # Encode the training and validation set inputs/outputs as sequences
    train_flattened = []
    for episode in train_data:
        episode_instructions = []
        episode_predictions = [["<start>", "<start>"]]
        for insts, outseq in episode:
            episode_instructions.append(insts)
            episode_predictions.append(outseq)
        episode_predictions.append(["<end>", "<end>"])
        train_flattened.append([episode_instructions, episode_predictions])

    val_flattened = []
    for episode in val_data:
        episode_instructions = []
        episode_predictions = [["<start>", "<start>"]]
        for insts, outseq in episode:
            episode_instructions.append(insts)
            episode_predictions.append(outseq)
        episode_predictions.append(["<end>", "<end>"])
        val_flattened.append([episode_instructions, episode_predictions])

    # At this point, the data is in the form of [[instruction paragraph, prediction paragraph]*num_episodes]
    # Need to encode it
    # Presumably want... different encodings for action vs target predictions
    # Presumably want... encoding to allow for episode_lenth*len_cutoff as the sequence length for inputs
    # Presumably want... encoding to allow for episode_length*2 as the sequence length for outputs

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_y = encode_data(train_flattened, vocab_to_index, len_cutoff, actions_to_index, targets_to_index, max_episode_len, max_instruction_len)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))

    val_np_x, val_np_y = encode_data(val_flattened, vocab_to_index, len_cutoff, actions_to_index, targets_to_index, max_episode_len, max_instruction_len)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))

    # NOTE: you want to preserve episodes: you want to have episode.length*[key = description] as an input sequence and episode.length[value = target-action] as a target sequence
    # NOTE: want to prepend BOS and append EOS to target sequence

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=minibatch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=minibatch_size)
    val_loader = None
    return train_loader, val_loader


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = EncoderDecoder()
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.01

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        output = model(inputs, labels)

        loss = criterion(output.squeeze(), labels[:, 0].long())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        em = output == labels
        prefix_em = prefix_em(output, labels)
        acc = 0.0

        # logging
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


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
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
