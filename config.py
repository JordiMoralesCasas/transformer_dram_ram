import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="RAM")


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group("Glimpse Network Params")
glimpse_arg.add_argument(
    "--patch_size", type=int, default=8, help="size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_size", type=float, default=0.05, help="size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_scale", type=int, default=1, help="scale of successive patches"
)
glimpse_arg.add_argument(
    "--num_patches", type=int, default=1, help="# of downscaled patches per glimpse"
)
glimpse_arg.add_argument(
    "--loc_hidden", type=int, default=128, help="hidden size of loc fc"
)
glimpse_arg.add_argument(
    "--glimpse_hidden", type=int, default=128, help="hidden size of glimpse fc"
)


# core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--core_type", type=str, default="transformer",choices=["transformer", "rnn"], help="Type of core network to use."
)
core_arg.add_argument(
    "--transformer_model", type=str, default="gpt2",choices=["gpt2", "trxl", "gtrxl", "DRAMLM"], help="Type of Transformer architecture."
    )
core_arg.add_argument(
    "--num_glimpses", type=int, default=6, help="# of glimpses, i.e. BPTT iterations"
)
core_arg.add_argument(
    "--hidden_size", type=int, default=256, help="hidden size of rnn"
    )
core_arg.add_argument(
    "--cell_size", type=int, default=256, help="hidden size of the LSTM units."
    )
core_arg.add_argument(
    "--inner_size", type=int, default=1024*4, help="Size of inner fc layers in GPT2."
    )
core_arg.add_argument(
    "--n_heads", type=int, default=1, help="Size of inner fc layers in GPT2."
    )

# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default=0.05, help="gaussian policy standard deviation"
)
reinforce_arg.add_argument(
    "--epsilon_greedy",
    type=str2bool,
    default=False,
    help="Whether to follow an epsilon-greedy strategy for exploration-exploitation",
)
reinforce_arg.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)
reinforce_arg.add_argument(
    "--rl_loss_coef", type=float, default=0.01, help="Coeficient that weights the REINFORCE loss term."
)

# Inference (LM) params
inference_arg = add_argument_group("Inference Params")
inference_arg.add_argument(
    "--max_length", type=int, default=5, help="maximum answer length to be produced during inference."
)

# data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--task",
    type=str,
    default="mnist",
    choices=["mnist", "svhn"],
    help="Task to solve.",
)
data_arg.add_argument(
    "--preprocess",
    default="crop",
    help='What kind of preprocessing on the SVHN dataset. If set to "crop", crop images around the digit bounding boxes. If an integer N is provided, \
        the digits are randomly placed in an image of size NxN.',
)
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=128, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=True,
    help="Whether to shuffle the train and valid indices",
)
data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=False,
    help="Whether to visualize a sample grid of the data",
)
data_arg.add_argument(
    "--ignore_index",
    type=int,
    default=-100,
    help="Index to be ignored during loss computation",
)


# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=200, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=50,
    help="Number of epochs to wait before stopping train",
)
train_arg.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adamw"],
    help="What optimizer to use.",
)
train_arg.add_argument(
    "--weight_decay", type=float, default=0.01, help="Weight Decay parameter for AdamW."
)


# other params
misc_arg = add_argument_group("Misc.")
data_arg.add_argument(
    "--debug_run",
    type=str2bool,
    default=False,
    help="Run a DEBUG RUN.",
)
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=True, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--save_results", type=str2bool, default=False, help="Whether to save test results in a result file."
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./data", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="/data/users/jmorales/model_files/last",
    help="Directory in which to save model checkpoints",
)

misc_arg.add_argument(
    "--wandb_name",
    type=str,
    default=None,
    help="How to name the current run in WandB. Not specifying this parameters means that WandB is not used.",
)
misc_arg.add_argument(
    "--resume",
    type=str2bool,
    default=False,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=1, help="How frequently to plot glimpses"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
