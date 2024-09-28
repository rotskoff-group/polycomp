# This file was created as a modification of a file from the ae6bdb948b1733a8c1bb862de8127c55c97e3074 commit
# of the neuraloperator package, from April 10, 2024 and recognition should go to the developers of
# that package. Please reference their paper on arxiv at 2010.08895.
# Insert Arxiv ID and github link

from pathlib import Path
import torch

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor


def load_custom_pt(
    data_name,
    batch_size,
    test_batch_sizes,
    test_train_split=0.1,
    data_path=Path(__file__).resolve().parent,
    train_resolution=32,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Load the polymer dataset"""
    data = torch.load(Path(data_path).joinpath(data_name).as_posix())

    # Take the dataset and split into test and train portions
    # then appropriately split input and output
    break_point = int(test_train_split * data["x"].shape[0])
    x_train = data["x"][:break_point].type(torch.float32).clone()
    y_train = data["y"][:break_point].type(torch.float32).clone()

    test_batch_size = test_batch_sizes.pop(0)

    x_test = data["x"][break_point:].type(torch.float32).clone()
    y_test = data["y"][break_point:].type(torch.float32).clone()
    del data

    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(
        x_train,
        y_train,
    )

    # Build the loaders in the format neuraloperator will want
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {train_resolution: test_loader}

    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding,
    )
    return train_loader, test_loaders, data_processor
