"""Module for registering and retrieving components used in the Athena architecture."""

from athena.architectures import AthenaResnet, AthenaTransformer, AthenaViT
from athena.encoders._base_encoder import BaseEncoder
from athena.encoders.input_encoders import ActionEncoder, ActionTokenizer
from athena.encoders.output_encoders import ArcsinWinProbEncoder, WinProbEncoder
from athena.loss_functions import CrossEntropyLoss, HLGaussLoss

ARCHITECTURES = {
    "resnet": AthenaResnet,
    "transformer": AthenaTransformer,
    "vit": AthenaViT,
}

INPUT_ENCODERS = {
    "action": ActionEncoder,
    "action_tokenizer": ActionTokenizer,
}

OUTPUT_ENCODERS = {
    "win_prob": WinProbEncoder,
    "arcsin_win_prob": ArcsinWinProbEncoder,
}

LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,
    "hl_gauss": HLGaussLoss,
}


def get_model(cfg):
    """Retrieve a model class based on the configuration.

    Args:
        cfg (Config): Configuration object containing model settings.

    Returns:
        nn.Module: An instance of the model class specified in the configuration.
    """
    return ARCHITECTURES[cfg.architecture.type](cfg)


def get_input_encoder(cfg) -> "BaseEncoder":
    """Retrieve an input encoder class based on the configuration.

    Args:
        cfg (Config): Configuration object containing encoder settings.

    Returns:
        BaseEncoder: An instance of the input encoder class specified in the configuration.
    """
    return INPUT_ENCODERS[cfg.encoder.input_encoder.type](cfg)


def get_output_encoder(cfg):
    """Retrieve an output encoder class by name.

    Args:
        cfg (Config): Configuration object containing encoder settings.

    Returns:
        BaseEncoder: An instance of the output encoder class specified in the configuration.
    """
    return OUTPUT_ENCODERS[cfg.encoder.output_encoder.type](cfg)


def get_loss_function(cfg):
    """Retrieve a loss function class based on the configuration.

    Args:
        cfg (Config): Configuration object containing loss function settings.

    Returns:
        nn.Module: An instance of the loss function class specified in the configuration.
    """
    return LOSS_FUNCTIONS[cfg.loss_function.type](cfg)
