from athena.architectures import AthenaMamba, AthenaResnet, AthenaTransformer, AthenaViT
from athena.encoders._base_encoder import BaseEncoder
from athena.encoders.input_encoders import ActionEncoder, ActionTokenizer
from athena.encoders.output_encoders import WinProbEncoder
from athena.loss_functions import CrossEntropyLoss, HLGaussLoss


ARCHITECTURES = {
    "mamba": AthenaMamba,
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
}

LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,
    "hl_gauss": HLGaussLoss,
}


def get_model(cfg):
    """
    Retrieve a model class based on the configuration.

    Args:
        cfg (Config): Configuration object containing model settings.
    Returns:
        nn.Module: An instance of the model class specified in the configuration.
    """
    return ARCHITECTURES[cfg.architecture.type](cfg)


def get_input_encoder(cfg) -> "BaseEncoder":
    """
    Retrieve an input encoder class based on the configuration.

    Args:
        cfg (Config): Configuration object containing encoder settings.

    Returns:
        BaseEncoder: An instance of the input encoder class specified in the configuration.
    """
    return INPUT_ENCODERS[cfg.encoder.input_encoder.type](cfg)


def get_output_encoder(name):
    """
    Retrieve an output encoder class by name.

    Args:
        name (str): The name of the output encoder.

    Returns:
        class: The output encoder class corresponding to the name.
    """
    return OUTPUT_ENCODERS.get(name.lower(), None)


def get_loss_function(cfg):
    """
    Retrieve a loss function class based on the configuration.

    Args:
        cfg (Config): Configuration object containing loss function settings.

    Returns:
        nn.Module: An instance of the loss function class specified in the configuration.
    """
    return LOSS_FUNCTIONS[cfg.loss_function.type](cfg)
