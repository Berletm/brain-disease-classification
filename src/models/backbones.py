from hydra.utils import instantiate
from torch import nn


def build_backbone(config):
    model = instantiate(
        config['network'],
        _execution_whitelist_=('torchvision.models.*',),
    )

    classifier_path = config['classifier']
    feature_dim = model.get_submodule(classifier_path).in_features

    model.set_submodule(classifier_path, nn.Identity())

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer in config['trainable_layers']:
        for parameter in model.get_submodule(layer).parameters():
            parameter.requires_grad = True

    return model, feature_dim
