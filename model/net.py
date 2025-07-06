from model.resnet import *
from model.transformer import *
from model.lenet import *
from model.mobilevit import *
from model.coatnet import *


def get_model(model_name, num_classes=10):
    """
    Returns a model based on the model name.
    Args:
        model_name (str): Name of the model to return.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: The requested model.
    """
    
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        return ResNet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes=num_classes)
    elif model_name == 'lenet':
        return LeNet(num_classes=num_classes)
    elif model_name == 'vit':
        return ViT(num_classes=num_classes)
    elif model_name == 'lenet_vit':
        return LeNet(num_classes=num_classes, attn=True)
    elif model_name == 'resnet18_vit':
        return ResNet18(num_classes=num_classes, attn=True)
    elif model_name == 'resnet34_vit':
        return ResNet34(num_classes=num_classes, attn=True)
    elif model_name == 'resnet50_vit':
        return ResNet50(num_classes=num_classes, attn=True)
    elif model_name == 'resnet_vit':
        return ResNet_ViT(num_classes=num_classes)
    elif model_name ==  'lenet_vit_late':
        return LeNet_ViT(num_classes=num_classes)
    elif model_name == 'mobilevit':
        return mobilevit(num_classes=num_classes)
    elif model_name == 'coatnet':
        return coatnet(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")