# Arg Parser

import argparse
import torch

def get_args():
    """
    Parses command line arguments for the project.
    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='ML/DL Project Argument Parser')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'lenet', 'vit', 
                            'lenet_vit', 'resnet18_vit', 'resnet50_vit', 'resnet34_vit', 'resnet_vit', 'lenet_vit_late',
                            'mobilevit', 'coatnet'],
                        metavar='MODEL',
                        help='Model to use for training (default: vit)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to use for training and testing (default: cifar10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and testing (default: 64)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of subprocesses to use for data loading (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train the model (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for the optimizer (default: 0.9)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging training progress (default: 10)')
    parser.add_argument('--save_model', action='store_true',
                        help='Flag to save the trained model (default: False)')
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to save the trained model (default: model.pth)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--plot_accuracy', action='store_true',
                        help='Flag to plot accuracy during training (default: False)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model (default: None)')
    parser.add_argument('--parameters', action='store_true',
                        help='Flag to count the number of trainable parameters in the model (default: False)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)