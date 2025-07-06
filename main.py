from model.net import *
from utils.args import *
from utils.dataset import *
from utils.train import *
from utils.metrics import *

def main():
    args = get_args()
    
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    device = args.device
    log_interval = args.log_interval
    save_model = args.save_model
    model_path = args.model_path

    train_loader = get_dataloader(args, train=True)
    test_loader = get_dataloader(args, train=False)
    
    num_classes = 10 if args.dataset == 'cifar10' else 200 if args.dataset == 'tiny-imagenet' else 1000
    if args.load_model:
        model = get_model(args.model, num_classes=num_classes)
        model = load_model(model, args.load_model, args.device)
    else:
        model = get_model(args.model, num_classes=num_classes)
        print('No pre-trained model loaded, using a new model {}.'.format(args.model))
    if args.parameters:
        num_params = count_parameters(model)
        print(f'The model has {num_params} trainable parameters.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train(model, train_loader, criterion, optimizer, device, epochs, log_interval=log_interval, plot_accuracy=args.plot_accuracy)

    test(model, test_loader, criterion, device)
    if save_model:
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
    else:
        print(r'Model not saved. Use --save_model to save the model.')
    
    
if __name__ == "__main__":
    main()

