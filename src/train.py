# code based on from pytorch documentation  
# https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .evaluate import evaluate_model


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_function,device="cpu", epochs=5, scheduler=None):
    """
    Train a model over the specified number of epochs.

        Parameters:
            model (nn.Module): the model to train.
            train_dataloader (torch.utils.data.DataLoader): dataloader of the images to use for training.
            val_dataloader (torch.tils.data.DataLoader); dataloader of the images to use for validation.
            optimizer (torch.optim.Optimizer):  Pytorch optimizer to update the parameters of the model.
            loss_function (Loss): Pytorch loss function to quantify the difference between the model's predictions and the actual values.
            device (str): : to which device to transfer the model and data. Default:'cpu'.
            epochs (int): the number of epochs to train the model. Default: 5.
            scheduler (one of the schedulers from torch.optim.lr_scheduler or None): scheduler to adjust the learning rate based on the number of epochs. 

        Returns:
            dictionary (dict of str:list[float]): the metrics history over the epochs. It contains the following keys:
                train_loss_history: the train loss over the epochs
                val_loss_history: the validation loss over the epochs 
                train_accuracy_history: the train accuracy over the epochs
                val_accuracy_history: the validation accuracy over the epochs
                lrs: the learning rate over the epochs (only changes when a scheduler is specified)
    """
    model = model.to(device)

    train_loss_history = [] # keep track of the train loss over the epochs
    train_accuracy_history = [] # keep track of train accuracy over the epochs
    val_loss_history = [] # keep track of validation loss over the epochs
    val_accuracy_history =  [] # keep track of val accuracy over the epochs
    lrs=[] # keep track of learning rates over the epochs

    for epoch in range(epochs):
        print(f" Epoch {epoch+1} / {epochs}:")

        num_batches = len(train_dataloader)

        # Set the model to training mode - important for batch normalization and dropout layers 
        model.train()
        train_loss = 0
        true_labels = []
        predicted_labels = []
        threshold = 0.5 # probability threshold to decide whether target is in class 0 (<=threshold) or class 1 (>threshold)
        # convert threshold to logit value (by applying the inverse sigmoid to threshold) this way we do not have to apply the sigmoid to the predicted logits, 
        # but can apply the inverse sigmoid of the threshold directly to predict the label
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log() 
        for images, labels in tqdm(train_dataloader,desc="Training"):
            # loop over batches
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1) # add dimension
        
            # Compute prediction and loss
            optimizer.zero_grad()
            outputs = model(images) # raw logits
            loss = loss_function(outputs.float(),labels.float())
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            predicted_vals = (outputs > logit_threshold).long() # compute predicted labels and convert to 0 or 1
            train_loss += loss.item()
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted_vals.cpu().tolist())

        avg_train_loss = train_loss / num_batches
        train_accuracy = accuracy_score(true_labels,predicted_labels)
        print(f"  Average train loss:{avg_train_loss:>8f}  \n Accuracy on the train dataset: { train_accuracy:.2%}")


        train_loss_history.append(avg_train_loss)
        train_accuracy_history.append(train_accuracy)

        # validate
        val_metrics = evaluate_model(model, val_dataloader, loss_function)
        
        print(f"  Average val loss:{val_metrics['loss']:>8f}  \n Accuracy on the validation dataset: {val_metrics['accuracy']:.2%}")
        val_loss_history.append(val_metrics['loss'])
        val_accuracy_history.append(val_metrics['accuracy'])

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

    print("Done!")

    return {
        'train_loss_history':train_loss_history, 
        'train_accuracy_history':train_accuracy_history, 
        'val_loss_history':val_loss_history, 
        'val_accuracy_history':val_accuracy_history,
        'lrs':lrs
    }
