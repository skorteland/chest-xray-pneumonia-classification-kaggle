import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, roc_auc_score,recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(model, val_dataloader, loss_function, device="cpu"):
    """
    Evaluate model performance

    Parameters:
        model (nn.module): neural network model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): dataloader for the images which to evaluate the model with.
        loss_function (Loss): Pytorch loss function to quantify the difference between the model's predictions and the actual values.
        device (str): to which device to transfer the model and data. Default:'cpu'.

    Returns:
        a dictionary containing the following metrics:
            loss (float): The average loss over the provided dataset.
            accuracy (float): The fraction of correctly classified samples.
            auc (float): Area Under Curve score.
            precision (float): The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp is the number of false positives. 
            recall (float): The recall is the ratio tp / (tp +fn) where tp is the number of true positives and fn is the number of false negatives.
            f1 (float): the F1 score.
            classification report (str): A text report showing the main classification metrics.
            confusion matrix (ndarray of shape (n_classes,n_classes)): Confusion matrix.
    """
    model=model.to(device)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    num_batches = len(val_dataloader)
    
    running_loss = 0
    true_labels = []
    predicted_labels= []
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # Also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad = True
    threshold = 0.5 # probability threshold to decide whether target is in class 0 (<=threshold) or class 1 (>threshold)
    # convert threshold to logit value (by applying the inverse sigmoid to threshold) this way we do not have to apply the sigmoid to the predicted logits, 
    # but can apply the inverse sigmoid of the threshold directly to predict the label
    logit_threshold = torch.tensor(threshold / (1 - threshold)).log() 
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images) # raw logits
            loss = loss_function(outputs.float(),labels.float())
            running_loss += loss.item()
            predicted_vals = (outputs > logit_threshold).long() # compute predicted labels and convert to 0 or 1
            
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted_vals.cpu().tolist())
        
    avg_val_loss = running_loss / num_batches
    
    accuracy = accuracy_score(true_labels,predicted_labels)
    precision = precision_score(true_labels,predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)
    recall = recall_score(true_labels,predicted_labels)
    f1 = f1_score(true_labels,predicted_labels)
    report = classification_report(true_labels,predicted_labels)
    cm = confusion_matrix(true_labels,predicted_labels)
    return {
        'loss': avg_val_loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision':precision,
        'recall': recall,
        'f1':f1,
        'classification report':report,
        'confusion matrix': cm
    }