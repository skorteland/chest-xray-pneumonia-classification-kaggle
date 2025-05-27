import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learningcurve(metrics_history):
    """
    Plot learning curves of train loss and validation loss and of train accuracy and validation/test accuracy.

        Parameters:
            metrics_history (dict): A dictionary containing the following keys:
            train_loss_history: history of train loss during the training.
            val_loss_history: history of validation loss during the training.
            train_accuracy_history: history of train accuracy during the training.
            val_accuracy_history: history of validation accuracy during the training.
    """
    fig, ax = plt.subplots(1, 2,figsize=(18,6))
    # plot the evolution of the losses over the epochs
    ax[0].plot(metrics_history['train_loss_history'], label='Train loss')
    ax[0].plot(metrics_history['val_loss_history'], label='Validation loss')
    ax[0].set_title('Loss at the end of each epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # plot the evolution of the accuracy over the epochs
    ax[1].plot(metrics_history['train_accuracy_history'], label='Training accuracy')
    ax[1].plot(metrics_history['val_accuracy_history'], label='Validation accuracy')
    ax[1].set_title('Accuracy at the end of each epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

def plot_confusionmatrix(cm):
    """
    Plot confusion matrix

        Parameters:
            cm (ndarray): confusion matrix.
    """
    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=["Normal", "Pneumonia"])
    disp.plot()

def plot_learningrate(lrs):
    """
    Plot learning rate over the epochs

        Parameters:
            lrs (list[float]): list containing the evolution of the learning rate over the epochs
    """

    # Visualize the confusion matrix
    fig = plt.figure(figsize=(6,3))
    plt.plot(lrs)
    plt.title("Learning rate over the epochs")
    plt.xlabel("Epoch")
    plt.ylabel("learning rate")