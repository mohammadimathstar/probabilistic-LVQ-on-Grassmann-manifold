import torch
from torch import nn, Tensor
import argparse
from typing import Callable
from sklearn.metrics import confusion_matrix, accuracy_score  # f1_score
from torch.nn import KLDivLoss, CrossEntropyLoss



def metrics(y_true: Tensor, y_pred: Tensor, nclasses):

    assert y_true.shape == y_pred.shape, f'their shape is labels: {y_true.shape}, pred:{y_pred.shape}'

    acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    c = confusion_matrix(
        y_true.cpu().numpy(), y_pred.cpu().numpy(),
        labels=range(nclasses),
        # normalize='true'
    )
    return 100 * acc, c




def compute_classification_metrics(y_true: Tensor, y_pred: Tensor, nclasses: int) -> tuple:
    """
    Compute classification accuracy and confusion matrix.

    Parameters:
    -----------
    y_true : Tensor
        True class labels, a tensor of shape (num_samples,).
    y_pred : Tensor
        Predicted class labels, a tensor of shape (num_samples,).
    nclasses : int
        Number of classes.

    Returns:
    --------
    tuple
        A tuple containing:
        - accuracy (float): Classification accuracy in percentage.
        - confusion_matrix (ndarray): Confusion matrix of shape (nclasses, nclasses).

    Raises:
    -------
    AssertionError
        If the shapes of `y_true` and `y_pred` do not match.
    """
    assert y_true.shape == y_pred.shape, f'Shape mismatch: labels: {y_true.shape}, predictions: {y_pred.shape}'

    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Compute accuracy
    accuracy = accuracy_score(y_true_np, y_pred_np)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=range(nclasses))

    return 100 * accuracy, conf_matrix


def smooth_labels(labels: Tensor, num_classes: int, epsilon: float = 0.1) -> Tensor:
    """
    Convert integer labels to smoothed probability distributions.

    Args:
        labels (torch.Tensor): Tensor of shape (batch_size,) containing class indices.
        num_classes (int): Total number of classes.
        epsilon (float): Smoothing factor (0 <= ε < 1).

    Returns:
        torch.Tensor: Smoothed label probabilities of shape (batch_size, num_classes).
    """
    assert 0 <= epsilon < 1, "epsilon should be between 0 and 1"

    batch_size = labels.size(0)
    
    # Start with all values = epsilon / (num_classes - 1)
    smooth = torch.full((batch_size, num_classes), epsilon / (num_classes - 1), device=labels.device)
    
    # Assign 1 - epsilon to the correct class
    smooth.scatter_(1, labels.unsqueeze(1).long(), 1.0 - epsilon)
    
    return smooth


class ReverseKLDivLoss(nn.Module):
    def __init__(self, reduction: str = 'batchmean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, log_probs: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:        
        p_hat = torch.softmax(log_probs, dim=1)
        p_hat = torch.clamp(p_hat, min=self.eps / soft_target.size(1))
        p_hat = p_hat / p_hat.sum(dim=1, keepdim=True)
        # p = torch.clamp(target, min=self.eps)
        loss = p_hat * (torch.log(p_hat) - torch.log(soft_target))
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'batchmean'
            return loss.sum() / log_probs.shape[0]


def kl_forward(u):
    return u * torch.log(u)

def Jensen_Shannon(u):
    return 0.5 * (u * torch.log(u) - (u + 1) * torch.log(0.5 *(1 + u)))

def SL(u):
    return torch.log(torch.tensor(2.0)) - torch.log(1 + u) 


class FDivergence(nn.Module):
    def __init__(self, reduction: str = 'batchmean', eps: float = 1e-5):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, log_probs: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        p_hat = torch.softmax(log_probs, dim=1)
        p_hat = torch.clamp(p_hat, min=self.eps / soft_target.size(1))
        p_hat = p_hat / p_hat.sum(dim=1, keepdim=True)

        u = soft_target / p_hat
        # loss = p_hat * kl_forward(u)
        # loss = p_hat * Jensen_Shannon(u)
        loss = p_hat * SL(u)
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # 'batchmean'
            return loss.sum() / log_probs.shape[0]
        

def get_loss_fn(args: argparse.Namespace) -> Callable[[Tensor], Tensor]:
    """
    Get activation function by name.

    name: e.g., "relu", "leaky_relu", "elu", "selu", "tanh", "sigmoid"
    """
    if args.loss_fn == "ce":
        return CrossEntropyLoss()
    elif args.loss_fn == "kl":
        return KLDivLoss(reduction="batchmean")
    elif args.loss_fn == "reverse_kl":
        return ReverseKLDivLoss(reduction="batchmean", eps=args.epsilon)
    elif args.loss_fn == "f_divergence":
        return FDivergence(reduction='batchmean', eps=args.epsilon) #divergence_type='hellinger')
    else:
        raise ValueError(f"Unsupported activation function: {args.loss_fn}")