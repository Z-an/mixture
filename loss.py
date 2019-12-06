import torch
import torch.nn.functional as F

def bpr_loss(positive_predictions, negative_predictions, weights=None, mask=None):

    loss = (1.0 - F.sigmoid((positive_predictions -
                             negative_predictions)))

    if weights is not None:
        loss = loss * weights

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

def hinge_loss(positive_predictions, negative_predictions, weights=None, mask=None):

    loss = torch.clamp(
        (negative_predictions - positive_predictions) + 1.0, 0.0)

    if weights is not None:
        loss = loss * weights

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

def adaptive_hinge_loss(positive_predictions, negative_predictions, weights=None, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closest to violating the
    ranking implicit in the pattern of user interactions.
    Approximates the idea of weighted approximate-rank pairwise loss
    introduced in [2]_
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    weights: tensor, optional
        Tensor containing weights.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    References
    ----------
    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """

    stacked_negative_predictions = torch.stack(negative_predictions, dim=0)
    highest_negative_predictions, _ = torch.max(stacked_negative_predictions, 0)

    return hinge_loss(
        positive_predictions,
        highest_negative_predictions.squeeze(),
        weights=weights,
        mask=mask
    )