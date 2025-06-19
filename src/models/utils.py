import torch

def unfreeze_layers(model, layer_names_or_indices, prefix=None):
    """
    Unfreezes specific layers in a model, given their names or indices.

    Parameters:
        model (torch.nn.Module): the full model (e.g. MobileNetV2, ResNet).
        layer_names_or_indices (list[str or int]): list of layer names (e.g. 'layer4') or indices (e.g. 15) to unfreeze.
        prefix (str or None): the name of the attribute that contains the backbone (e.g. features'). If None, the model itself is treated as the backbone. Default: None
    """
    # mobilenetv2: backbone_prefixes = ('features',), head_prefixes = ('classifier',)
    # resnet: backbone prefixes = None, head_prefixes = ('fc',)
    # efficientnet: backbone_prefixes = ('features',), head_prefixes = ('classifier',)
    backbone = getattr(model, prefix) if prefix else model

    for key in layer_names_or_indices:
        # Try to access layer by index (e.g. backbone[15])
        try:
            layer = backbone[key] # Only works if backbone is nn.Sequential
        except (TypeError, IndexError, KeyError):
            # assume it's a named layer (e.g. backbone.layer4)
            try:
                layer = getattr(backbone, key)
            except AttributeError:
                raise ValueError(f"Layer '{key}' not found in the backbone '{prefix or 'model'}'.")
        
        if isinstance(layer,torch.nn.Module):
            for param in layer.parameters():
                param.requires_grad =True
        else:
            raise TypeError(f"The target '{key}'is not a torch.nn.Module")
    
def get_optimizer(
    model,
    backbone_lr=1e-4,
    head_lr=1e-3,
    backbone_prefixes=('features', 'backbone'),
    head_prefixes=('classifier', 'fc'),
    weight_decay=1e-4,
    optimizer_cls=torch.optim.AdamW
):
    """
    Create an optimizer with different learning rates for backbone and head.

    Parameters:
    - model: The PyTorch model.
    - backbone_lr: Learning rate for backbone.
    - head_lr: Learning rate for classifier/head layers.
    - backbone_prefixes: Names of modules considered part of the backbone.
    - head_prefixes: Names of modules considered part of the classifier/head.
    - weight_decay: L2 regularization strength.
    - optimizer_cls: Optimizer class (default: AdamW).

    Returns:
    - torch.optim.Optimizer
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Assign to head or backbone by prefix
        if any(name.startswith(prefix) for prefix in head_prefixes):
            head_params.append(param)
        elif any(name.startswith(prefix) for prefix in backbone_prefixes):
            backbone_params.append(param)
        else:
            # Default to backbone
            backbone_params.append(param)

    return optimizer_cls([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=weight_decay)