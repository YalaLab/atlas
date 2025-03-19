from collections import OrderedDict
from sandstone.utils.loading import concat_all_gather
import torch


def prefix_dict(d, prefix):
    r = OrderedDict()
    for k, v in d.items():
        r[prefix + k] = v
    return r


def gather_predictions_dict(predictions, gather_keys=None):
    gathered_preds = {
        k: concat_all_gather(v) if isinstance(v, torch.Tensor) else v
        for k, v in predictions.items() if gather_keys is None or k in gather_keys
    }
    return gathered_preds


def gather_step_outputs(outputs):
    if len(outputs) == 0:
        return {}
    output_dict = OrderedDict()
    if isinstance(outputs[-1], list):
        outputs = outputs[0]

    for k in outputs[-1].keys():
        if k == "logs":
            output_dict[k] = gather_step_outputs([output[k] for output in outputs])
        elif (
            isinstance(outputs[-1][k], torch.Tensor) and len(outputs[-1][k].shape) == 0
        ):
            output_dict[k] = torch.stack(
                [output[k] for output in outputs if k in output]
            )
        elif isinstance(outputs[-1][k], torch.Tensor):
            output_dict[k] = torch.cat(
                [output[k] for output in outputs if k in output], dim=0
            )
        else:
            output_dict[k] = torch.tensor(
                [output[k] for output in outputs if k in output]
            )
    return output_dict


def get_weight_norm(model, norm="2"):
    """Calculate the weight norm for each parameter in the model."""
    return {
        f"weight_norm/{name}": torch.linalg.norm(param) / torch.numel(param)
        for name, param in model.named_parameters()
    }


def get_grad_norm(model, check_nan=False, log_weight_norm=False, norm=2):
    """Calculate gradient norms and optionally weight norms for model parameters."""
    if norm != 2:
        raise ValueError("Only l2-norm supported")

    grad_norm_dict = {}
    is_nan_dict = {}
    
    # Get parameters with gradients
    params_with_grad = [(name, param) for name, param in model.named_parameters() if param.grad is not None]
    
    if params_with_grad:
        # Calculate gradient norms
        grad_norm_dict = {
            f"grad_norm/{name}": param.grad.data.norm(norm)
            for name, param in params_with_grad
        }
        
        # Check for NaN gradients if required
        if check_nan:
            is_nan_dict = {
                name: param.grad for name, param in params_with_grad
                if torch.isnan(param.grad).any()
            }
        
        # Calculate total gradient norm
        grad_norm_dict["total_grad_norm"] = torch.tensor(list(grad_norm_dict.values())).norm(norm)
    
    # Add weight norms if required
    if log_weight_norm:
        grad_norm_dict.update(get_weight_norm(model))
    
    return grad_norm_dict, is_nan_dict


