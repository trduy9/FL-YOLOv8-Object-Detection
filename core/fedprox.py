import torch
from ultralytics.models.yolo.detect import DetectionTrainer

def load_global_model_weights(weights_path):
    """
    Loads parameter tensors from global model checkpoint to serve as w_global reference in FedProx.
    """
    if not weights_path:
        return {}
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            state_dict = ckpt['model'].state_dict()
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        # Detach and keep on CPU to avoid unnecessary GPU memory consumption
        return {k: v.detach().clone().cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    except Exception as e:
        print(f"Warning: Could not load global weights from {weights_path}: {e}")
        return {}


def make_fedprox_trainer_class(base_trainer_cls, global_state_dict, mu):
    """
    Creates a dynamic subclass of base_trainer_cls that applies Perturbed Gradient Descent (FedProx):
        grad = grad + mu * (w - w_global)
    before the optimizer step.

    Args:
        base_trainer_cls: The base detection trainer class (e.g. DetectionTrainer, GaussianDetectionTrainer).
        global_state_dict (dict): Reference global model parameters w_global at round start.
        mu (float): Proximal term penalty coefficient (mu >= 0).

    Returns:
        type: Subclass with FedProx optimizer_step override.
    """
    if mu <= 0 or not global_state_dict:
        return base_trainer_cls

    class FedProxDetectionTrainer(base_trainer_cls):
        def optimizer_step(self):
            """
            FedProx Perturbed Gradient Descent step:
            1. Unscale gradients from AMP.
            2. Add proximal penalty: grad = grad + mu * (param - global_param).
            3. Clip grad norm and step optimizer.
            """
            # 1. Unscale gradients first to operate on true gradients
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass

            # 2. Add proximal gradient perturbation: mu * (w - w_global)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Look up by direct name or stripped prefix if applicable
                    ref = None
                    if name in global_state_dict:
                        ref = global_state_dict[name]
                    elif name.startswith("model.") and name[6:] in global_state_dict:
                        ref = global_state_dict[name[6:]]
                    elif f"model.{name}" in global_state_dict:
                        ref = global_state_dict[f"model.{name}"]

                    if ref is not None and ref.shape == param.shape:
                        ref_device = ref.to(param.device, non_blocking=True)
                        param.grad.data.add_(param.data - ref_device, alpha=mu)

            # 3. Standard gradient clipping and optimizer update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    return FedProxDetectionTrainer
