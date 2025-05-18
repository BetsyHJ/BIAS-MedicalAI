import torch

class DifferentiableGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output  # keep graph

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # keep graph

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class_idx):
        """
        input_tensor: [B, C, H, W]
        target_class_idx: [B] tensor of class indices
        """
        output = self.model(input_tensor)  # forward pass, shape: [B, num_classes]
        selected_scores = output[range(output.shape[0]), target_class_idx]  # shape: [B]

        self.model.zero_grad()
        selected_scores.sum().backward(retain_graph=True)

        # Compute weights: [B, C, 1, 1]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps: [B, H, W]
        cam = torch.relu((weights * self.feature_maps).sum(dim=1))  # shape: [B, H, W]

        # Normalize CAM
        cam = cam - cam.min(dim=(1, 2), keepdim=True)[0]
        cam = cam / (cam.max(dim=(1, 2), keepdim=True)[0] + 1e-8)

        return cam  # differentiable, usable in loss function

def soft_iou_loss(cam, mask, eps=1e-6):
    """
    cam: Tensor of shape [B, 1, H, W] — normalized Grad-CAM
    mask: Tensor of shape [B, 1, H, W] — ground truth mask (from BBox)
    """
    intersection = (cam * mask).sum(dim=(2, 3))
    union = cam.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou  # loss = 1 - IoU