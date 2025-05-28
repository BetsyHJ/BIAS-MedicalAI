import torch
import torch.nn.functional as F

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

        # def backward_hook(module, grad_input, grad_output):
        #     self.gradients = grad_output[0]  # keep graph

        self.target_layer.register_forward_hook(forward_hook)
        # self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class_idx):
        """
        input_tensor: [B, C, H, W]
        target_class_idx: [B] tensor of class indices
        """
        output = self.model(input_tensor)  # forward pass, shape: [B, num_classes]
        selected_scores = output[range(output.shape[0]), target_class_idx]  # shape: [B]

        # self.model.zero_grad() # adding this will erase the gradient from the main gradient, i.e., that related to cls
        # selected_scores.sum().backward(retain_graph=True, create_graph=True) # this will add gradient, later optimzer.step() will update the related parameters, instead we do below and hide the backward hook

        grads = torch.autograd.grad(
            selected_scores.sum(),
            self.feature_maps,
            retain_graph=True,
            create_graph=True
        )[0]

        # Compute weights: [B, C, 1, 1]
        # weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps: [B, H, W]
        cam = torch.relu((weights * self.feature_maps).sum(dim=1))  # shape: [B, H, W]
        
        # Upsampling using interpolate, following what pytorch_grad_cam do in their code
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.size()[-2:], mode='bilinear', align_corners=False)

        # Normalize CAM, using (2, 3) rather than (1, 2) as I use upsequeeze in upsampling
        cam_min = torch.amin(cam, dim=(2, 3), keepdim=True)
        cam_max = torch.amax(cam, dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        # cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

        return cam.squeeze(1)  # differentiable, usable in loss function

def soft_iou_loss(cam, mask, eps=1e-6):
    """
    cam: Tensor of shape [B, H, W] — normalized Grad-CAM
    mask: Tensor of shape [B, H, W] — ground truth mask (from BBox)
    """
    intersection = (cam * mask).sum(dim=(1, 2))
    union = cam.sum(dim=(1, 2)) + mask.sum(dim=(1, 2)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou  # loss = 1 - IoU