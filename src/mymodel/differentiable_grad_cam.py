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

    def __call__(self, input_tensor):
        """
        input_tensor: [B, C, H, W]
        returns: CAMs for all classes: [B, num_classes, H, W]
        """
        # Forward pass
        output = self.model(input_tensor)  # [B, num_classes]
        num_classes = output.shape[1]
        
        cams = []
        for class_idx in range(num_classes):
            # Get selected scores for current class
            selected_scores = output[:, class_idx]  # [B]

            grads = torch.autograd.grad(
                selected_scores.sum(),     # Sum over batch
                self.feature_maps,
                retain_graph=True,
                create_graph=True
            )[0]  # [B, C, H, W]

            weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            cam = torch.relu((weights * self.feature_maps).sum(dim=1, keepdim=True))  # [B, 1, H, W]

            # Upsample
            cam = F.interpolate(cam, size=input_tensor.size()[-2:], mode='bilinear', align_corners=False)  # [B, 1, H, W]

            cams.append(cam)

        # Stack over classes: list of [B, 1, H, W] → [B, C, H, W]
        cams = torch.cat(cams, dim=1)

        # Normalize CAMs per-class
        cam_min = cams.amin(dim=(2, 3), keepdim=True)
        cam_max = cams.amax(dim=(2, 3), keepdim=True)
        cams = (cams - cam_min) / (cam_max - cam_min + 1e-8)

        return cams  # [B, C, H, W]


    # def __call__(self, input_tensor, target_class_idx):
    #     """
    #     input_tensor: [B, C, H, W]
    #     target_class_idx: [B] tensor of class indices
    #     """
    #     output = self.model(input_tensor)  # forward pass, shape: [B, num_classes]
    #     selected_scores = output[range(output.shape[0]), target_class_idx]  # shape: [B]

    #     # self.model.zero_grad() # adding this will erase the gradient from the main gradient, i.e., that related to cls
    #     # selected_scores.sum().backward(retain_graph=True, create_graph=True) # this will add gradient, later optimzer.step() will update the related parameters, instead we do below and hide the backward hook

    #     grads = torch.autograd.grad(
    #         selected_scores.sum(),
    #         self.feature_maps,
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]

    #     # Compute weights: [B, C, 1, 1]
    #     # weights = self.gradients.mean(dim=(2, 3), keepdim=True)
    #     weights = grads.mean(dim=(2, 3), keepdim=True)

    #     # Weighted sum of feature maps: [B, H, W]
    #     cam = torch.relu((weights * self.feature_maps).sum(dim=1))  # shape: [B, H, W]
        
    #     # Upsampling using interpolate, following what pytorch_grad_cam do in their code
    #     cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.size()[-2:], mode='bilinear', align_corners=False)

    #     # Normalize CAM, using (2, 3) rather than (1, 2) as I use upsequeeze in upsampling
    #     cam_min = torch.amin(cam, dim=(2, 3), keepdim=True)
    #     cam_max = torch.amax(cam, dim=(2, 3), keepdim=True)
    #     cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    #     # cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    #     # cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

    #     return cam.squeeze(1)  # differentiable, usable in loss function

def soft_iou_loss(cam, mask, eps=1e-6):
    """
    cam: Tensor of shape [B, H, W] — normalized Grad-CAM
    mask: Tensor of shape [B, H, W] — ground truth mask (from BBox)
    """
    intersection = (cam * mask).sum(dim=(1, 2))
    union = cam.sum(dim=(1, 2)) + mask.sum(dim=(1, 2)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou  # loss = 1 - IoU

def background_suppression_loss(cam, mask):
    """
    cam: Tensor of shape [B, H, W] — normalized Grad-CAM
    mask: Tensor of shape [B, H, W] — ground truth mask (from BBox)
    """
    background_mask = 1.0 - mask.float()  # [B, H, W]
    background_activation = cam * background_mask  # suppress activation outside bbox
    bg_loss = F.relu(background_activation).mean()
    return bg_loss

def negative_cam_penalty(cams, target_labels, mask, margin=0.0):
    """
    cams: [B, C, H, W]
    target_labels: [B] (int64)
    mask: [B, H, W]
    """
    B, C, H, W = cams.shape

    # One-hot encode target labels
    target_mask = F.one_hot(target_labels, num_classes=C).permute(0, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
    not_target_mask = 1.0 - target_mask.float()  # [B, C, 1, 1]

    # Expand mask for all classes
    mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]

    # Penalize activations from non-target classes inside bbox mask
    masked_activation = cams * not_target_mask * mask_expanded  # [B, C, H, W]
    penalty = F.relu(masked_activation - margin).mean()

    return penalty

def negative_cam_penalty2(cams, target_labels, margin=0.0):
    """
    cams: [B, C, H, W]
    target_labels: [B] (int64)
    """
    B, C, H, W = cams.shape

    # One-hot encode target labels
    target_mask = F.one_hot(target_labels, num_classes=C).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
    not_target_mask = 1.0 - target_mask.float()  # [B, C, 1, 1]

    # Penalize activations from non-target classes inside bbox mask
    masked_activation = cams * not_target_mask  # [B, C, H, W]
    penalty = F.relu(masked_activation - margin).mean()

    return penalty