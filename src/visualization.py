import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from disease_localization_metrics import bbox_to_mask, normalize_cam, pointing_game, compute_iou, compute_dice

# # Using GradCAM from python_grad_cam instead
# class GradCAM: 
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.feature_maps = None
#         self.hook_layers()

#     def hook_layers(self):
#         def forward_hook(module, input, output):
#             self.feature_maps = output.detach()

#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0].detach()

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_backward_hook(backward_hook)

#     def generate_heatmap(self, input_tensor, target_classes):
#         heatmaps = []
#         output = self.model(input_tensor)  # Forward pass

#         for target_class in target_classes:
#             self.model.zero_grad()

#             # Compute gradient for this class
#             class_score = output[:, target_class]
#             class_score.backward(retain_graph=True)

#             # Compute weights
#             weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

#             # Compute Grad-CAM heatmap
#             heatmap = torch.sum(weights * self.feature_maps, dim=1).squeeze()
            
#             heatmap = torch.nn.functional.relu(heatmap) # as there is only 1.3% data belongs to Pneumonia. So the prediction trend to give low probability to it. instead of using relu, we visualize it with red color (not a good idea so keep the original).
            
#             # Normalize heatmap
#             epsilon = 1e-10
#             heatmap /= (torch.max(heatmap) + epsilon)

#             heatmaps.append(heatmap.cpu().numpy())

#         return np.array(heatmaps)  # Shape: (num_classes, H, W)

def draw_bbox(image_pil, bbox, color='red', ax=None):
    """
    image: np.array (H, W, C) or (H, W)
    bbox: (x, y, w, h)
    """
    x, y, w, h = bbox
    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(image_pil)
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./out/check_bbox.pdf", format='pdf', bbox_inches="tight")
        plt.show()
    else:
        ax.imshow(image_pil)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)


def visualize_resnet_activation_heatmaps(model, input_tensor, target_classes=None, image_id=None, layer=None, selected_feature_ids=None, alpha=0.6, target_class_labels=None, bbox_info=None, ori_img=None, save_fig=False):
    """
    Visualizes feature maps from a given layer of a ResNet model by overlaying them on the input image.
    
    Args:
        model (torchvision.models.ResNet): Pretrained ResNet model.
        image (PIL.Image or torch.Tensor): Input image.
        layer (torch.nn.Module, optional): Target layer to extract feature maps from. Defaults to second last conv layer.
        num_features (int): Number of feature maps to visualize (default: 16).
        alpha (float): Transparency of the heatmap overlay (default: 0.6).
    """
    model.eval()
    # # first denormalize the pixel_values
    # Create mean and std tensors and reshape them to (3, 1, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Denormalize: reverse the normalization
    denormalized_tensor = input_tensor.cpu().numpy() * std + mean
    denormalized_tensor = np.clip(denormalized_tensor, 0, 1)
    np_image = (denormalized_tensor * 255).astype(np.uint8)
    np_image = np.transpose(np_image, (1, 2, 0))

    image = Image.fromarray(np_image)

    # Default target layer (last conv layer of ResNet, just before the final full-connected layer)
    if layer is None:
        layer = model.resnet.layer4[-1]
    
    # Select target classes for visualization
    if target_classes is None:
        target_classes = [0, 3, 7]  # Example: Visualizing Grad-CAM for labels 0, 3, and 7

    visualization_mode = 'Grad_Cam' # "feature" for feature maps; or "Grad_Cam"
    
    # Generate Grad-CAM heatmap
    # grad_cam = GradCAM(model, layer) # This is for the self-defined Grad-CAM 
    # heatmaps = grad_cam.generate_heatmap(input_tensor.unsqueeze(0), target_classes) # This is for the self-defined Grad-CAM 
    # Using the GradCAM from pythorch_grad_cam instead
    cam = GradCAM(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=[ClassifierOutputTarget(x) for x in target_classes])
    heatmaps = [grayscale_cam[0, :]]

    if save_fig == False:
        # evaluate
        if bbox_info is not None:
            gt_mask = bbox_to_mask(bbox_info[1], heatmaps[0].shape, ori_img.size)
            normalize_heatmap = normalize_cam(heatmaps[0])
            hit = pointing_game(normalize_heatmap, gt_mask)
            iou = compute_iou(normalize_heatmap, gt_mask, threshold=0.3)
            dice = compute_dice(normalize_heatmap, gt_mask, threshold=0.3)
            print("Saving into:", f"./out/cam_heatmap_{bbox_info[0]}/" + f"generate_plot3_grad_cam_heatmap_{image_id}.pdf")
            print(hit, iou, dice)
            return hit, iou, dice
        return None

    # Convert original image to numpy array
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0  # Normalize to [0,1]

    # Plot feature maps overlaid on the original image
    rows = int(len(target_classes) ** 0.5)
    cols = (len(target_classes) + rows - 1) // rows + 1 # Ensure enough col for display
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    ax = axes.flat[0]
    if bbox_info is None:    
        ax.imshow(image_np)
        ax.set_title(f"original_{image_id}")
        ax.axis('off')
    else:
        draw_bbox(ori_img, bbox_info[1], ax=ax)
        
    # Display heatmaps for each selected class
    for i, heatmap in enumerate(heatmaps):
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.resize(heatmap_tensor, (224, 224)).squeeze(0).squeeze(0).numpy()
        # Convert feature map to heatmap
        heatmap = plt.cm.viridis(heatmap)[:, :, :3]  # Remove alpha channel if we apply relu to only count for positive activates
        # heatmap = plt.cm.seismic(heatmap)[:, :, :3]  # Remove alpha channel

        # Blend heatmap with original image
        overlay = (0.4 * image_np + 0.6 * heatmap)  # Alpha blending

        # Display image with overlay
        # ax = axes if len(heatmaps) == 1 else axes.flat[i+1]
        ax = axes.flat[i+1]
        ax.imshow(overlay)
        ax.set_title(f"Grad-CAM for {target_class_labels[i]}" if target_class_labels is not None else f"Grad-CAM for Class {target_classes[i]}")
        ax.axis('off')

    plt.tight_layout()
    path = f"./out/cam_heatmap_{bbox_info[0]}/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + f"generate_plot3_grad_cam_heatmap_{image_id}_newpackage.pdf", format='pdf', bbox_inches="tight")
    print("Saving into:", path + f"generate_plot3_grad_cam_heatmap_{image_id}_newpackage.pdf")
    plt.show()
    return None

def visualize_resnet_activation_heatmaps_batch(model, subset, layer=None, alpha=0.6, save_fig=False, class_labels=None):
# (model, input_tensor, target_classes=None, image_id=None, layer=None, selected_feature_ids=None, alpha=0.6, target_class_labels=None, bbox_info=None, ori_img=None, save_fig=True):
    """
    Visualizes feature maps from a given layer of a ResNet model by overlaying them on the input image.
    
    Args:
        model (torchvision.models.ResNet): Pretrained ResNet model.
        image (PIL.Image or torch.Tensor): Input image.
        layer (torch.nn.Module, optional): Target layer to extract feature maps from. Defaults to second last conv layer.
        num_features (int): Number of feature maps to visualize (default: 16).
        alpha (float): Transparency of the heatmap overlay (default: 0.6).
    """
    model.eval()
    # Default target layer (last conv layer of ResNet, just before the final full-connected layer)
    if layer is None:
        layer = model.resnet.layer4[-1]

    # # first denormalize the pixel_values
    # Create mean and std tensors and reshape them to (3, 1, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    input_tensors = subset["pixel_values"]
    # print(input_tensors.shape)
    # Denormalize: reverse the normalization
    images = []
    for i in range(input_tensors.shape[0]):
        denormalized_tensor = input_tensors[i].cpu().numpy() * std + mean
        denormalized_tensor = np.clip(denormalized_tensor, 0, 1)
        np_image = (denormalized_tensor * 255).astype(np.uint8)
        np_image = np.transpose(np_image, (1, 2, 0))
        images.append(Image.fromarray(np_image))
    # print(subset["target_class"])
    target_classes = [ClassifierOutputTarget(x) for x in subset["target_class"]]
    

    visualization_mode = 'Grad_Cam' # "feature" for feature maps; or "Grad_Cam"
    # Using the GradCAM from pythorch_grad_cam instead
    cam = GradCAM(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=input_tensors, targets=target_classes)
    heatmaps = grayscale_cam

    if save_fig == False:
        # evaluate
        bbox_info = subset["BBox"]
        image_ids = subset["Image Index"]
        target_classes = subset["target_class"]
        ori_imgs = subset["image"]
        # print(bbox_info)
        # print(image_ids)
        # print(target_classes)
        metric_results = {}
        for i in range(len(bbox_info)):    
            gt_mask = bbox_to_mask(bbox_info[i], heatmaps[i].shape, ori_imgs[i].size)
            normalize_heatmap = normalize_cam(heatmaps[i])
            hit = pointing_game(normalize_heatmap, gt_mask)
            iou = compute_iou(normalize_heatmap, gt_mask, threshold=0.3)
            dice = compute_dice(normalize_heatmap, gt_mask, threshold=0.3)
            gt_label = class_labels[target_classes[i]]
            if gt_label not in metric_results:
                metric_results[gt_label] = {"hit":[hit], "iou":[iou], "dice":[dice]}
            else:
                metric_results[gt_label]["hit"].append(hit)
                metric_results[gt_label]["iou"].append(iou)
                metric_results[gt_label]["dice"].append(dice)
            # print("Saving into:", f"./out/cam_heatmap_{class_labels[target_classes[i]]}/" + f"generate_plot3_grad_cam_heatmap_{image_ids[i]}_batchtest.pdf")
            # print(hit, iou, dice)
        return metric_results
        

    # # Convert original image to numpy array
    # image_resized = image.resize((224, 224))
    # image_np = np.array(image_resized) / 255.0  # Normalize to [0,1]

    # # Plot feature maps overlaid on the original image
    # rows = int(len(target_classes) ** 0.5)
    # cols = (len(target_classes) + rows - 1) // rows + 1 # Ensure enough col for display
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    # ax = axes.flat[0]
    # if bbox_info is None:    
    #     ax.imshow(image_np)
    #     ax.set_title(f"original_{image_id}")
    #     ax.axis('off')
    # else:
    #     draw_bbox(ori_img, bbox_info[1], ax=ax)
        
    # # Display heatmaps for each selected class
    # for i, heatmap in enumerate(heatmaps):
    #     heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    #     heatmap = F.resize(heatmap_tensor, (224, 224)).squeeze(0).squeeze(0).numpy()
    #     # Convert feature map to heatmap
    #     heatmap = plt.cm.viridis(heatmap)[:, :, :3]  # Remove alpha channel if we apply relu to only count for positive activates
    #     # heatmap = plt.cm.seismic(heatmap)[:, :, :3]  # Remove alpha channel

    #     # Blend heatmap with original image
    #     overlay = (0.4 * image_np + 0.6 * heatmap)  # Alpha blending

    #     # Display image with overlay
    #     # ax = axes if len(heatmaps) == 1 else axes.flat[i+1]
    #     ax = axes.flat[i+1]
    #     ax.imshow(overlay)
    #     ax.set_title(f"Grad-CAM for {target_class_labels[i]}" if target_class_labels is not None else f"Grad-CAM for Class {target_classes[i]}")
    #     ax.axis('off')

    # plt.tight_layout()
    # path = f"./out/cam_heatmap_{bbox_info[0]}/"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # plt.savefig(path + f"generate_plot3_grad_cam_heatmap_{image_id}_newpackage.pdf", format='pdf', bbox_inches="tight")
    # print("Saving into:", path + f"generate_plot3_grad_cam_heatmap_{image_id}_newpackage.pdf")
    # plt.show()
    return None

def visualize_resnet_feature_maps(model, input_tensor, layer=None, selected_feature_ids=None, alpha=0.6):
    """
    Visualizes feature maps from a given layer of a ResNet model by overlaying them on the input image.
    
    Args:
        model (torchvision.models.ResNet): Pretrained ResNet model.
        image (PIL.Image or torch.Tensor): Input image.
        layer (torch.nn.Module, optional): Target layer to extract feature maps from. Defaults to second last conv layer.
        num_features (int): Number of feature maps to visualize (default: 16).
        alpha (float): Transparency of the heatmap overlay (default: 0.6).
    """
    model.eval()

    # # Ensure image is a PIL Image
    # if isinstance(image, torch.Tensor):
    #     image = transforms.ToPILImage()(image)

    # # Preprocessing (ImageNet normalization)
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # # first denormalize the pixel_values
    # Create mean and std tensors and reshape them to (3, 1, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Denormalize: reverse the normalization
    denormalized_tensor = input_tensor.cpu().numpy() * std + mean
    denormalized_tensor = np.clip(denormalized_tensor, 0, 1)
    np_image = (denormalized_tensor * 255).astype(np.uint8)
    np_image = np.transpose(np_image, (1, 2, 0))

    image = Image.fromarray(np_image)

    # Default target layer (last conv layer of ResNet, just before the final full-connected layer)
    if layer is None:
        layer = model.resnet.layer4[-1]
    
    # Hook to extract feature maps
    feature_maps = None
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach().cpu().squeeze(0)

    # Register hook
    hook_handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor.unsqueeze(0))

    # Remove hook
    hook_handle.remove()
    print("feature maps", feature_maps.size())
    

    # Select first `num_features` feature maps
    # selected_feature_ids = [  14,  147,  165,  192,  373,  394,  503,  568,  676,  684,  706, 754,  779,  799,  886,  991,  993, 1088, 1156, 1158, 1170, 1208, 1240, 1278, 1284, 1362, 1422, 1494, 1698, 1762] # cluster 31
    selected_feature_ids = [  15,   53,   55,   67,   74,  138,  194,  196,  259,  307,  313, 391,  399,  407,  457,  504,  550,  584,  697,  699,  731,  750, 814,  850,  851,  913,  938,  956, 1183, 1200, 1241, 1272, 1354, 1366, 1450, 1519, 1563, 1659, 1717, 1760, 1770, 1827, 1863, 1908, 1940, 2024] # cluster 32
    if not selected_feature_ids:
        selected_features = feature_maps[:16]
    else:
        selected_features = feature_maps[selected_feature_ids]

    num_features = len(selected_features)
    print("The number of selected features is", num_features)

    # Convert original image to numpy array
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0  # Normalize to [0,1]

    # Plot feature maps overlaid on the original image
    cols = int(num_features ** 0.5)
    rows = (num_features + cols - 1) // cols  # Ensure enough rows for display
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i >= num_features:
            ax.axis('off')
            continue

        feature_map = selected_features[i]
        print(feature_map)

        # Normalize feature map
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

        # Resize using torchvision
        feature_map_resized = F.resize(feature_map.unsqueeze(0), (224, 224)).squeeze(0).numpy()

        # Convert feature map to heatmap
        heatmap = plt.cm.viridis(feature_map_resized)[:, :, :3]  # Remove alpha channel

        # Blend heatmap with original image
        overlay = (1 - alpha) * image_np + alpha * heatmap

        # Display image with overlay
        ax.imshow(overlay)
        ax.set_title(f'Feature {selected_feature_ids[i] if selected_feature_ids is not None else i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("generate_plot3_feature_heatmap.pdf", format='pdf', bbox_inches="tight")
    plt.show()
