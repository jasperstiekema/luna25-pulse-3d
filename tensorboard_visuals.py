import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import io
from torchvision.utils import make_grid

def log_all_conv_filters(model, writer, epoch, tag="Filters/Conv3d"):
    conv = model.backbone.stem[0]
    w = conv.weight.data.clone().cpu()  # shape: [F, C, D, H, W]
    # take central depth slice and only first channel if C>1
    slices = w[:, 0, w.shape[2]//2].unsqueeze(1)  # [F,1,H,W]
    grid = make_grid(slices, nrow=8, normalize=True, scale_each=True)
    writer.add_image(tag, grid, epoch)


def log_conv_filters(model, writer, epoch, tag="Filters/Conv3d"):
    """
    Logs the first conv layer's filters to TensorBoard.
    Assumes a 3D Conv layer in model.backbone.stem[0].
    """
    conv = model.backbone.stem[0]
    if isinstance(conv, torch.nn.Conv3d):
        weights = conv.weight.data.clone().cpu()
        num_filters = min(8, weights.shape[0])
        for i in range(num_filters):
            # middle depth slice
            slice_ = weights[i, 0, weights.shape[2] // 2]
            writer.add_image(f"{tag}_{i}", slice_.unsqueeze(0), epoch)


def register_activation_hook(model, layer_name="stem", store_dict=None):
    """
    Registers a forward hook to capture activations from a model layer.
    """
    def hook_fn(module, input, output):
        store_dict[layer_name] = output.detach().cpu()

    if store_dict is None:
        raise ValueError("Please pass a mutable dict to store activations.")
    
    layer = getattr(model.backbone, layer_name)
    layer.register_forward_hook(hook_fn)


def log_feature_maps(feat_map, writer, epoch, tag="Activations", max_channels=4):
    """
    Logs feature maps (3D) to TensorBoard.
    """
    B, C, D, H, W = feat_map.shape
    for i in range(min(C, max_channels)):
        slice_img = feat_map[0, i, D // 2]
        writer.add_image(f"{tag}/ch_{i}", slice_img.unsqueeze(0), epoch)


def plot_roc_curve(fpr, tpr):
    """
    Creates an ROC curve image (PIL) for TensorBoard.
    """
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label='ROC curve')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def log_roc_curve(writer, fpr, tpr, epoch, tag="Validation/ROC"):
    """
    Logs ROC curve image to TensorBoard.
    """
    img = plot_roc_curve(fpr, tpr)
    img_tensor = transforms.ToTensor()(img)
    writer.add_image(tag, img_tensor, epoch)


def log_prediction_samples(model, val_loader, writer, epoch, device, max_samples=4):
    """
    Logs a few prediction slices with label vs predicted score.
    """
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i > 0:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Extract metadata
            age = batch["age"].float().to(device)
            gender = batch["gender"].float().to(device)
            meta = torch.cat((age.unsqueeze(1), gender.unsqueeze(1)), dim=1)
            
            cls_logit = model(images)
            outputs = torch.sigmoid(cls_logit).squeeze(1).cpu().numpy()
            imgs = images.cpu().numpy()
            labels = labels.cpu().numpy()
            ages = age.cpu().numpy()
            genders = gender.cpu().numpy()

            for j in range(min(max_samples, imgs.shape[0])):
                mid_slice = imgs[j, 0, imgs.shape[2] // 2]
                fig, ax = plt.subplots()
                ax.imshow(mid_slice, cmap="gray")
                gender_str = "F" if genders[j] > 0.5 else "M"
                ax.set_title(f"GT: {int(labels[j][0])}, Pred: {outputs[j]:.2f}, Age: {ages[j]*100:.0f}, Gender: {gender_str}")
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = transforms.ToTensor()(img)
                writer.add_image(f"Predictions/Sample_{j}", img_tensor, epoch)
