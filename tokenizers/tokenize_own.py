import torch
import os, sys
import numpy as np
from pandas import read_csv, DataFrame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader

FOV = 50
model_path = r"D:\PULSE\results\auc check\lr_1e-4_auc_check-3D-20251009\0.8788_model.pth"
csv_dir = r"D:/DATA/LBxSF_labeled_segmented_radius.csv"
data_dir = rf"D:/DATA/own dataset pulse crops"
output_dir = rf"D:/PULSE/tokens"
csv_output_dir = rf"D:/PULSE/tokens/tokens.csv"


def get_tokens(model, x):
    """Extract CLS and patch tokens from Pulse3D model."""
    with torch.no_grad():
        # === same as model.forward() but stops before classification head ===
        conv_in = model.backbone.stem[0].in_channels
        if x.shape[1] == 1 and conv_in > 1:
            x = x.expand(-1, conv_in, -1, -1, -1)

        feat = model.backbone.stem(x)
        feat = model.backbone.layer1(feat)
        feat = model.backbone.layer2(feat)
        feat = model.backbone.layer3(feat)
        feat = model.backbone.layer4(feat)

        x_pooled = feat  # you can optionally pool to model.pool_size
        B, C, t_act, h_act, w_act = x_pooled.shape
        tokens = x_pooled.flatten(2).transpose(1, 2)  # (B, S, C)
        tokens = model.pos_dropout(tokens)
        tokens = tokens + model.pe_scale * model.pe_3d

        cls = model.cls_token.expand(B, -1, -1) + model.cls_pe
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens.permute(1, 0, 2)

        for layer in model.transformer_layers:
            tokens = layer(tokens)
        tokens = tokens.permute(1, 0, 2)

        cls_token = tokens[:, 0, :]     # (B, embed_dim)
        patch_tokens = tokens[:, 1:, :]  # (B, S, embed_dim)
        return cls_token, patch_tokens


def main():
    df = read_csv(csv_dir)

    data_loader = get_data_loader(
        data_dir,
        df,
        mode="3D",
        workers=4,
        batch_size=1,
        rotations=None,
        translations=None,
        size_mm=FOV,
        size_px=64,
    )

    # === Load model ===
    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    patient_ids, y_true = [], []
    cls_embeddings, patch_embeddings = [], []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)
            cls_token, patch_tokens = get_tokens(model, images)

            cls_np = cls_token.squeeze(0).cpu().numpy()
            patch_np = patch_tokens.squeeze(0).cpu().numpy()  # shape: (S, C)

            basename = os.path.basename(batch["path"][0])
            patient_id = basename.split("_")[0]
            patient_ids.append(patient_id)
            y_true.append(labels.item())

            # === Save embeddings ===
            np.save(os.path.join(output_dir, f"{patient_id}_cls.npy"), cls_np)
            np.save(os.path.join(output_dir, f"{patient_id}_patch.npy"), patch_np)

            cls_embeddings.append(cls_np)

            print(f"Extracted tokens for patient {patient_id}: CLS={cls_np.shape}, PATCH={patch_np.shape}")

    # === Save CLS embeddings summary CSV ===
    results_df = DataFrame({
        "patient_id": patient_ids,
        "true_label": y_true,
        "embedding_path": [os.path.join(output_dir, f"{pid}_cls.npy") for pid in patient_ids]
    })
    results_df.to_csv(csv_output_dir, index=False)

    print(f"\n✅ Token extraction complete.")
    print(f"CLS embeddings saved to: {output_dir}")
    print(f"Summary CSV: {csv_output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
