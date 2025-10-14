
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from timm.layers import DropPath


class TransformerBlock(nn.Module):
    """
    A single Transformer encoder layer with Pre-Norm, DropPath and LayerScale.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale1 = nn.Parameter(torch.ones(embed_dim) * 1e-5)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale2 = nn.Parameter(torch.ones(embed_dim) * 1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + self.drop_path1(self.layer_scale1.unsqueeze(0).unsqueeze(0) * attn_out)
        y2 = self.norm2(x)
        ff_out = self.ff(y2)
        x = x + self.drop_path2(self.layer_scale2.unsqueeze(0).unsqueeze(0) * ff_out)
        return x

class Pulse3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 1,
        pool_size=(8,4,4),  # target temporal/spatial grid
        dropout_prob: float = 0.1,
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        ff_ratio: int = 4,
        drop_path_rate: float = 0.2,
        freeze_bn: bool = False,
    ):
        super().__init__()
        # Backbone
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # # Adapt first conv for 1-channel input
        pretrained_first_conv = self.backbone.stem[0].weight.clone()
        self.backbone.stem[0] = nn.Conv3d(
            input_channels, 64,
            kernel_size=(3,7,7),
            stride=(1,2,2),
            padding=(1,3,3),
            bias=False
        )
        self.backbone.stem[0].weight.data = pretrained_first_conv.mean(dim=1, keepdim=True)
        # Remove head
        orig_in = self.backbone.fc.in_features
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Parameters
        self.embed_dim = orig_in
        ff_dim = self.embed_dim * ff_ratio
        self.pool_size = pool_size

        # Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        dpr = torch.linspace(0, drop_path_rate, num_transformer_layers).tolist()
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, ff_dim, dropout_prob, dpr[i])
            for i in range(num_transformer_layers)
        ])
        self.pos_dropout = nn.Dropout(dropout_prob)

        # MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.embed_dim, num_classes)
        )

        self.pe_3d = nn.Parameter(
            torch.randn(1, self.pool_size[0] * self.pool_size[1] * self.pool_size[2], self.embed_dim)
        )

        self.pe_scale = nn.Parameter(torch.tensor(0.1))
        self.cls_pe = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # Freeze BN
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        conv_in = self.backbone.stem[0].in_channels
        if x.shape[1] == 1 and conv_in > 1:
            x = x.expand(-1, conv_in, -1, -1, -1)
        # CNN feature
        feat = self.backbone.stem(x)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)  # (B, C, T', H', W')
        # print(feat.shape)
        # 1) Adaptive pooling to fixed grid
        # T_p, H_p, W_p = self.pool_size
        # x_pooled = nn.functional.adaptive_avg_pool3d(feat, (T_p, H_p, W_p))
        # B, C, t_act, h_act, w_act = x_pooled.shape
        x_pooled = feat
        B, C, t_act, h_act, w_act = x_pooled.shape
        
        # Flatten to tokens
        tokens = x_pooled.flatten(2).transpose(1, 2)  # (B, S, C)
        tokens = self.pos_dropout(tokens)

        pe = self.pe_3d
        tokens = tokens + self.pe_scale * pe

        # 3) CLS token + Transformer
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pe
        # cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens.permute(1, 0, 2)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = tokens.permute(1, 0, 2)

        # 4) Classify
        cls_out = tokens[:, 0, :]
        return self.head(cls_out)


################################################################################
# simple test
################################################################################
if __name__ == "__main__":
    model = Pulse3D()
    voxel = torch.randn(2, 1, 64, 64, 64)
    out = model(voxel)
    print("Output:", out.shape) 