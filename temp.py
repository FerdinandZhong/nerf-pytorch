import torch
from vit_pytorch import ViT
from einops import rearrange, repeat
from transformer import Transformer
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ViT2(ViT):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0,
        emb_dropout=0
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.unflatten = nn.Unflatten(1, (image_size//patch_size, image_size//patch_size))
        self.basic_conv = BasicConv2d(dim, dim, 1)
        # self.final_conv = 

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        print(x.shape)

        x = x[:,1:,:]
        print(x.shape)

        x = self.unflatten(x)
        print(x.shape)

        return x


# v = ViT2(
#     image_size=256,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1,
#     pool="mean"
# )

# img = torch.randn(1, 3, 256, 256)

# preds = v(img) # (1, 1000)


to_patch_embedding = nn.Sequential(
    Rearrange('c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 32, p2 = 32),
    nn.Linear(3*32*32, 1024),
)

encoder = Transformer(
    depth=6, num_heads=1, embed_dim=1024, num_patches=64
)

img = torch.randn(3, 256, 256)

patches = to_patch_embedding(img)

print(patches.shape)

output = encoder(patches)
print(f"output shape: {output.shape}")