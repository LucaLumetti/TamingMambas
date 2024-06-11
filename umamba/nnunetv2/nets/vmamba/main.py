from vmamba import Backbone_VSSM
import torch

print("Inizio...")
model = Backbone_VSSM(
        in_chans=3,
        num_classes=2,
        # Copied from configs files of segmentation folder on github
        depths=[2, 2, 15, 2],
        dims=128,
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v3_noz",  # v05_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
        out_indices=(0,1),
        norm_layer="ln2d"
    )
#model.to("cuda")
x = torch.randn(1, 3, 256, 256)
out = model(x)
with open("output.txt", "w") as file:
    file.write(f"Length of out: {len(out)}\n")
    file.write(f"Shape of out[0]: {out[0].shape}\n")

