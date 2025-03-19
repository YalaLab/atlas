
from einops import rearrange
import torch
import torch.nn as nn
from sandstone.models.msa import StackMSA


class Atlas(nn.Module):
    def __init__(
        self,
        args,
        kwargs_list,
        embed_dim=384,
        num_classes=100,
        fc_norm=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        assert kwargs_list is not None, "kwargs_list should be provided"
        self.args = args
        self.kwargs_list = kwargs_list
        self.num_classes = num_classes

        atlas_models = [StackMSA(args=args, **kwargs) for kwargs in kwargs_list]
        self.atlas_models = nn.ModuleList(atlas_models)

        self.embed_dim = embed_dim
        self.readout = "avg"

        self.readout_norm = nn.LayerNorm(self.embed_dim)

        self.fc_norm = nn.LayerNorm(self.embed_dim) if fc_norm else nn.Identity()
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward_head(self, x, pre_logits: bool = False, scales=None):
        # collect features from relevant scales
        if scales is None:
            scales = range(len(x))

        feats = [x[i] for i in scales]
        feats_BNC = torch.cat(feats, dim=1)

        # normalize the inputs
        if isinstance(self.readout_norm, nn.LayerNorm):
            x_BNC = self.readout_norm(feats_BNC)
        else:
            x_BNC = feats_BNC

        if self.readout == "avg":
            x_BC = x_BNC.mean(dim=1)
        else:
            raise NotImplementedError
        x_BC = self.fc_norm(x_BC)
        return x_BC if pre_logits else self.fc(x_BC)


    def forward(self, x, batch=None):
        base_model = self.atlas_models[0]

        bsz = x.shape[0]
        x = base_model.forward_raw(x)
        layouts = base_model.multiscale_layout
        num_stages = len(self.atlas_models)

        for level, atlas_model in enumerate(self.atlas_models):
            is_last = len(x) == 1 or level == num_stages - 1
            grid_sizes = [layout["grid_size"] for layout in layouts]
            x = atlas_model.forward_features(
                x, grid_sizes=grid_sizes, process_readout=False
            )
            if not is_last:
                layouts = layouts[1:]
                x = x[1:]

        feats = [rearrange(scale, "(b nw) k c -> b (nw k) c", b=bsz) for scale in x]
        x = self.forward_head(feats)
        return {"logit": x}
