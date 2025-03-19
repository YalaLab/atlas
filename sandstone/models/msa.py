import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Optional, Union, Tuple
from timm.models.layers import DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp
from sandstone.models.attention import CrossWindowBlock


class AbstractModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class EarlyConvResidual(nn.Module):
    """
    per block design of "FasterViT : Hatamizadeh et al"
    """
    def __init__(self, dim, drop_path_rate=0.0, layer_scale_init_value=None, kernel_size=3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                dim, 
                dim, 
                kernel_size=kernel_size,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(dim, eps=1e-5),
            nn.GELU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                dim, 
                dim, 
                kernel_size=kernel_size,
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(dim, eps=1e-5)
        )
        
        self.has_layer_scale = layer_scale_init_value is not None
        if self.has_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()


    def forward(self, x):
        residual = x
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Apply layer scaling if enabled
        if self.has_layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        
        x = residual + self.drop_path(x)
        return x


class EarlyConvDS(nn.Module):
    """
    per block design of "FasterViT : Hatamizadeh et al"
    """
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.conv = nn.Conv2d(
                        in_channels=dim,
                        out_channels=dim_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    )

    def forward(self, x):
        return self.conv(self.norm(x))


class MultiConvEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        self.proj = nn.Identity()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, embed_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            EarlyConvResidual(embed_dim),
            EarlyConvResidual(embed_dim),
            EarlyConvResidual(embed_dim),
        )
        self.ds1 = EarlyConvDS(embed_dim)
        self.conv2 = nn.Sequential(
            EarlyConvResidual(2 * embed_dim),
            EarlyConvResidual(2 * embed_dim),
            EarlyConvResidual(2 * embed_dim),
        )
        self.ds2 = EarlyConvDS(2 * embed_dim)

        self.flatten = flatten

    def forward(self, x):
        x = x.squeeze(2)
        x = self.proj(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.ds1(x)
        x = self.conv2(x)
        x = self.ds2(x)
        if self.flatten:
            x = rearrange(x, "b c h w -> b (h w) c")
        return x


class RelativePositionEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.pos_emb = None
        self.projector = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=False),
        )

    def forward(self, input_tensor):
        input_dim = input_tensor.shape[1]

        ctxlen = int(input_dim**0.5)
        relative_coords_h = torch.arange(
            0, ctxlen, device=input_tensor.device, dtype=input_tensor.dtype
        )
        relative_coords_w = torch.arange(
            0, ctxlen, device=input_tensor.device, dtype=input_tensor.dtype
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .contiguous()
            .unsqueeze(0)
        )
        relative_coords_table -= ctxlen // 2
        relative_coords_table /= ctxlen // 2

        input_tensor = input_tensor + self.projector(
            relative_coords_table.flatten(2).transpose(1, 2))
        return input_tensor


class MultiScaleAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        sr_ratio=1,
        norm_layer=nn.LayerNorm,
        pool_op="max",
        merge_ratio=16,
        local2global=4,
        window_dims=4,
        window_size=8,
        weight_share=True,
        ignore_registers=False,
        accumulate_window_summary=True,
        multiscale_layout=None,
        **kwargs,
    ):
        super().__init__()
        self._init_basic_config(
            dim,
            num_heads,
            drop,
            attn_drop,
            qkv_bias,
            mlp_ratio,
            drop_path,
            merge_ratio,
            local2global,
            window_dims,
            init_values,
            norm_layer,
            weight_share,
            multiscale_layout,
        )

        self._init_multiscale_attention()
        self._init_multiscale_position_embeddings()

    def _init_basic_config(
        self,
        dim,
        num_heads,
        drop,
        attn_drop,
        qkv_bias,
        mlp_ratio,
        drop_path,
        merge_ratio,
        local2global,
        window_dims,
        init_values,
        norm_layer,
        weight_share,
        multiscale_layout,
    ):
        """Initialize basic configuration parameters."""
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**0.5
        self.merge_ratio = merge_ratio
        self.local2global = local2global
        self.window_dims = window_dims
        self.init_values = init_values
        self.norm_layer = norm_layer
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path

        # Dropout configurations
        self.attn_drop_p = attn_drop
        self.drop = drop
        self.proj_drop = nn.Dropout(drop)

        # Component configurations
        self.qkv_bias = qkv_bias
        self.additional_scale = None
        # self.num_tokens_per_window = math.prod(merge_ratio)
        # self.num_windows = math.prod(window_dims)
        self.communication_protocol = "all2all_sattn__sequential"

        # aggregate information from the lower to higher levels per block
        # currently supports : one2one_xattn, no_l2g
        self.aggregation_protocol = "one2one_xattn"
        self.multiscale_layout = multiscale_layout

        self.out_scales = {}
        self.cache_qkv = {}

        self.weight_share = weight_share

    def _init_multiscale_attention(self):
        """Initialize multiscale attention components, with one x-attn block per window."""
        self.blocks = nn.ModuleList(
            [
                CrossWindowBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop,
                    attn_drop=self.attn_drop_p,
                    drop_path=self.drop_path,
                    norm_layer=self.norm_layer,
                )
                for layout in self.multiscale_layout
            ]
        )

    def _init_multiscale_position_embeddings(self):
        """Initialize position embeddings.

        Args:
            num_scales (int): Number of different scale position embeddings to create.
        """
        self.posemb = nn.ModuleList(
            [
                RelativePositionEmbed(self.dim) for layout in self.multiscale_layout
            ]
        )

    def propagate_bottom_up(
        self,
        stages: List[torch.Tensor],
        grid_sizes: List[Tuple[int, int]],
        merge_ratio: int,
        local2global: int,
    ) -> List[torch.Tensor]:
        """
        Propagate information from local to global representations in bottom-up pass.
        
        Args:
            stages: List of tensors at different scales
            grid_sizes: List of grid sizes for each scale
            merge_ratio: Size of merging window
            downscaling_op: Pooling operator for downscaling
            
        Returns:
            Updated list of stages with propagated information
        """
        downscaling_op = nn.MaxPool2d(kernel_size=local2global)
        stages = stages.copy()  # Avoid modifying input

        for i in range(len(stages)-1):
            current_stage = stages[i]
            current_grid_size = grid_sizes[i]
            nw = current_grid_size[0] * current_grid_size[1]

            # Downscaling process
            current_stage = rearrange(
                current_stage,
                "bnw (m1 m2) c -> bnw c m1 m2",
                m1=merge_ratio, m2=merge_ratio
            )
            current_stage = downscaling_op(current_stage)
            
            # Spatial rearrangement
            current_stage = rearrange(
                current_stage,
                "(b nw) c m1 m2 -> b nw m1 m2 c",
                nw=nw
            )
            current_stage = rearrange(
                current_stage,
                "b (h w) m1 m2 c -> b (h m1) (w m2) c",
                h=current_grid_size[0], w=current_grid_size[1]
            )
            
            # Handle different spatial dimensions
            h, w = current_stage.shape[1:3]
            if h < merge_ratio or w < merge_ratio:
                local2global = rearrange(current_stage, "b h w c -> b (h w) c")
            else:
                local2global = rearrange(
                    current_stage,
                    "b (h m1) (w m2) c -> (b h w) (m1 m2) c",
                    m1=merge_ratio, m2=merge_ratio
                )
            
            stages[i+1] = stages[i+1] + local2global
        
        return stages


    def forward_sequential(
        self,
        scales: List[torch.Tensor],
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        
        merge_ratio = self.merge_ratio
        local2global = self.local2global
        self.num_scales = len(scales)

        # Add position embeddings to all stages
        if self.weight_share:
            # When weights are shared, use local embedding for all stages
            scales = [self.swin_local_pos_embed(scale) for scale in scales]
        else:
            # assume a separate pos-embed for each scale
            for idx in range(self.num_scales):
                # scales[idx] = self.posemb[idx](scales[idx], self.multiscale_layout[idx]["window_dims"])
                scales[idx] = self.posemb[idx](scales[idx])
        scales = self.propagate_bottom_up(scales, grid_sizes, merge_ratio, local2global)

        self.out_scales = {}

        # message passing from higher to lower level scales
        for S in range(self.num_scales - 1, -1, -1):
            x_S = scales[S]

            if "all2all_sattn" in self.communication_protocol:
                outs = self._process__sequential__all2all_sattn(x_S, S)
                if S in self.out_scales:
                    self.out_scales[S]["version"] += 1
                    self.out_scales[S]["tokens"] = outs
                else:
                    self.out_scales[S] = {"version": 1, "tokens": outs}
            else:
                raise NotImplementedError

        # # message passing from lower to higher level scales
        if self.aggregation_protocol != "nol2g":
            if self.aggregation_protocol == "one2one_xattn":
                fn = self._aggregate_one2one_xattn
            else:
                raise NotImplementedError

            for S in range(1, self.num_scales):
                outs = fn(S)
                self.out_scales[S]["version"] += 1
                self.out_scales[S]["tokens"] = outs

        # delete the cache and outscales
        out_scales = [self.out_scales[S]["tokens"] for S in range(self.num_scales)]
        self.out_scales = {}
        self.cache_qkv = {}
        return out_scales

    def forward(self, scales, grid_sizes=None):
        if "sequential" in self.communication_protocol:
            return self.forward_sequential(scales, grid_sizes)
        else:
            raise NotImplementedError

    def get_qkv(self, x_S, S, keys=["q", "kv"], update_cache=False):
        """
        implements a minimal QKV cache
        """
        # update if cache version and token version are different
        for key in keys:
            cache_idx = f"{S}-{key}"
            if cache_idx in self.cache_qkv:
                if (
                    self.cache_qkv[cache_idx]["version"]
                    != self.out_scales[S]["version"]
                ):
                    self.cache_qkv[cache_idx] = {
                        "tokens": self.blocks[S].get_qkv_tokens(x_S, key),
                        "version": self.out_scales[S]["version"],
                    }
            else:
                self.cache_qkv[cache_idx] = {
                    "tokens": self.blocks[S].get_qkv_tokens(x_S, key),
                    "version": 0,
                }

        qkv = []
        if "q" in keys:
            qkv.append(self.cache_qkv[f"{S}-q"]["tokens"])
        if "kv" in keys:
            qkv.extend(self.cache_qkv[f"{S}-kv"]["tokens"])
        return qkv

    def _aggregate_one2one_xattn(self, S):
        """
        Aggregate cross-attention from scale S to T.
        """
        x_S = self.out_scales[S]["tokens"]
        x_Sm1 = self.out_scales[S - 1]["tokens"]

        q_S = self.get_qkv(x_S, S, keys=["q"])[0]
        k_Sm1, v_Sm1 = self.get_qkv(x_Sm1, S - 1, keys=["kv"])

        ## assume Sm1 is B x H x H x C with window size KxK
        ## then Sm is B x [H/K K] [H/K K] x C -> [B x (H/K * H/K)] x K x K x C
        k1, k2 = self.multiscale_layout[S]['grid_size']
        m1 = int(math.sqrt(self.multiscale_layout[S]['window_size']))
        q_S = rearrange(q_S, "(b k1 k2) h (m1 m2) c -> b h (k1 m1) (k2 m2) c", k1=k1, k2=k2, m1=m1, m2=m1)
        s1, s2 = self.multiscale_layout[S-1]['grid_size']
        q_S = rearrange(q_S, "b h (s1 m1) (s2 m2) c -> (b s1 s2) h (m1 m2) c", s1=s1, s2=s2)
        m1 = int(math.sqrt(q_S.shape[2]))

        xattn_l2g = self.blocks[S].xattn_qkv(q_S, k_Sm1, v_Sm1)
        xattn_l2g = rearrange(xattn_l2g, "(b s1 s2) (m1 m2) c -> b (s1 m1) (s2 m2) c", s1=s1, s2=s2, m1=m1, m2=m1)
        xattn_l2g = rearrange(xattn_l2g, "b (k1 m1) (k2 m2) c -> (b k1 k2) (m1 m2) c", k1=k1, k2=k2)

        # xattn_l2g = rearrange(xattn_l2g, "(b k) n c -> b (k n) c", b=b)
        x_S = self.blocks[S].skip_with_drop(x_S, xattn_l2g)
        x_S = self.blocks[S].mlp_residual(x_S)

        return x_S

    def _process__sequential__all2all_sattn(self, x_S, S):
        # get the QKV for x_S
        q_S, k_S, v_S = self.get_qkv(x_S, S)

        k_Sp1, v_Sp1 = [k_S], [v_S]
        if len(self.out_scales) > 0:
            for T, out_t in self.out_scales.items():
                x_t = out_t["tokens"]
                num_repeats = x_S.shape[0] // x_t.shape[0]
                k_t, v_t = self.get_qkv(x_t, T, keys=["kv"])
                k_t = k_t.repeat_interleave(num_repeats, dim=0)
                v_t = v_t.repeat_interleave(num_repeats, dim=0)

                k_Sp1.append(k_t)
                v_Sp1.append(v_t)

        k_Sp1 = torch.cat(k_Sp1, dim=2)
        v_Sp1 = torch.cat(v_Sp1, dim=2)

        x_S = self.blocks[S].skip_with_drop(
            x_S, self.blocks[S].xattn_qkv(q_S, k_Sp1, v_Sp1)
        )
        x_S = self.blocks[S].mlp_residual(x_S)

        return x_S


class MultiScaleAttentionLayer(nn.Module):
    """
    MultiScaleAttentionLayer: A single layer of the MSA that processes
    input features through multiple attention blocks with window-based operations.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        input_resolution: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        only_local: bool = False,
        multiscale_layout=None,
        merge_ratio=8,
        local2global=4,
        **kwargs,
    ):
        """Initialize the MultiScaleAttentionLayer.

        Args:
            dim: Feature dimension size
            depth: Number of attention blocks in the layer
            input_resolution: Input spatial resolution
            num_heads: Number of attention heads
            window_size: Size of local attention windows
            conv: Whether to use convolution-based processing
            mlp_ratio: Expansion ratio for MLP hidden dimension
            qkv_bias: Enable bias terms in QKV projections
            qk_scale: Scaling factor for QK attention
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            only_local: Restrict to local attention only
        """
        super().__init__()

        # Basic configuration
        self.conv = conv
        self.window_size = window_size
        self.grid_size = input_resolution // window_size
        self.num_windows = self.grid_size**2

        # Calculate stride ratio for attention
        sr_ratio = input_resolution // window_size if not only_local else 1

        # Handle drop path scheduling
        if isinstance(drop_path, list):
            drop_path_rates = drop_path
        else:
            drop_path_rates = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                MultiScaleAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    input_resolution=input_resolution,
                    weight_share=False,
                    merge_ratio=merge_ratio,
                    local2global=local2global,
                    multiscale_layout=multiscale_layout,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, grid_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        # Process through attention blocks
        for block in self.blocks:
            x = block(x, grid_sizes)

        return x


class StackMSA(AbstractModel):
    """
    StackMSA: A multi-scale vision transformer architecture that processes
    images at multiple resolutions using a hierarchical structure.
    """

    def __init__(
        self,
        args,
        dim,
        in_dim,
        num_features,
        depths=8,
        window_size=8,
        mlp_ratio=4,
        num_heads=8,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer=MultiScaleAttentionLayer,
        img_size=[256, 256, 1],
        merge_ratio=8,
        local2global=4,
        patch_size=16,
        bsz=64,
        embed_op="convstem",
        **kwargs,
    ):
        super().__init__(args)
        # Model configuration
        self.num_classes = num_classes
        self.merge_ratio = merge_ratio
        self.local2global = local2global
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.num_features = num_features
        self.readout_norm = "batchnorm"
        self.depths = depths
        self.img_size = img_size
        self.embed_op = embed_op
        self.bsz = bsz

        # Layers initialization

        if self.embed_op == "convstem":
            self.patch_embed = MultiConvEmbed(
                in_chans=in_chans, embed_dim=self.num_features
            )
        else:
            self.patch_embed = None

        self.multiscale_layout = self.prepare_multiscale_layout(
            self.img_size, self.merge_ratio, self.local2global, self.patch_size
        )

        # Drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths)]

        # Initialize transformer layers
        self.levels = nn.ModuleList()
        layer_args = self._prepare_layer_args(
            kwargs["layer_args"].copy(),
            dim=dim,
            num_features=self.num_features,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dpr=dpr,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            multiscale_layout=self.multiscale_layout,
            merge_ratio=merge_ratio,
            local2global=local2global,
        )
        self.levels.append(layer(**layer_args))

        # Initialize normalization and classification head
        self._init_norm_and_head()


    def prepare_multiscale_layout(self, img_size, merge_ratio, local2global, patch_size):
        """
        given the input size, merge_ratio and local2global
        prepare the layout for multiscale attention and config 
        for the architecture
        """
        H, W, _ = img_size
        H = H // patch_size
        seqlen = H**2
        downsampling_ratio = local2global ** 2
        min_seqlen = merge_ratio ** 2
        kernel_size  = 1
        
        if seqlen <= min_seqlen:
            return [
                {
                    "grid_size": [1, 1],
                    "window_size": seqlen,
                    "num_windows": 1,
                    "seq_length": seqlen
                }
            ]
        num_stages = math.ceil((math.log2(seqlen/min_seqlen) / math.log2(downsampling_ratio))) + 1

        multiscale_layout = []

        for scale in range(num_stages):
            current_resolution = H // kernel_size
            
            if current_resolution <= merge_ratio:
                multiscale_layout.append({
                    "grid_size": [1, 1],
                    "window_size": current_resolution ** 2,
                    "num_windows": 1,
                    "seq_length": current_resolution ** 2
                })
                break
            else:
                grid_size = current_resolution // merge_ratio
                multiscale_layout.append({
                    "grid_size": [grid_size, grid_size],
                    "window_size": merge_ratio ** 2,
                    "num_windows": grid_size ** 2,
                    "seq_length": (grid_size ** 2) * (merge_ratio ** 2)
                })
            
            kernel_size *= local2global
            
        assert len(multiscale_layout) == num_stages
        return multiscale_layout

    def _prepare_layer_args(self, layer_args, **kwargs):
        """Prepare arguments for layer initialization."""
        layer_args.update(
            {
                "dim": kwargs["dim"],
                "depth": kwargs["depths"],
                "num_heads": kwargs["num_heads"],
                "window_size": kwargs["window_size"],
                "mlp_ratio": kwargs["mlp_ratio"],
                "drop_path": kwargs["dpr"],
                "qkv_bias": kwargs["qkv_bias"],
                "qk_scale": kwargs["qk_scale"],
                "drop": kwargs["drop_rate"],
                "attn_drop": kwargs["attn_drop_rate"],
                "transformer_blocks": False,
                "conv": False,
                "only_local": False,
                "multiscale_layout": self.multiscale_layout,
                "merge_ratio": kwargs["merge_ratio"],
                "local2global": kwargs["local2global"],
                "input_resolution": layer_args["input_resolution"] // 4,
            }
        )
        return layer_args

    def _init_norm_and_head(self):
        """Initialize normalization layers and classification head."""
        if self.readout_norm == "batchnorm":
            self.norm = nn.BatchNorm2d(self.dim)
        elif self.readout_norm == "layernorm":
            self.norm = LayerNorm2d(self.dim)
        else:  # flatln
            self.norm = nn.LayerNorm(self.dim)

        self.avgpool = (
            nn.AdaptiveAvgPool2d(1) if self.readout_norm != "flatln" else None
        )
        self.head = (
            nn.Linear(self.dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

    def forward_raw(self, x, inflate=True):
        """Process input through the raw forward pass."""
        x = self.patch_embed(x)
        stages = self._build_multiscale_tokens(x)
        return stages


    def _build_multiscale_tokens(self, x_BCHW):
        """
        Build tokens for all scales

        Assuming inputs from the patch embed to be x_BCHW,
        build tokens at all scales using maxpool with
        progressively larger window sizes (aka downscaling)
        """
        seqlen = x_BCHW.shape[2] * x_BCHW.shape[3]
        downsampling_ratio = self.local2global**2
        kernel_size = 1

        min_seqlen = self.merge_ratio**2

        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        stages = []
        self.grid_sizes = []

        for scale in range(num_stages):
            local2global_op = nn.MaxPool2d(kernel_size=kernel_size)
            x_scale_BCHW = local2global_op(x_BCHW)
            kernel_size *= self.local2global

            # check if windowing is possible
            b, c, h, w = x_scale_BCHW.shape
            if h > self.merge_ratio and w > self.merge_ratio:
                # run windowing
                x_scale_win = rearrange(
                    x_scale_BCHW,
                    "b c (h m1) (w m2) -> b h w m1 m2 c",
                    m1=self.merge_ratio,
                    m2=self.merge_ratio,
                )
                grid_size = [x_scale_win.shape[1], x_scale_win.shape[2]]
                x_scale_win = rearrange(
                    x_scale_win, "b h w m1 m2 c -> (b h w) (m1 m2) c"
                )
            else:
                x_scale_win = rearrange(x_scale_BCHW, "b c h w -> b (h w) c")
                grid_size = [1, 1]

            stages.append(x_scale_win)
            self.grid_sizes.append(grid_size)

        return stages

    def forward_features(self, x, process_readout=True, grid_sizes=None):
        """Forward pass through feature extraction layers."""
        grid_sizes = [layout["grid_size"] for layout in self.multiscale_layout]

        for level in self.levels:
            x = level(x, grid_sizes=grid_sizes)

        if process_readout:
            if self.readout_norm == "batchnorm":
                return self._process_batchnorm_features(x)
            elif self.readout_norm == "flatln":
                return self._process_flatln_features(x)
            else:  # layernorm
                return self._process_layernorm_features(x)
        else:
            return x

    def _process_batchnorm_features(self, x):
        """Process features using batch normalization."""
        bsz = self.bsz
        resolution = self.img_size[0]

        patchembed_downsample = 16
        downsample = self.local2global

        readout_feats = None
        readout_res = resolution // patchembed_downsample

        m0 = readout_res
        for scale in x:
            x0 = rearrange(scale, "(b nw) k c -> b c (nw k)", b=bsz)
            x0 = rearrange(x0, "b c (m0 m1) -> b c m0 m1", m0=m0, m1=m0)
            x0 = F.interpolate(x0, size=(readout_res, readout_res), mode="nearest")
            m0 = m0 // downsample

            if readout_feats is None:
                readout_feats = x0
            else:
                readout_feats += x0

        readout_feats = self.norm(readout_feats)
        readout_feats = self.avgpool(readout_feats)
        return torch.flatten(readout_feats, 1)

    def _process_flatln_features(self, x):
        """Process features using flat layer normalization."""
        feats = rearrange(x[0], "(b nw) k c -> b (nw k) c", b=64)
        x = self.norm(feats)
        return x.mean(1)

    def _process_layernorm_features(self, x):
        """Process features using layer normalization."""
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, batch=None):
        """Forward pass through the entire network."""
        x = self.forward_raw(x)
        x = self.forward_features(x)
        x = self.head(x)
        return {"logit": x}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Get keywords for parameters that should not use weight decay."""
        return {"rpb"}
