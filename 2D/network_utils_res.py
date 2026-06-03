import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from timm.layers import DropPath
from einops import rearrange


class Conv(nn.Module):
    def __init__(self, dim_i, dim_o, k, groups=1, bias=True, device=None):
        super(Conv, self).__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(dim_i, dim_o, kernel_size=k, stride=1, padding=p, groups=groups, bias=bias, device=device)

    def forward(self, x):
        return self.conv(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = torch.mean(x * x, dim=-1, keepdim=True)
        return x / torch.sqrt(sigma + 1e-8) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-8) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DyT(nn.Module):
    def __init__(self, dim, init_a):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_a)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = torch.tanh(self.alpha * to_3d(x))
        x = self.weight * x + self.bias
        return to_4d(x, h, w)


# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#         # self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.dwconv = Conv(hidden_features * 2, hidden_features * 2, k=3, groups=hidden_features * 2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.dim_sp = hidden_features // 4
        self.conv_init = nn.Conv2d(dim, hidden_features, 1, bias=bias)

        self.conv1_1 = Conv(self.dim_sp, self.dim_sp, k=3, groups=self.dim_sp, bias=bias)
        self.conv1_2 = Conv(self.dim_sp, self.dim_sp, k=5, groups=self.dim_sp, bias=bias)
        self.conv1_3 = Conv(self.dim_sp, self.dim_sp, k=7, groups=self.dim_sp, bias=bias)

        self.gelu = nn.GELU()
        self.conv_fina = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_dwconv = Conv(dim * 3, dim * 3, k=3, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, LN_flag=True, drop_path=0.):
        super(TransformerBlock, self).__init__()

        if LN_flag:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.norm2 = LayerNorm(dim, LayerNorm_type)
        else:
            self.norm1 = DyT(dim=dim, init_a=0.8)
            self.norm2 = DyT(dim=dim, init_a=0.2)
        self.attn = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ResidualGroup(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, LN_flag=True, drop_path_rates=None):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, LN_flag=LN_flag, drop_path=drop_path_rates[_])
                for _ in range(num_blocks)
            ]
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.dw = Conv(dim, dim, k=3, groups=dim, bias=bias)

    def forward(self, x):
        return self.dw(self.pw(self.blocks(x))) + x


class DeepEncoder_Trans(nn.Module):
    """深度编码器，由通道注意力残差组组成"""
    def __init__(self,
                 num_features=48,
                 num_heads=4,
                 ffn_expansion_factor=1.66,
                 bias=False,
                 # LayerNorm_type='WithBias',
                 LayerNorm_type='BiasFree',
                 num_groups=4,
                 num_blocks=4,
                 LN_flag=True):
        """
        :param num_features: 通道数
        :param num_groups: 通道注意力残差组数量
        :param num_blocks: 每个残差组中的残差块数量
        """
        super(DeepEncoder_Trans, self).__init__()
        self.residual_groups = nn.ModuleList(
            [ResidualGroup(num_features, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks=num_blocks, LN_flag=LN_flag) for _ in range(num_groups)]
        )
        # self.final_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1, bias=bias),
            # Conv(num_features, num_features, k=3, groups=num_features, bias=bias),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, groups=num_features, bias=bias)
        )

    def forward(self, x):
        """
        :param x: 输入特征
        :return: 输出深度特征
        """
        residual = x
        for i, group in enumerate(self.residual_groups):
            x = group(x)
        x = self.final_conv(x)
        return x + residual


class DeepEncoder_AxialTransUNet(nn.Module):
    def __init__(self,
                 dim=32,
                 num_blocks=(1, 1, 2),
                 num_heads=(4, 4, 4),
                 ffn_expansion_factor=1.75,
                 bias=True,
                 # LayerNorm_type='WithBias',
                 LayerNorm_type='BiasFree',
                 LN_flag=True,
                 max_drop_path_rate=0.):
        super(DeepEncoder_AxialTransUNet, self).__init__()

        self.num_layers = len(num_blocks)

        drop_path_rates = self.compute_drop_path_rates(num_blocks, max_drop_path_rate)
        idx = 0

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.encoder.append(
                ResidualGroup(dim, num_heads[i], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[i],
                              LN_flag=LN_flag, drop_path_rates=drop_path_rates[idx:idx+num_blocks[i]])
            )
            self.down.append(nn.Identity())
            idx += num_blocks[i]

        self.bottleneck = ResidualGroup(dim, num_heads[-1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[-1],
                                        LN_flag=LN_flag, drop_path_rates=drop_path_rates[idx:idx+num_blocks[-1]])

        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.up.append(nn.Identity())
            self.decoder.append(
                ResidualGroup(dim, num_heads[self.num_layers - 2 - i], ffn_expansion_factor, bias, LayerNorm_type,
                              num_blocks[self.num_layers - 2 - i], LN_flag=LN_flag,
                              drop_path_rates=drop_path_rates[idx-num_blocks[self.num_layers - 2 - i] : idx])
            )
            idx -= num_blocks[self.num_layers - 2 - i]

        self.skip_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            Conv(dim, dim, k=3, groups=dim, bias=bias))


    def compute_drop_path_rates(self, depths, max_drop_path_rate):
        total_blocks = sum(depths)
        drop_path_rates = []
        current_block = 0

        for stage_depth in depths:
            for block_idx_in_stage in range(stage_depth):
                depth_ratio = (current_block + block_idx_in_stage) / (total_blocks - 1)
                dpr = max_drop_path_rate * depth_ratio
                drop_path_rates.append(dpr)
            current_block += stage_depth

        return drop_path_rates


    def forward(self, x):
        features = []

        enc = x
        for i in range(self.num_layers - 1):
            enc = self.encoder[i](enc)
            features.append(enc)
            enc = self.down[i](enc)

        dec = self.bottleneck(enc)

        for i in range(self.num_layers - 1):
            dec = self.up[i](dec)

            skip_feature = features[self.num_layers - 2 - i]

            dec = dec + skip_feature

            dec = self.decoder[i](dec)

        out = self.skip_conv(dec)

        return out