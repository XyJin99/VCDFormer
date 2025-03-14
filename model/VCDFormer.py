import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=10):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.ws = ws
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, t, H, W):
        BT, N, C = x.shape
        B = BT // t
        attn_x = x.view(-1, t, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        if pad_r > 0 or pad_b > 0:
            attn_x = F.pad(attn_x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        h_group, w_group = math.ceil(H / self.ws), math.ceil(W / self.ws)
        assert attn_x.shape[2] == h_group * self.ws, 'The wrong padding.'

        attn_x = attn_x.reshape(B, t, h_group, self.ws, w_group, self.ws, C).permute(0, 2, 4, 1, 3, 5, 6).\
            reshape(B, h_group*w_group, -1, C)

        qkv = self.qkv(attn_x).reshape(B, h_group*w_group, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(2, 3).reshape(B, h_group*w_group, -1, C)
        attn_out = attn_out.reshape(B, h_group, w_group, t, self.ws, self.ws, C).permute(0, 3, 1, 4, 2, 5, 6).\
            reshape(B, t, h_group * self.ws, w_group * self.ws, C)

        if (h_group > H // self.ws) or (w_group > W // self.ws):    # delete padding
            attn_out = attn_out[:, :, :H, :W, :].contiguous()       # B t H W C
        attn_out = attn_out.reshape(B, -1, C)                       # B N C
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        x = x.view(-1, N, C)
        return x


class STWT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ws=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ws=ws)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, t, H, W):
        x = x + self.drop_path(self.attn1(self.norm1(x), t, H, W))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.view(-1, H, W, C).permute(0, 3, 1, 2)
            x_ = self.sr(x_)
            x_ = x_.view(B, -1, C, x_.shape[2], x_.shape[3]).transpose(1, 2)
            x_ = x_.view(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SRT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn2 = Attention2(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, t, H, W):
        x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def tsm(tensor, version='circulant'):
    shape = B, T, C, H, W = tensor.shape
    split_size = C // 4

    pre_tensor, post_tensor, peri_tensor = tensor.split(
        [split_size, split_size, C - 2 * split_size],
        dim=2
    )
    if version == 'zero':
        pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]  # NOQA
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]  # NOQA
    elif version == 'circulant':
        pre_tensor = torch.cat((pre_tensor[:, -1:, ...],  # NOQA
                                pre_tensor[:, :-1, ...]), dim=1)  # NOQA
        post_tensor = torch.cat((post_tensor[:, 1:, ...],  # NOQA
                                 post_tensor[:, :1, ...]), dim=1)  # NOQA
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, t, H, W):
        BT, N, C = x.shape
        x = x.view(-1, t, H, W, C).permute(0, 1, 4, 2, 3)
        x = tsm(x).view(-1, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLPLayer(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=768, dropout=0.1, interpolate_mode='bilinear', align_corners=False):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.dropout = nn.Dropout2d(dropout)
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        self.linear_c4 = MLPLayer(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLPLayer(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLPLayer(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLPLayer(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
            nn.ReLU()
        )
        self.linear_project = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim // 4, kernel_size=3 ,padding=1),
            nn.ReLU()
        )
        self.linear_saliency_pred = nn.Conv2d(embedding_dim // 4, 1, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, t, _, _, _ = c4.shape
        c1 = c1.view(n * t, -1, c1.shape[3], c1.shape[4])
        c2 = c2.view(n * t, -1, c2.shape[3], c2.shape[4])
        c3 = c3.view(n * t, -1, c3.shape[3], c3.shape[4])
        c4 = c4.view(n * t, -1, c4.shape[3], c4.shape[4])

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n * t, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n * t, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n * t, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode=self.interpolate_mode, align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n * t, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = F.interpolate(_c, size=_c.shape[-1]*2, mode=self.interpolate_mode, align_corners=self.align_corners)
        _c = self.linear_project(_c)
        _c = F.interpolate(_c, size=_c.shape[-1]*2, mode=self.interpolate_mode, align_corners=self.align_corners)
        x = self.dropout(_c)
        x = self.linear_saliency_pred(x)
        x = x.view(n, t, -1, x.shape[2], x.shape[3])
        return x


class LSTRB(nn.Module):
    def __init__(self, in_chans, embed_dim=768):
        super(LSTRB, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, 1, 1, bias=True, groups=embed_dim), )

    def forward(self, x, t, H, W):
        BT, N, C = x.shape
        x = x.view(-1, t, H, W, C).permute(0, 1, 4, 2, 3)
        cnn_feat = tsm(x).view(-1, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class VCDNet(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[4, 4, 4, 4], mlp_ratios=[3, 3, 3, 3], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 2, 2], ws=[10, 10, 10, 10], sr_ratio=[8, 4, 2, 1], extra_norm=True):
        super(VCDNet, self).__init__()
        self.depths = depths

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(norm_layer(dim))

        self.patch_embeddings = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeddings.append(
                    OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[i]))
            else:
                self.patch_embeddings.append(
                    OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[i - 1], embed_dim=embed_dims[i]))

        self.LSTRB = nn.ModuleList([LSTRB(embed_dim, embed_dim) for embed_dim in embed_dims])

        cur = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([STWT(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, ws=ws[k]) for i in range(depths[k])])
            self.blocks1.append(_block)

            _block = nn.ModuleList([SRT(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratio[k]) for i in range(depths[k])])
            self.blocks2.append(_block)

            cur += depths[k]

        self.decoder = SegFormerHead(in_channels=embed_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        out = []
        b, t, c, h, w = x.shape

        for i in range(len(self.depths)):
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            x, H, W = self.patch_embeddings[i](x)
            for j, blk in enumerate(zip(self.blocks1[i], self.blocks2[i])):
                blk1, blk2 = blk
                x = blk1(x, t, H, W)
                x = blk2(x, t, H, W)
                if j == 0:
                    x = self.LSTRB[i](x, t, H, W)
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.view(b, t, H, W, -1)
            x = x.permute(0, 1, 4, 2, 3)
            out.append(x)

        output = self.decoder(out)
        return output

if __name__ == '__main__':
    import time
    model = VCDNet()
    model.eval()
    b = 1
    input = torch.randn([b, 5, 3, 320, 320])
    t1 = time.time()
    output = model(input)
    print(time.time()-t1)
    print(output.shape)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

