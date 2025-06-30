# model: SRFormer
# SRFormer: Permuted Self-Attention for Single Image Super-Resolution
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import to_2tuple, trunc_normal_


class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x  # 返回输入以避免将其变为 None


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import to_2tuple, trunc_normal_


class PredictorLG(nn.Module):
    def __init__(self, dim, window_size=7, ratio=0.5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # 输出单通道概率图 (b, 2, h, w)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, int(dim * ratio), 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(int(dim * ratio), 2, 1, 1, 0)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape

        # 计算偏移图 (b, 2, h, w)
        offset_map = self.conv(x)  # (b, 2, h, w)

        # 四舍五入为整数偏移
        offset_map_int = torch.round(offset_map).to(torch.int32)  # (b, 2, h, w)

        # 生成全局索引
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device))
        idx = grid_y * w + grid_x
        idx = idx.view(1, h * w).expand(b, -1)

        return offset_map_int, idx, x



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# 定义dwconv类
class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1,
                      groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        b, n, c = x.shape

        # 自动推断最接近的 (H, W)
        h = int(n ** 0.5)
        while n % h != 0:
            h -= 1
        w = n // h

        assert h * w == n, f"dwconv reshape失败：{h}x{w} ≠ {n}"

        x = x.view(b, h, w, c).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = self.depthwise_conv(x)
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()  # [B, N, C]
        return x


# 定义window_partition和window_reverse函数
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 修改了填充的顺序，确保与维度顺序一致
    h_padded, w_padded = h + pad_h, w + pad_w

    # 添加了对填充后尺寸的检查
    assert h_padded % window_size == 0 and w_padded % window_size == 0, \
        f"After padding, dimensions must be divisible by window_size. Got ({h_padded}, {w_padded}) with window_size {window_size}"

    x = x.permute(0, 2, 3, 1).contiguous()  # 将通道维度移到最后，便于后续处理
    x = x.view(b, h_padded // window_size, window_size, w_padded // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def deform_window_partition(x, offset_map_int, window_size):
    """
    Args:
        x: (b, c, h, w)
        offset_map_int: (b, 2, h, w)  # 整数偏移图
        window_size: 窗口大小
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, c, h, w = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, pad_w, 0, pad_h))  # [b, c, h+p, w+p]
    offset_map_int = F.pad(offset_map_int, (0, pad_w, 0, pad_h))  # [b, 2, h+p, w+p]

    h_padded, w_padded = h + pad_h, w + pad_w

    # 将图像转为 (b, h, w, c)
    x = x.permute(0, 2, 3, 1).contiguous()  # [b, h+p, w+p, c]

    # 构建网格坐标
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h_padded, device=x.device),
        torch.arange(w_padded, device=x.device)
    )
    grid_y = grid_y.expand(b, -1, -1)  # [b, h+p, w+p]
    grid_x = grid_x.expand(b, -1, -1)

    # 加上偏移
    offset_y = offset_map_int[:, 0]  # [b, h+p, w+p]
    offset_x = offset_map_int[:, 1]  # [b, h+p, w+p]

    new_grid_y = torch.clamp(grid_y + offset_y, 0, h_padded - 1)
    new_grid_x = torch.clamp(grid_x + offset_x, 0, w_padded - 1)

    # 创建变形后的特征图
    batch_idx = torch.arange(b, device=x.device).view(b, 1, 1).expand(-1, h_padded, w_padded)
    x_warped = x[batch_idx, new_grid_y, new_grid_x]  # [b, h+p, w+p, c]

    # 窗口划分
    x_warped = x_warped.permute(0, 3, 1, 2).contiguous()
    windows = window_partition(x_warped, window_size)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# 定义ConvFFN类


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 统一处理参数
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 线性层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        # 深度可分离卷积
        self.dwconv = dwconv(hidden_features=hidden_features)

        # 输出层
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # 空模块（用于调试/扩展）
        self.before_add = emptyModule()
        self.after_add = emptyModule()

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.before_add(x)
        x = x + self.dwconv(x, x_size)  # 添加残差连接
        x = self.after_add(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def to(self, device):
        self.device = device
        # 确保所有已构建的层也移动到指定设备
        if self.in_conv is not None:
            self.in_conv.to(device)
        if self.out_offsets is not None:
            self.out_offsets.to(device)
        if self.out_mask is not None:
            self.out_mask.to(device)
        return super().to(device)


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class PSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        # 使用permuted_window_size作为实际窗口大小
        if isinstance(window_size, (tuple, list)):
            self.window_size = window_size[0]
        else:
            self.window_size = window_size
        self.permuted_window_size = self.window_size // 2
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)  # 添加这一行初始化attn_drop

        # KV压缩为原来一半
        assert dim % 4 == 0, f"特征维度{dim}必须能被4整除以进行KV压缩"
        kv_dim = dim // 2
        head_dim_kv = dim // num_heads

        self.q = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * kv_dim, bias=qkv_bias)

        self.scale = qk_scale or head_dim_kv ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(kv_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 基于permuted_window_size构建相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size - 1) * (2 * self.permuted_window_size - 1), num_heads)
        )

        # 使用permuted_window_size生成坐标
        coords_h = torch.arange(self.permuted_window_size)
        coords_w = torch.arange(self.permuted_window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.permuted_window_size - 1  # 确保索引从0开始
        relative_coords[:, :, 1] += self.permuted_window_size - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        # 计算 Q, K, V
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()  # [heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)  # [B, heads, N, N]

        # ✅ 如果 mask 尺寸与 N 不一致，直接忽略
        if mask is not None:
            if mask.shape[-1] != N:
                #print(f"Warning: mask.shape[-1] = {mask.shape[-1]} ≠ N = {N}, 不使用 mask。")
                mask = None

        if mask is not None:
            nw_mask = mask.shape[0]
            assert B % nw_mask == 0, f"B ({B}) 必须能被 nw_mask ({nw_mask}) 整除"
            num_window_batch = B // nw_mask

            expanded_mask = mask.unsqueeze(0).unsqueeze(2)  # [1, nw, 1, N, N]
            expanded_mask = expanded_mask.expand(num_window_batch, nw_mask, self.num_heads, N, N)
            expanded_mask = expanded_mask.reshape(B, self.num_heads, N, N)

            attn = attn + expanded_mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        flops = 0
        flops += n * self.dim * 1.5 * self.dim
        flops += self.num_heads * n * (self.dim // (2 * self.num_heads)) * n
        flops += self.num_heads * n * n * (self.dim // (2 * self.num_heads))
        flops += n * self.dim * self.dim
        return flops


# basicsr/archs/srformer_arch.py


class PSA_Block(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,  # 确保这个参数被接收
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size if isinstance(window_size, int) else window_size[0]
        self.permuted_window_size = self.window_size // 2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 偏移预测模块
        self.offset_head = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim // 4, 2, kernel_size=1)
        )

        # 添加dwconv模块
        self.dwconv = dwconv(hidden_features=dim)

        # 注意力模块
        self.norm1 = norm_layer(dim)
        self.proj_in = nn.Linear(dim, dim)
        self.attn = PSA(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        # 修复点：统一使用关键字参数初始化ConvFFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # 窗口预测器
        self.predictor = PredictorLG(dim=dim, window_size=self.permuted_window_size, ratio=0.5)

        # 注意力掩码
        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(input_resolution)
        else:
            self.attn_mask = None


        # 🧠 新增窗口级门控模块
        self.window_gate = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim // 2, self.dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim // 2, self.dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim // 4, 16, kernel_size=1),  # 中间维度压缩到16
            nn.GELU(),

            nn.Conv2d(16, 2, kernel_size=1),  # 最终输出2类 logits
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

        # 空模块（保持兼容性）
        self.after_norm1 = emptyModule()
        self.after_attention = emptyModule()
        self.residual_after_attention = emptyModule()
        self.after_norm2 = emptyModule()
        self.after_mlp = emptyModule()
        self.residual_after_mlp = emptyModule()

    def calculate_mask(self, x_size):
        """生成移位窗口的注意力掩码"""
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1), device=self.offset_head[0].weight.device)

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        window_size = self.permuted_window_size

        # 补边
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = x.view(b, h, w, c)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.view(b, -1, c)
            h, w = h + pad_h, w + pad_w

        shortcut = x

        # 归一化
        x = self.norm1(x)
        x_2d = x.view(b, h, w, c).permute(0, 3, 1, 2)  # [B,C,H,W]

        # 获取偏移图和变形窗口
        offset_map_int, idx_full, x_warped_base = self.predictor(x_2d)
        x_warped = deform_window_partition(x_2d, offset_map_int, window_size)

        num_windows_h = h // window_size
        num_windows_w = w // window_size
        num_windows = num_windows_h * num_windows_w

        # 展平为 (b, num_windows, window_size*window_size, c)
        x_win = x_warped.view(b, num_windows, window_size * window_size, c)

        # 计算每个窗口的偏移强度（绝对值和）
        offset_mag = offset_map_int.abs().sum(dim=1)  # [b, h, w]
        offset_mag = offset_mag.view(b, -1)  # [b, h*w]

        # 取每个窗口区域的偏移强度均值
        offset_mag_windows = offset_mag.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
        offset_mag_windows = offset_mag_windows.contiguous().view(b, num_windows_h, num_windows_w, -1).float().mean(
            dim=-1)
        offset_mag_windows = offset_mag_windows.view(b, -1)  # [b, num_windows]

        # 排序并取 top-k 和 bottom-k 的窗口索引
        k = num_windows // 2
        _, idx_topk = torch.topk(offset_mag_windows, k, dim=1)  # top-k 窗口
        idx_all = torch.arange(num_windows, device=x.device).expand(b, -1)
        mask = idx_all.scatter(1, idx_topk, 0).bool()
        selected = idx_all[mask]
        idx_bottomk_list = selected.split(k)
        max_len = max(len(t) for t in idx_bottomk_list)

        padded_list = []
        for t in idx_bottomk_list:
            if len(t) < max_len:
                pad_size = max_len - len(t)
                padded = torch.cat([t, torch.zeros(pad_size, device=t.device, dtype=t.dtype)])
                padded_list.append(padded)
            else:
                padded_list.append(t[:max_len])
        idx_bottomk = torch.stack(padded_list, dim=0)

        # 提取两个分支的窗口
        v1 = torch.gather(x_win, 1, idx_topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size * window_size, c))
        v2 = torch.gather(x_win, 1,
                          idx_bottomk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size * window_size, c))

        B, K = v1.shape[0], v1.shape[1]
        v1 = v1.view(B * K, window_size * window_size, c)
        attn_out = self.attn(self.proj_in(v1), self.attn_mask)
        attn_out = attn_out.view(B, K, window_size * window_size, c)
        attn_out = attn_out.view(B, -1, c)

        conv_out = self.dwconv(v2.view(B, -1, c), (h, w))

        # 🧠 新增：窗口级门控模块
        # 将窗口展平用于卷积处理
        x_win_for_gate = x_win.view(B * num_windows, c, window_size, window_size)
        gate_logits = self.window_gate(x_win_for_gate)  # (b*num_windows, 2)

        # reshape 回到 (b, num_windows, 2)
        gate_logits = gate_logits.view(B, num_windows, 2)

        # 每个窗口单独决策
        window_choice = torch.argmax(gate_logits, dim=-1)  # shape: (b, num_windows)


        # 初始化最终输出
        merged_result = []

        # 合并结果
        window_elements = window_size * window_size
        idx_topk_expanded = idx_topk.unsqueeze(-1).expand(-1, -1, window_elements)
        arange = torch.arange(window_elements, device=x.device).view(1, 1, -1)
        current_idx1 = (idx_topk_expanded * window_elements + arange).view(B, -1)
        current_idx2 = (idx_bottomk.unsqueeze(-1).expand(-1, -1, window_elements) * window_elements + arange).view(B,
                                                                                                                   -1)

        # 构建一个全零张量作为合并容器
        out_i_template = torch.zeros((x.shape[1], c), device=x.device)

        for i in range(B):
            idx1_i = torch.clamp(current_idx1[i], 0, x.shape[1] - 1)
            idx2_i = torch.clamp(current_idx2[i], 0, x.shape[1] - 1)

            # 创建当前 batch 的输出容器
            out_i = out_i_template.clone()

            # 遍历每个窗口进行选择
            for win_idx in range(num_windows):
                if window_choice[i][win_idx] == 0:
                    # 使用卷积分支
                    start = win_idx * window_elements
                    end = start + window_elements
                    indices = idx2_i[start:end]
                    values = conv_out[i][start:end]
                    out_i.index_add_(0, indices, values)
                else:
                    # 使用注意力分支
                    start = win_idx * window_elements
                    end = start + window_elements
                    indices = idx1_i[start:end]
                    values = attn_out[i][start:end]
                    out_i.index_add_(0, indices, values)

            merged_result.append(out_i)

        merged = torch.stack(merged_result, dim=0)
        merged = merged.view(b, h, w, c)

        if pad_h > 0 or pad_w > 0:
            merged = merged[:, :h - pad_h, :w - pad_w, :]

        x = shortcut + self.drop_path(merged.view(b, -1, c))
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))
        # 使用中间特征图可视化：x_2d shape = (B, C=96, H, W)
        if not self.training:
            self.save_gate_heatmap(window_choice, h, w, window_size, input_image=x_2d, prefix="vis_overlay")

        return x

    def save_gate_heatmap(self, window_choice, h, w, window_size, prefix="vis", input_image=None):
        import os
        import numpy as np
        import cv2
        from datetime import datetime

        os.makedirs("gate_vis", exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        B, num_windows = window_choice.shape
        num_h = h // window_size
        num_w = w // window_size

        for i in range(B):
            heatmap = window_choice[i].view(num_h, num_w).detach().cpu().numpy().astype(np.uint8)

            # 每个窗口扩展为 window_size x window_size 像素块
            heatmap_img = np.kron(heatmap, np.ones((window_size, window_size), dtype=np.uint8)) * 127

            # 将值为0的区域设为深色，1为亮色（或你也可以改为彩色）
            heatmap_img_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

            # 保存热力图


    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * 25
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            PSA_Block(
                dim=dim,  # 确保这里使用传入的 dim（应为 60）
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PSA_Group(nn.Module):
    """Residual Swin Transformer Block (PSA_Group).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(PSA_Group, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.before_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_Residual = emptyModule()

    def forward(self, x, x_size):
        return self.after_PSA_Group_Residual(self.after_PSA_Group_conv(self.patch_embed(
            self.conv(self.patch_unembed(self.before_PSA_Group_conv(self.residual_group(x, x_size)), x_size)))) + x)

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, window_size=24, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # 确保img_size能被window_size整除
        if img_size % window_size != 0:
            img_size = img_size + (window_size - img_size % window_size)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, window_size=24, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # 确保img_size能被window_size整除
        if img_size % window_size != 0:
            img_size = img_size + (window_size - img_size % window_size)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, x_size):
        # 将序列化数据转换回图像格式
        b, hw, c = x.shape
        h, w = x_size
        # 确保hw等于h*w
        assert hw == h * w, f"Input sequence length ({hw}) doesn't match expected size ({h}*{w})"
        # 调整维度顺序以生成图像
        x = x.transpose(1, 2).view(b, c, h, w)
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    class Upsample(nn.Sequential):
        def __init__(self, scale, num_feat):
            m = []
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log(scale, 2))):
                    m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                    m.append(nn.PixelShuffle(2))
            elif scale == 3:
                m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
            super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class SRFormer(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=60,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,  # 改为8的倍数，确保能被256整除
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,  # 确保这里是4，表示4倍超分
                 img_range=1.,
                 upsampler='pixelshuffle',  # 明确指定上采样方式
                 resi_connection='1conv',
                 **kwargs):
        super(SRFormer, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # 使用 embed_dim: 60

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=256,  # 固定为目标分辨率
            window_size=window_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            window_size=window_size,
            patch_size=patch_size,
            in_chans=embed_dim,  # 使用 embed_dim: 60
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Permuted Self Attention Group  (PSA_Group)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PSA_Group(
                dim=embed_dim,  # 如果有下采样则按倍数增长
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        # 在__init__方法中找到上采样部分，替换为：
        # 替换原有的上采样部分
        if self.upsampler == 'pixelshuffle':
            # 4倍上采样需要两个2倍上采样阶段
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 256, 3, 1, 1),
                nn.LeakyReLU(inplace=True))

            self.upsample = nn.Sequential(
                # 第一阶段上采样 (64x64 -> 128x128)
                nn.Conv2d(256, 256 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True),
                # 第二阶段上采样 (128x128 -> 256x256)
                nn.Conv2d(256, 256 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_last = nn.Conv2d(256, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # 轻量级上采样保持不变
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # 确保输入尺寸可以被 window_size 整除
        pad_h = (self.window_size - x_size[0] % self.window_size) % self.window_size
        pad_w = (self.window_size - x_size[1] % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # 归一化处理 - 确保非原地操作
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range

        x = self.conv_first(x)
        x = x + self.conv_after_body(self.forward_features(x))  # 确保残差连接是非原地的

        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)

        # 反归一化 - 确保非原地操作
        x = x / self.img_range + mean

        # 移除 padding 并确保输出尺寸正确
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 9 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

    # 修改batch_index_select函数以解决索引不匹配问题
    # 增加对idx的范围检查和调试信息
    def batch_index_select(x, idx):
        """
        Args:
            x: (B, N, C)
            idx: (B, K)
        Returns:
            out: (B, K, C)
        """
        B, N, C = x.shape
        offset = torch.arange(B, device=x.device).reshape(B, 1) * N

        # 确保idx的范围在[0, N)之间
        if idx.min() < 0 or idx.max() >= N:
            print(f"Warning: Index out of bounds in batch_index_select. idx range: [{idx.min()}, {idx.max()}], N: {N}")
            idx = torch.clamp(idx, 0, N - 1)

        idx = idx + offset
        idx = idx.reshape(-1)

        x_flat = x.reshape(B * N, C)
        output = x_flat[idx].reshape(B, -1, C)
        return output

model = SRFormer()
print(model)


