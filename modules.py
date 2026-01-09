import numpy as np
import torch
import time
import sys
import os
import zlib
import bz2
import lzma
import heapq
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torchvision

from utils.loss_utils import ssim
from lpipsPyTorch import lpips

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    zstd = None
    HAS_ZSTD = False

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    brotli = None
    HAS_BROTLI = False

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except ImportError:
    lz4f = None
    HAS_LZ4 = False

try:
    import zopfli.zlib as zopfli_zlib
    HAS_ZOPFLI = True
except ImportError:
    zopfli_zlib = None
    HAS_ZOPFLI = False

def _dict_entry_size(key, value):
    size = sys.getsizeof(key)
    if isinstance(value, np.ndarray):
        size += value.nbytes
    elif isinstance(value, (bytes, bytearray, memoryview)):
        size += len(value)
    elif hasattr(value, '__sizeof__'):
        size += value.__sizeof__()
    return size


def estimate_dict_size(dic):
    total_size = 0
    for k, v in dic.items():
        total_size += _dict_entry_size(k, v)
    return total_size


def _payload_size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    try:
        return sys.getsizeof(value)
    except Exception:
        return 0


def _format_size(value_bytes):
    if value_bytes >= 1024 * 1024:
        return value_bytes / (1024 * 1024), "MB"
    if value_bytes >= 1024:
        return value_bytes / 1024, "KB"
    return value_bytes, "B"


def _format_size_str(value_bytes):
    value, unit = _format_size(value_bytes)
    return f"{value:.2f} {unit}"


def _attribute_storage_breakdown(dic):
    """
    Estimate per-attribute storage for packed tensors (dc, opacity, scale, rotation).
    Uses proportional split within each packed tensor, so values are approximate but
    help visualize the composition in charts.
    """
    length_map = {'f_dc': 3, 'opacities': 1, 'scale': 3, 'rotation': 4}
    attr_sizes = {}

    for group in ['int2', 'int4', 'int8', 'fp16']:
        names = dic.get(f'{group}_names', [])
        if not names:
            continue
        payload_key = None
        for candidate in (f'M_{group}_compressed', f'M_{group}'):
            if candidate in dic:
                payload_key = candidate
                break
        if payload_key is None:
            continue

        payload_size = _payload_size(dic[payload_key])
        total_width = sum(length_map.get(n, 0) for n in names)
        if payload_size <= 0 or total_width <= 0:
            continue

        for n in names:
            width = length_map.get(n, 0)
            if width == 0:
                continue
            attr_sizes[n] = attr_sizes.get(n, 0.0) + payload_size * (width / total_width)
    return attr_sizes


def _attribute_payloads(dic):
    """
    Approximate payload-only sizes for dc/opacity/scaling/rotation/xyz/sh.
    """
    length_map = {'f_dc': 3, 'opacities': 1, 'scale': 3, 'rotation': 4}
    payloads = {}

    def _add(name, payload):
        payloads[name] = payloads.get(name, 0.0) + payload

    for group in ['int2', 'int4', 'int8', 'fp16']:
        names = dic.get(f'{group}_names', [])
        if not names:
            continue
        payload_key = None
        for candidate in (f'M_{group}_compressed', f'M_{group}'):
            if candidate in dic:
                payload_key = candidate
                break
        if group == 'fp16' and payload_key is None and 'M_fp16' in dic:
            payload_key = 'M_fp16'
        if payload_key is None:
            continue

        payload_size = _payload_size(dic[payload_key])
        total_width = sum(length_map.get(n, 0) for n in names)
        if total_width <= 0:
            continue

        for n in names:
            width = length_map.get(n, 0)
            if width == 0:
                continue
            ratio = width / total_width
            _add(n, payload_size * ratio)

    if not dic.get('sh_all_zero', False):
        for candidate in ('M_rest_compressed', 'M_rest', 'f_rest'):
            if candidate in dic:
                _add('sh', _payload_size(dic[candidate]))
                break

    for candidate in ('xyz_q16_compressed', 'xyz_delta_compressed', 'xyz_q16', 'xyz_delta_raw', 'xyz'):
        if candidate in dic:
            _add('xyz', _payload_size(dic[candidate]))
            break

    return payloads


def _sum_prefix(dic, prefix):
    return sum(_payload_size(v) for k, v in dic.items() if k.startswith(prefix))


def _tensor_bytes(t):
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    if isinstance(t, np.ndarray):
        return t.nbytes
    return 0

def _attr_size_stats(gaussians, sh_zero_mask=None, sh_expected_ratio=None):
    xyz_t = gaussians._xyz.detach()
    f_dc_t = gaussians._features_dc.detach()
    f_rest_t = gaussians._features_rest.detach()
    opacities_t = gaussians._opacity.detach()
    scale_t = gaussians._scaling.detach()
    rotation_t = gaussians.rotation_activation(gaussians._rotation).detach()
    attr_bytes = {
        'xyz': _tensor_bytes(xyz_t),
        'dc': _tensor_bytes(f_dc_t),
        'sh': _tensor_bytes(f_rest_t),
        'opacity': _tensor_bytes(opacities_t),
        'scaling': _tensor_bytes(scale_t),
        'rotation': _tensor_bytes(rotation_t),
    }
    sh_active_bytes = None
    sh_active_ratio = None
    if sh_zero_mask is not None and sh_zero_mask.numel() == f_rest_t.shape[0]:
        active = (~sh_zero_mask).sum().item()
        sh_active_ratio = active / max(1, sh_zero_mask.numel())
        sh_active_bytes = sh_active_ratio * attr_bytes['sh']
    sh_effective_bytes = sh_active_bytes if sh_active_bytes is not None else attr_bytes['sh']

    total = sum(attr_bytes.values())
    base_total = total - attr_bytes['sh'] + sh_effective_bytes
    if base_total <= 0:
        return None

    return {
        'attr_bytes': attr_bytes,
        'sh_active_bytes': sh_active_bytes,
        'sh_active_ratio': sh_active_ratio,
        'sh_effective_bytes': sh_effective_bytes,
        'base_total': base_total,
    }


def _ordered_items(attr_dict, order=None):
    if not order:
        for k, v in attr_dict.items():
            yield k, v
        return
    seen = set()
    for k in order:
        if k in attr_dict and k not in seen:
            yield k, attr_dict[k]
            seen.add(k)
    for k, v in attr_dict.items():
        if k not in seen:
            yield k, v


def _print_attr_sizes(title, stats, order=None):
    if stats is None:
        return
    attr_bytes = stats['attr_bytes']
    base_total = stats['base_total']
    sh_effective_bytes = stats['sh_effective_bytes']
    sh_active_bytes = stats['sh_active_bytes']
    sh_active_ratio = stats['sh_active_ratio']
    print(f"    {title}")
    for name, size in _ordered_items(attr_bytes, order=order):
        display_size = sh_effective_bytes if name == 'sh' else size
        percent = (display_size / base_total * 100.0) if base_total > 0 else 0.0
        if name == 'sh':
            extras = []
            if sh_active_bytes is not None and sh_active_ratio is not None:
                extras.append(f"raw: {_format_size_str(attr_bytes['sh'])}; active_ratio: {sh_active_ratio * 100:.2f}%")
            extra_str = (" | " + "; ".join(extras)) if extras else ""
            print(f"    {name}: {_format_size_str(display_size)} ({percent:.2f}%)" + extra_str)
        else:
            print(f"    {name}: {_format_size_str(display_size)} ({percent:.2f}%)")


def _attr_display_total(stats):
    if stats is None:
        return None
    # Totals should use the same effective sh size as the displayed breakdown
    return stats['base_total']


def _std_key(name):
    """Normalize attribute key names for ordering and display."""
    if name == 'scale':
        return 'scaling'
    if name == 'f_dc':
        return 'dc'
    if name == 'opacities':
        return 'opacity'
    return name


def _log_attr_sizes_from_gaussians(gaussians, title, sh_zero_mask=None, sh_expected_ratio=None, log=True):
    """
    Compute and optionally log attribute sizes; returns the computed stats.
    """
    stats = _attr_size_stats(gaussians, sh_zero_mask=sh_zero_mask, sh_expected_ratio=sh_expected_ratio)
    if log and stats is not None:
        _print_attr_sizes(title, stats)
    return stats


def log_top_storage_entries(dic, top_k=5, original_attr_stats=None, current_attr_stats=None):
    sizes = [(_dict_entry_size(k, v), k) for k, v in dic.items()]
    sizes.sort(reverse=True)
    total = sum(s for s, _ in sizes)
    label_map = {'f_dc': 'dc', 'opacities': 'opacity', 'scale': 'scaling', 'rotation': 'rotation'}
    payloads_raw = _attribute_payloads(dic)
    payloads = {}
    for k, v in payloads_raw.items():
        payloads[_std_key(k)] = payloads.get(_std_key(k), 0) + v
    payload_total = sum(payloads.values()) if payloads else 0
    s_total = _sum_prefix(dic, 'S_')
    z_total = _sum_prefix(dic, 'Z_')

    def _print_entry(key, size):
        percent = (size / total * 100.0) if total > 0 else 0.0
        value, unit = _format_size(size)
        print(f"    {key}: {value:.2f} {unit} ({percent:.2f}%)")

    print("[INFO] Storage breakdown:")
    for size, key in sizes:
        _print_entry(key, size)
    print(f"    Total (all entries): {_format_size_str(total)}")

    if original_attr_stats or current_attr_stats or payloads or s_total or z_total:
        print("[INFO] Storage summary:")
        if original_attr_stats:
            _print_attr_sizes("Original PLY attribute sizes:", original_attr_stats, order=['xyz', 'rotation', 'scaling', 'opacity', 'dc', 'sh'])
            orig_total = _attr_display_total(original_attr_stats)
            print(f"    Total (original attributes): {_format_size_str(orig_total)}")
        if current_attr_stats:
            _print_attr_sizes("Current attribute sizes (post-prune):", current_attr_stats, order=['xyz', 'rotation', 'scaling', 'opacity', 'dc', 'sh'])
            curr_total = _attr_display_total(current_attr_stats)
            print(f"    Total (current attributes): {_format_size_str(curr_total)}")
        if payloads:
            print("    Key payloads (approximate):")
            for name, size in _ordered_items(payloads, order=['xyz', 'rotation', 'scaling', 'opacity', 'dc', 'sh']):
                percent = (size / payload_total * 100.0) if payload_total > 0 else 0.0
                print(f"    {label_map.get(name, name)}: {_format_size_str(size)} ({percent:.2f}%)")

        if s_total or z_total:
            sz_total = s_total + z_total
            denom = payload_total + sz_total if (payload_total + sz_total) > 0 else sz_total or total
            print("    Quantization S/Z overhead:")
            if s_total:
                p = (s_total / denom * 100.0) if denom > 0 else 0.0
                print(f"    S_total: {_format_size_str(s_total)} ({p:.2f}%)")
            if z_total:
                p = (z_total / denom * 100.0) if denom > 0 else 0.0
                print(f"    Z_total: {_format_size_str(z_total)} ({p:.2f}%)")
            if payload_total:
                print(f"    Total (payloads + S/Z): {_format_size_str(payload_total + sz_total)}")

# ----- Pruner Module -----

class Pruner:
    def __init__(self):
        pass

    def prune_points_one_shot(self, prune_mask, sh_mask, gaussians):
        valid_points_mask = ~prune_mask

        gaussians._features_rest = gaussians._features_rest.detach()
        gaussians._features_rest[sh_mask] = 0.0


        gaussians._xyz = gaussians._xyz[valid_points_mask]
        gaussians._features_dc = gaussians._features_dc[valid_points_mask]
        gaussians._features_rest = gaussians._features_rest[valid_points_mask]
        gaussians._opacity = gaussians._opacity[valid_points_mask]
        gaussians._scaling = gaussians._scaling[valid_points_mask]
        gaussians._rotation = gaussians._rotation[valid_points_mask]

    def run(self, prune_rate, sh_rate, import_score, gaussians):
        # Sort importance scores in ascending order
        sorted_scores, sorted_indices = torch.sort(import_score, dim=0)
        num_elements = import_score.size(0)

        # Determine prune mask for the least significant gaussians
        num_prune = int(num_elements * prune_rate)
        prune_indices = sorted_indices[:num_prune]
        prune_mask = torch.zeros(num_elements, dtype=torch.bool)
        prune_mask[prune_indices] = True

        # Determine SH mask for the least significant + SH gaussians
        num_sh = int(num_elements * (prune_rate + sh_rate))
        sh_indices = sorted_indices[:num_sh]
        sh_mask = torch.zeros(num_elements, dtype=torch.bool)
        sh_mask[sh_indices] = True

        # Final SH mask is the set difference between SH and prune masks
        sh_mask = torch.logical_xor(prune_mask, sh_mask)

        # Apply pruning and SH adjustment
        self.prune_points_one_shot(prune_mask, sh_mask, gaussians)


# ----- Quantizer Module -----

class Quantizer:
    def __init__(self, segments: int = None, bits_config: dict = None, use_entropy_coding: bool = True,
                 xyz_codec: str = 'raw32', use_delta_encoding: bool = False,
                 compression_method: str = 'zlib', compression_level: int = None):
        self.segments = segments
        self.bits = bits_config or {
            "bits_f_dc": 8,
            "bits_f_rest": 4,
            "bits_opacity": 4,
            "bits_scaling": 8,
            "bits_rotation": 8
        }
        self.use_entropy_coding = use_entropy_coding
        # XYZ codec config
        # xyz_codec:
        #   'raw32' -> store xyz as float32
        #   'raw16' -> store xyz as float16
        #   'raw8' -> quantize to 8-bit without Morton ordering
        #   'morton16' -> quantize to 16-bit with Morton reordering; optional delta + zlib
        #   'morton8' -> quantize to 8-bit with Morton reordering; optional delta + zlib
        #   'morton32' -> quantize to 32-bit with Morton reordering; optional delta + zlib
        self.xyz_codec = 'raw32' if xyz_codec == 'raw' else xyz_codec
        if self.xyz_codec == 'morton32':
            self.xyz_qbits = 32
        elif self.xyz_codec in {'morton8', 'raw8'}:
            self.xyz_qbits = 8
        else:
            self.xyz_qbits = 16
        self.use_delta_encoding = bool(use_delta_encoding)
        if compression_method is None:
            self.compression_method = None
        else:
            self.compression_method = compression_method.lower()
            if self.compression_method not in {'zlib', 'lzma', 'bz2', 'zstd', 'zopfli', 'xz', 'brotli', 'lz4'}:
                raise ValueError(f"Unsupported compression method: {self.compression_method}")
        if self.xyz_codec not in {'raw32', 'raw16', 'raw8', 'morton16', 'morton8', 'morton32'}:
            raise ValueError(f"Unsupported xyz_codec: {self.xyz_codec}")
        self.compression_level = compression_level
        self.latest_attr_stats = None

    def _normalize_segments(self, num_segments, N):
        """
        Convert user-provided segment count into a safe value.
        None or <=0 means "no segmentation", i.e., use a single segment covering all rows.
        """
        if num_segments is None:
            return 1
        try:
            num = int(num_segments)
        except Exception:
            return 1
        if num <= 0:
            return 1
        return min(max(num, 1), N)

    # -------------------- Morton + Delta for XYZ -------------------- #

    @staticmethod
    def _part1by2(v):
        """Interleave bits with two zeros for Morton codes."""
        v = v.astype(np.uint64)
        v &= 0x0000FFFF
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    def _quantize_xyz16(self, xyz):
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        spans = maxs - mins
        spans[spans == 0] = 1.0
        scale = (2 ** self.xyz_qbits - 1) / spans
        if self.xyz_qbits <= 8:
            q_dtype = np.uint8
        elif self.xyz_qbits <= 16:
            q_dtype = np.uint16
        else:
            q_dtype = np.uint32
        q = np.round((xyz - mins) * scale).astype(q_dtype)
        return q, mins.astype(np.float32), maxs.astype(np.float32)

    def _quantize_xyz16_t(self, xyz_t):
        mins = torch.amin(xyz_t, dim=0)
        maxs = torch.amax(xyz_t, dim=0)
        spans = maxs - mins
        spans = torch.where(spans == 0, torch.ones_like(spans), spans)
        scale = (2 ** self.xyz_qbits - 1) / spans
        q = torch.round((xyz_t - mins) * scale)
        q = torch.clamp(q, 0, 2 ** self.xyz_qbits - 1).to(torch.int64)
        return q, mins.detach().cpu().numpy().astype(np.float32), maxs.detach().cpu().numpy().astype(np.float32)

    def _dequantize_xyz16(self, q, mins, maxs, qbits=None):
        if qbits is None:
            qbits = self.xyz_qbits
        spans = (maxs - mins)
        spans[spans == 0] = 1.0
        scale = (spans / (2 ** qbits - 1)).astype(np.float32)
        mins_f32 = mins.astype(np.float32)
        return q.astype(np.float32) * scale[np.newaxis, :] + mins_f32[np.newaxis, :]

    def _morton_codes16(self, qxyz):
        x = self._part1by2(qxyz[:, 0])
        y = self._part1by2(qxyz[:, 1])
        z = self._part1by2(qxyz[:, 2])
        return x | (y << 1) | (z << 2)

    def _part1by2_21(self, v):
        """Interleave bits with two zeros for up to 21-bit values."""
        v = v.astype(np.uint64)
        v &= np.uint64(0x1FFFFF)
        v = (v | (v << 32)) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << 16)) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << 8)) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << 4)) & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << 2)) & np.uint64(0x1249249249249249)
        return v

    def _morton_codes32_np(self, qxyz):
        coords = qxyz.astype(np.uint64)
        mask21 = np.uint64((1 << 21) - 1)
        x_lo = coords[:, 0] & mask21
        y_lo = coords[:, 1] & mask21
        z_lo = coords[:, 2] & mask21
        x_hi = coords[:, 0] >> 21
        y_hi = coords[:, 1] >> 21
        z_hi = coords[:, 2] >> 21
        low = self._part1by2_21(x_lo) | (self._part1by2_21(y_lo) << 1) | (self._part1by2_21(z_lo) << 2)
        high = self._part1by2_21(x_hi) | (self._part1by2_21(y_hi) << 1) | (self._part1by2_21(z_hi) << 2)
        return high, low

    def _part1by2_tensor(self, v):
        v = v.to(torch.int64)
        v &= 0x0000FFFF
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    def _morton_codes16_t(self, qxyz_t):
        coords = qxyz_t.to(torch.int64)
        x = self._part1by2_tensor(coords[:, 0])
        y = self._part1by2_tensor(coords[:, 1])
        z = self._part1by2_tensor(coords[:, 2])
        return x | (y << 1) | (z << 2)

    def _morton_reorder_indices(self, q_all_t, nonzero_idx, zero_idx):
        if nonzero_idx.numel() > 0:
            nz_codes = self._morton_codes16_t(q_all_t.index_select(0, nonzero_idx))
            nz_order = torch.argsort(nz_codes)
            nonzero_sorted = nonzero_idx.index_select(0, nz_order)
        else:
            nonzero_sorted = nonzero_idx

        if zero_idx.numel() > 0:
            z_codes = self._morton_codes16_t(q_all_t.index_select(0, zero_idx))
            z_order = torch.argsort(z_codes)
            zero_sorted = zero_idx.index_select(0, z_order)
        else:
            zero_sorted = zero_idx

        return nonzero_sorted, zero_sorted

    def _morton_reorder_indices32(self, q_all_t, nonzero_idx, zero_idx):
        q_all = q_all_t.detach().cpu().numpy().astype(np.uint32)
        device = q_all_t.device

        def _sort_idx(idx_t):
            if idx_t.numel() == 0:
                return idx_t
            idx_np = idx_t.detach().cpu().numpy()
            qxyz = q_all[idx_np]
            codes_hi, codes_lo = self._morton_codes32_np(qxyz)
            order = np.lexsort((codes_lo, codes_hi))
            sorted_idx = idx_np[order]
            return torch.from_numpy(sorted_idx).to(device=device)

        return _sort_idx(nonzero_idx), _sort_idx(zero_idx)

    def _delta_encode(self, qxyz):
        # qxyz: (N,3) uint16/uint32 in target order
        if self.xyz_qbits <= 16:
            base = qxyz[0].astype(np.uint16)
            diffs = np.diff(qxyz.astype(np.int32), axis=0)  # (N-1,3) int32
            # ZigZag encode to make small negatives positive
            zz = (diffs << 1) ^ (diffs >> 31)
            return base, zz.astype(np.uint32)
        base = qxyz[0].astype(np.uint32)
        diffs = np.diff(qxyz.astype(np.int64), axis=0)  # (N-1,3) int64
        zz = (diffs << 1) ^ (diffs >> 63)
        return base, zz.astype(np.uint64)

    def _delta_decode(self, base_u16, zz, qbits=None):
        if qbits is None:
            qbits = self.xyz_qbits
        if qbits <= 16:
            base = base_u16.astype(np.int32)
            # inverse zigzag
            diffs = (zz.astype(np.int32) >> 1) ^ -(zz.astype(np.int32) & 1)
            q = np.empty((zz.shape[0] + 1, 3), dtype=np.int32)
            q[0] = base
            q[1:] = diffs
            np.cumsum(q, axis=0, out=q)
            return q.astype(np.uint16)
        base = base_u16.astype(np.int64)
        diffs = (zz.astype(np.int64) >> 1) ^ -(zz.astype(np.int64) & 1)
        q = np.empty((zz.shape[0] + 1, 3), dtype=np.int64)
        q[0] = base
        q[1:] = diffs
        np.cumsum(q, axis=0, out=q)
        return q.astype(np.uint32)

    # -------------------- Entropy Coding Utilities -------------------- #
    
    def entropy_encode(self, data):
        """
        Apply entropy coding (zlib compression) to quantized data.
        Returns compressed bytes and original shape for reconstruction.
        """
        if not self.use_entropy_coding:
            return data, None
        
        # Flatten the array and convert to bytes
        data_bytes = data.tobytes()
        method = self.compression_method
        level = self.compression_level
        if method is None:
            return data, None
        if method == 'zlib':
            lvl = 9 if level is None else max(0, min(9, level))
            compressed = zlib.compress(data_bytes, level=lvl)
        elif method == 'lzma':
            preset = 6 if level is None else max(0, min(9, level))
            compressed = lzma.compress(data_bytes, preset=preset)
        elif method == 'xz':
            preset = 6 if level is None else max(0, min(9, level))
            compressed = lzma.compress(data_bytes, preset=preset, format=lzma.FORMAT_XZ)
        elif method == 'bz2':
            lvl = 9 if level is None else max(1, min(9, level))
            compressed = bz2.compress(data_bytes, compresslevel=lvl)
        elif method == 'zstd':
            if not HAS_ZSTD:
                raise RuntimeError("zstandard module not available; install `zstandard` to use zstd compression.")
            lvl = 3 if level is None else max(-5, min(22, level))
            cctx = zstd.ZstdCompressor(level=lvl)
            compressed = cctx.compress(data_bytes)
        elif method == 'zopfli':
            if not HAS_ZOPFLI:
                raise RuntimeError("zopfli module not available; install `zopfli` to use zopfli compression.")
            # zopfli.zlib.compress returns a zlib-compatible stream; map level to iterations (1-100)
            iterations = 15 if level is None else max(1, min(100, level))
            compressed = zopfli_zlib.compress(data_bytes, numiterations=iterations)
        elif method == 'brotli':
            if not HAS_BROTLI:
                raise RuntimeError("brotli module not available; install `brotli` to use brotli compression.")
            # Brotli quality 0-11 (higher = smaller/slower). Use default 11 if None.
            q = 11 if level is None else max(0, min(11, level))
            compressed = brotli.compress(data_bytes, quality=q)
        elif method == 'lz4':
            if not HAS_LZ4:
                raise RuntimeError("lz4.frame module not available; install `lz4` to use lz4 compression.")
            # lz4 frame compression_level: 0 (default fast) to 16 (slower/better)
            lvl = 0 if level is None else max(0, min(16, level))
            compressed = lz4f.compress(data_bytes, compression_level=lvl)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        return compressed, data.shape
    
    def entropy_decode(self, compressed_data, shape, dtype, method=None):
        """
        Decode entropy-coded data back to original quantized values.
        """
        if method is None:
            method = self.compression_method
        if method is None:
            raise ValueError("compression_method is None; cannot entropy-decode.")

        if method == 'zlib':
            decompressed_bytes = zlib.decompress(compressed_data)
        elif method == 'lzma':
            decompressed_bytes = lzma.decompress(compressed_data)
        elif method == 'xz':
            decompressed_bytes = lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
        elif method == 'bz2':
            decompressed_bytes = bz2.decompress(compressed_data)
        elif method == 'zstd':
            if not HAS_ZSTD:
                raise RuntimeError("zstandard module not available; install `zstandard` to use zstd compression.")
            dctx = zstd.ZstdDecompressor()
            decompressed_bytes = dctx.decompress(compressed_data)
        elif method == 'zopfli':
            # zopfli zlib stream is compatible with standard zlib decompress
            decompressed_bytes = zlib.decompress(compressed_data)
        elif method == 'brotli':
            if not HAS_BROTLI:
                raise RuntimeError("brotli module not available; install `brotli` to use brotli compression.")
            decompressed_bytes = brotli.decompress(compressed_data)
        elif method == 'lz4':
            if not HAS_LZ4:
                raise RuntimeError("lz4.frame module not available; install `lz4` to use lz4 compression.")
            decompressed_bytes = lz4f.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        # Reconstruct array
        data = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        return data

    # -------------------- Bit-Packing Utilities -------------------- #

    def pack_4bit_to_8bit(self, a, b):
        return (a.astype(np.uint8) << 4) | b.astype(np.uint8)

    def unpack_8bit_to_4bit(self, packed_value):
        return (packed_value >> 4) & 0x0F, packed_value & 0x0F

    def pack_2bit_to_8bit(self, a, b, c, d):
        return (
            (a.astype(np.uint8) << 6) |
            (b.astype(np.uint8) << 4) |
            (c.astype(np.uint8) << 2) |
            d.astype(np.uint8)
        )

    def unpack_8bit_to_2bit(self, packed_value):
        return (
            (packed_value >> 6) & 0x03,
            (packed_value >> 4) & 0x03,
            (packed_value >> 2) & 0x03,
            packed_value & 0x03
        )

    # -------------------- Int8 Quantization -------------------- #

    def asymmetric_quantize_int8_per_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(max(num_segments, 1), N)
        indices = np.linspace(0, N, num_segments + 1, dtype=int)

        segment_bounds = [
            (indices[i], indices[i + 1])
            for i in range(num_segments)
            if indices[i] < indices[i + 1]
        ]

        scales = np.zeros((len(segment_bounds), K))
        zero_points = np.zeros((len(segment_bounds), K))
        quantized_data = np.zeros_like(r, dtype=np.int8)

        def process_segment(i):
            start, end = segment_bounds[i]
            seg = r[start:end, :]

            alpha = seg.min(axis=0)
            beta = seg.max(axis=0)
            scale = (beta - alpha) / 255.0
            scale[scale == 0] = 1.0
            zero_point = -(alpha / scale + 128)

            q = np.round(seg / scale + zero_point).astype(np.int8)
            q = np.clip(q, -128, 127)
            q[:, scale == 0] = -128

            return i, q, scale, zero_point

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_segment, i) for i in range(len(segment_bounds))]
            for future in futures:
                i, q_seg, scale, zp = future.result()
                start, end = segment_bounds[i]
                quantized_data[start:end, :] = q_seg
                scales[i, :] = scale
                zero_points[i, :] = zp

        return quantized_data, scales, zero_points

    def asymmetric_dequantize_int8_per_segment_parallel(self, quantized_data, scales, zero_points, num_segments, shape):
        N, K = shape
        num_segments = min(num_segments, N)
        indices = np.linspace(0, N, num_segments + 1, dtype=int)

        segment_bounds = [
            (indices[i], indices[i + 1])
            for i in range(num_segments)
            if indices[i] < indices[i + 1]
        ]

        reconstructed = np.zeros(shape, dtype=np.float32)

        def process_segment(i):
            start, end = segment_bounds[i]
            q = quantized_data[start:end, :]
            return i, (q.astype(np.float32) - zero_points[i][np.newaxis, :]) * scales[i][np.newaxis, :]

        with ThreadPoolExecutor() as executor:
            for future in executor.map(process_segment, range(len(segment_bounds))):
                i, seg = future
                start, end = segment_bounds[i]
                reconstructed[start:end, :] = seg

        return reconstructed

    # -------------------- Int4 Quantization -------------------- #

    def quantize_segment(self, segment_r, alpha_q=0, beta_q=15):
        alpha = segment_r.min(axis=0)
        beta = segment_r.max(axis=0)
        scale = (beta - alpha) / (beta_q - alpha_q)
        scale[scale == 0] = 1.0
        zero_point = alpha_q - alpha / scale

        quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
        quantized = np.clip(quantized, alpha_q, beta_q)

        quantized = quantized.reshape(segment_r.shape[0] // 2, 2, segment_r.shape[1])
        packed = self.pack_4bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :])
        return packed, scale, zero_point

    def asymmetric_quantize_int4_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(num_segments, N)
        segment_size = int(np.ceil(N / num_segments))
        segment_size += segment_size % 2

        total_padded = num_segments * segment_size
        padding = total_padded - N
        r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

        r_segments = r_padded.reshape(num_segments, segment_size, K)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.quantize_segment, r_segments))

        packed_segments, scales, zero_points = zip(*results)
        return (
            np.vstack(packed_segments),
            np.vstack(scales),
            np.vstack(zero_points),
            segment_size,
            N
        )

    def dequantize_segment(self, packed_segment, scale, zero_point, segment_size, K):
        a, b = self.unpack_8bit_to_4bit(packed_segment)
        quantized = np.empty((segment_size, K), dtype=np.float32)
        quantized[::2], quantized[1::2] = a, b

        scale = scale[np.newaxis, :]
        zero_point = zero_point[np.newaxis, :]
        return (quantized - zero_point) * scale

    def asymmetric_dequantize_int4_segment_parallel(self, packed_matrix, scales, zero_points, segment_size, N_original):
        K = packed_matrix.shape[1]
        segment_size += segment_size % 2
        num_segments = len(scales)
        per_segment = segment_size // 2

        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(
                    self.dequantize_segment,
                    packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                    scales[i],
                    zero_points[i],
                    segment_size,
                    K
                )
                for i in range(num_segments)
            ]
            dequantized = [f.result() for f in results]

        return np.vstack(dequantized)[:N_original, :]

    # -------------------- Int2 Quantization -------------------- #

    def quantize_segment_int2(self, segment_r, alpha_q=0, beta_q=3):
        alpha = segment_r.min(axis=0)
        beta = segment_r.max(axis=0)
        scale = (beta - alpha) / (beta_q - alpha_q)
        scale[scale == 0] = 1.0
        zero_point = alpha_q - alpha / scale

        quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
        quantized = np.clip(quantized, alpha_q, beta_q)

        quantized = quantized.reshape(segment_r.shape[0] // 4, 4, segment_r.shape[1])
        packed = self.pack_2bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :], quantized[:, 2, :], quantized[:, 3, :])
        return packed, scale, zero_point

    def asymmetric_quantize_int2_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(num_segments, N)
        segment_size = int(np.ceil(N / num_segments))
        if segment_size % 4 != 0:
            segment_size += 4 - (segment_size % 4)

        total_padded = num_segments * segment_size
        padding = total_padded - N
        r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

        r_segments = r_padded.reshape(num_segments, segment_size, K)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.quantize_segment_int2, r_segments))

        packed_segments, scales, zero_points = zip(*results)
        return (
            np.vstack(packed_segments),
            np.vstack(scales),
            np.vstack(zero_points),
            segment_size,
            N
        )

    def dequantize_segment_int2(self, packed_segment, scale, zero_point, segment_size, K):
        a, b, c, d = self.unpack_8bit_to_2bit(packed_segment)
        quantized = np.empty((segment_size, K), dtype=np.float32)
        quantized[::4], quantized[1::4], quantized[2::4], quantized[3::4] = a, b, c, d

        scale = scale[np.newaxis, :]
        zero_point = zero_point[np.newaxis, :]
        return (quantized - zero_point) * scale

    def asymmetric_dequantize_int2_segment_parallel(self, packed_matrix, scales, zero_points, segment_size, N_original):
        K = packed_matrix.shape[1]
        if segment_size % 4 != 0:
            segment_size += 4 - (segment_size % 4)

        per_segment = segment_size // 4
        num_segments = len(scales)

        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(
                    self.dequantize_segment_int2,
                    packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                    scales[i],
                    zero_points[i],
                    segment_size,
                    K
                )
                for i in range(num_segments)
            ]
            dequantized = [f.result() for f in results]

        return np.vstack(dequantized)[:N_original, :]

    def quantize(self, gaussians, path, num_segments=None,
                bits_f_dc=8, bits_f_rest=4,
                bits_opacity=4, bits_scaling=8, bits_rotation=8,
                prune_rate=None, sh_rate=None):
        t_last = time.time()
        t_log = []
        def stamp(name):
            nonlocal t_last
            now = time.time()
            t_log.append((name, now - t_last))
            t_last = now

        # Extract features; keep tensors on device for preprocessing
        xyz_t = gaussians._xyz.detach()
        f_dc_t = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        f_rest_t = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        opacities_t = gaussians._opacity.detach()
        scale_t = gaussians._scaling.detach()
        rotation_t = gaussians.rotation_activation(gaussians._rotation).detach()
        N = xyz_t.shape[0]
        stamp("prep_tensors")

        zero_mask_t = torch.all(f_rest_t == 0, dim=1)
        nonzero_idx = torch.nonzero(~zero_mask_t, as_tuple=False).flatten()
        zero_idx = torch.nonzero(zero_mask_t, as_tuple=False).flatten()
        stamp("zero_mask_split")

        sh_expected_ratio = None
        if prune_rate is not None and sh_rate is not None:
            denom = max(1e-6, 1.0 - float(prune_rate))
            sh_expected_ratio = max(0.0, min(1.0, 1.0 - float(sh_rate) / denom))

        self.latest_attr_stats = _log_attr_sizes_from_gaussians(
            gaussians,
            "Current attribute sizes (post-prune):",
            sh_zero_mask=zero_mask_t,
            sh_expected_ratio=sh_expected_ratio,
            log=False
        )

        if self.xyz_codec == 'morton16':
            q_all_t, xyz_quant_mins, xyz_quant_maxs = self._quantize_xyz16_t(xyz_t)
            nonzero_sorted, zero_sorted = self._morton_reorder_indices(q_all_t, nonzero_idx, zero_idx)
            sorted_idx = torch.cat([nonzero_sorted, zero_sorted], dim=0)
            qxyz_sorted_t = q_all_t.index_select(0, sorted_idx)
            stamp("xyz_quant_morton")
        elif self.xyz_codec == 'morton8':
            q_all_t, xyz_quant_mins, xyz_quant_maxs = self._quantize_xyz16_t(xyz_t)
            nonzero_sorted, zero_sorted = self._morton_reorder_indices(q_all_t, nonzero_idx, zero_idx)
            sorted_idx = torch.cat([nonzero_sorted, zero_sorted], dim=0)
            qxyz_sorted_t = q_all_t.index_select(0, sorted_idx)
            stamp("xyz_quant_morton")
        elif self.xyz_codec == 'morton32':
            q_all_t, xyz_quant_mins, xyz_quant_maxs = self._quantize_xyz16_t(xyz_t)
            nonzero_sorted, zero_sorted = self._morton_reorder_indices32(q_all_t, nonzero_idx, zero_idx)
            sorted_idx = torch.cat([nonzero_sorted, zero_sorted], dim=0)
            qxyz_sorted_t = q_all_t.index_select(0, sorted_idx)
            stamp("xyz_quant_morton")
        else:
            nonzero_sorted = nonzero_idx
            zero_sorted = zero_idx
            sorted_idx = torch.cat([nonzero_sorted, zero_sorted], dim=0)
            qxyz_sorted_t = None
            xyz_quant_mins = xyz_quant_maxs = None
            stamp("xyz_reorder_raw")

        boundary_index = int(nonzero_sorted.shape[0])

        xyz = xyz_t.index_select(0, sorted_idx).cpu().numpy() if self.xyz_codec not in {'morton16', 'morton8', 'morton32'} else None
        f_dc = f_dc_t.index_select(0, sorted_idx).cpu().numpy()
        f_rest = f_rest_t.index_select(0, nonzero_sorted).cpu().numpy()
        opacities = opacities_t.index_select(0, sorted_idx).cpu().numpy()
        scale = scale_t.index_select(0, sorted_idx).cpu().numpy()
        rotation = rotation_t.index_select(0, sorted_idx).cpu().numpy()
        stamp("cpu_transfer")

        num_segments = self._normalize_segments(num_segments, N)

        npz_data = {
            'bits_f_rest': bits_f_rest,
            'sh_all_zero': f_rest.shape[0] == 0,
            'boundary_index': boundary_index
        }

        # Encode XYZ depending on codec
        if self.xyz_codec == 'morton16':
            qxyz_sorted = qxyz_sorted_t.cpu().numpy().astype(np.uint16)
            npz_data.update({'xyz_codec': 'morton16', 'xyz_qbits': self.xyz_qbits,
                             'xyz_mins': xyz_quant_mins, 'xyz_maxs': xyz_quant_maxs})
            if self.use_delta_encoding and qxyz_sorted.shape[0] > 0:
                base, zz = self._delta_encode(qxyz_sorted)
                payload = zz.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(payload)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_base': base, 'xyz_delta_compressed': compressed,
                                     'xyz_delta_shape_uint32': (N - 1, 3),
                                     'xyz_delta_view_shape': payload.shape,
                                     'xyz_delta_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_base': base, 'xyz_delta_raw': zz.astype(np.uint32)})
                stamp("xyz_delta_encode")
            else:
                # No delta: store raw q16 as bytes and compress
                qbytes_view = qxyz_sorted.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(qbytes_view)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_q16_compressed': compressed,
                                     'xyz_q16_shape_uint16': (N, 3),
                                     'xyz_q16_view_shape': qbytes_view.shape,
                                     'xyz_q16_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_q16': qxyz_sorted})
                stamp("xyz_q16_encode")
        elif self.xyz_codec == 'morton8':
            qxyz_sorted = qxyz_sorted_t.cpu().numpy().astype(np.uint8)
            npz_data.update({'xyz_codec': 'morton8', 'xyz_qbits': self.xyz_qbits,
                             'xyz_mins': xyz_quant_mins, 'xyz_maxs': xyz_quant_maxs})
            if self.use_delta_encoding and qxyz_sorted.shape[0] > 0:
                base, zz = self._delta_encode(qxyz_sorted)
                payload = zz.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(payload)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_base': base, 'xyz_delta_compressed': compressed,
                                     'xyz_delta_shape_uint32': (N - 1, 3),
                                     'xyz_delta_view_shape': payload.shape,
                                     'xyz_delta_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_base': base, 'xyz_delta_raw': zz.astype(np.uint32)})
                stamp("xyz_delta_encode")
            else:
                qbytes_view = qxyz_sorted.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(qbytes_view)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_q8_compressed': compressed,
                                     'xyz_q8_shape_uint8': (N, 3),
                                     'xyz_q8_view_shape': qbytes_view.shape,
                                     'xyz_q8_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_q8': qxyz_sorted})
                stamp("xyz_q8_encode")
        elif self.xyz_codec == 'morton32':
            qxyz_sorted = qxyz_sorted_t.cpu().numpy().astype(np.uint32)
            npz_data.update({'xyz_codec': 'morton32', 'xyz_qbits': self.xyz_qbits,
                             'xyz_mins': xyz_quant_mins, 'xyz_maxs': xyz_quant_maxs})
            if self.use_delta_encoding and qxyz_sorted.shape[0] > 0:
                base, zz = self._delta_encode(qxyz_sorted)
                payload = zz.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(payload)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_base': base, 'xyz_delta_compressed': compressed,
                                     'xyz_delta_shape': (N - 1, 3),
                                     'xyz_delta_dtype': str(zz.dtype),
                                     'xyz_delta_view_shape': payload.shape,
                                     'xyz_delta_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_base': base, 'xyz_delta_raw': zz, 'xyz_delta_dtype': str(zz.dtype)})
                stamp("xyz_delta_encode")
            else:
                qbytes_view = qxyz_sorted.view(np.uint8)
                compressed, shape_bytes = self.entropy_encode(qbytes_view)
                if self.use_entropy_coding:
                    npz_data.update({'xyz_q_compressed': compressed,
                                     'xyz_q_shape': (N, 3),
                                     'xyz_q_dtype': str(qxyz_sorted.dtype),
                                     'xyz_q_view_shape': qbytes_view.shape,
                                     'xyz_q_shape_bytes': shape_bytes})
                else:
                    npz_data.update({'xyz_q': qxyz_sorted, 'xyz_q_dtype': str(qxyz_sorted.dtype)})
                stamp("xyz_q32_encode")
        else:
            if self.xyz_codec == 'raw32':
                npz_data['xyz_codec'] = 'raw32'
                npz_data['xyz'] = xyz.astype(np.float32)
                stamp("xyz_fp32_store")
            elif self.xyz_codec == 'raw16':
                npz_data['xyz_codec'] = 'raw16'
                npz_data['xyz'] = xyz.astype(np.float16)
                stamp("xyz_fp16_store")
            elif self.xyz_codec == 'raw8':
                qxyz, xyz_quant_mins, xyz_quant_maxs = self._quantize_xyz16(xyz)
                npz_data.update({'xyz_codec': 'raw8', 'xyz_qbits': self.xyz_qbits,
                                 'xyz_mins': xyz_quant_mins, 'xyz_maxs': xyz_quant_maxs,
                                 'xyz_q8': qxyz.astype(np.uint8)})
                stamp("xyz_q8_store")
            else:
                raise ValueError(f"Unexpected xyz_codec {self.xyz_codec}")

        if not npz_data['sh_all_zero']:
            if bits_f_rest == 2:
                q, S, Z, seg_size, Nq = self.asymmetric_quantize_int2_segment_parallel(f_rest, num_segments)
                q_int8 = q.astype(np.int8)
                compressed, shape = self.entropy_encode(q_int8)
                if self.use_entropy_coding:
                    npz_data.update({'M_rest_compressed': compressed, 'M_rest_shape': shape, 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
                else:
                    npz_data.update({'M_rest': q_int8, 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
                stamp("f_rest_int2")
            elif bits_f_rest == 4:
                q, S, Z, seg_size, Nq = self.asymmetric_quantize_int4_segment_parallel(f_rest, num_segments)
                q_int8 = q.astype(np.int8)
                compressed, shape = self.entropy_encode(q_int8)
                if self.use_entropy_coding:
                    npz_data.update({'M_rest_compressed': compressed, 'M_rest_shape': shape, 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
                else:
                    npz_data.update({'M_rest': q_int8, 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
                stamp("f_rest_int4")
            elif bits_f_rest == 8:
                q, S, Z = self.asymmetric_quantize_int8_per_segment_parallel(f_rest, num_segments)
                q_int8 = q.astype(np.int8)
                compressed, shape = self.entropy_encode(q_int8)
                if self.use_entropy_coding:
                    npz_data.update({'M_rest_compressed': compressed, 'M_rest_shape': shape, 'S_rest': S, 'Z_rest': Z})
                else:
                    npz_data.update({'M_rest': q_int8, 'S_rest': S, 'Z_rest': Z})
                stamp("f_rest_int8")
            else:
                npz_data['f_rest'] = f_rest.astype(np.float16)
                stamp("f_rest_fp16")

        bits_map = {'f_dc': bits_f_dc, 'opacities': bits_opacity, 'scale': bits_scaling, 'rotation': bits_rotation}
        var_map = {'f_dc': f_dc, 'opacities': opacities, 'scale': scale, 'rotation': rotation}
        bit_groups = {2: [], 4: [], 8: [], 16: []}

        for k, b in bits_map.items():
            if b in bit_groups:
                bit_groups[b].append(k)

        if bit_groups[2]:
            int2_attributes = np.concatenate([var_map[k] for k in bit_groups[2]], axis=1)
            q, S, Z, seg_size, Nq = self.asymmetric_quantize_int2_segment_parallel(int2_attributes, num_segments)
            q_int8 = q.astype(np.int8)
            compressed, shape = self.entropy_encode(q_int8)
            if self.use_entropy_coding:
                npz_data.update({'M_int2_compressed': compressed, 'M_int2_shape': shape, 'S_int2': S, 'Z_int2': Z, 'segment_size_int2': seg_size, 'N_int2': Nq})
            else:
                npz_data.update({'M_int2': q_int8, 'S_int2': S, 'Z_int2': Z, 'segment_size_int2': seg_size, 'N_int2': Nq})
            stamp("int2_pack")
        if bit_groups[4]:
            int4_attributes = np.concatenate([var_map[k] for k in bit_groups[4]], axis=1)
            q, S, Z, seg_size, Nq = self.asymmetric_quantize_int4_segment_parallel(int4_attributes, num_segments)
            q_int8 = q.astype(np.int8)
            compressed, shape = self.entropy_encode(q_int8)
            if self.use_entropy_coding:
                npz_data.update({'M_int4_compressed': compressed, 'M_int4_shape': shape, 'S_int4': S, 'Z_int4': Z, 'segment_size_int4': seg_size, 'N_int4': Nq})
            else:
                npz_data.update({'M_int4': q_int8, 'S_int4': S, 'Z_int4': Z, 'segment_size_int4': seg_size, 'N_int4': Nq})
            stamp("int4_pack")
        if bit_groups[8]:
            int8_attributes = np.concatenate([var_map[k] for k in bit_groups[8]], axis=1)
            q, S, Z = self.asymmetric_quantize_int8_per_segment_parallel(int8_attributes, num_segments)
            q_int8 = q.astype(np.int8)
            compressed, shape = self.entropy_encode(q_int8)
            if self.use_entropy_coding:
                npz_data.update({'M_int8_compressed': compressed, 'M_int8_shape': shape, 'S_int8': S, 'Z_int8': Z})
            else:
                npz_data.update({'M_int8': q_int8, 'S_int8': S, 'Z_int8': Z})
            stamp("int8_pack")
        if bit_groups[16]:
            fp16_attributes = np.concatenate([var_map[k] for k in bit_groups[16]], axis=1)
            npz_data['M_fp16'] = fp16_attributes.astype(np.float16)
            stamp("fp16_pack")

        npz_data.update({
            'N': N,
            'num_segments': num_segments,
            # xyz stored above according to codec
            'int2_names': bit_groups[2],
            'int4_names': bit_groups[4],
            'int8_names': bit_groups[8],
            'fp16_names': bit_groups[16],
            'npz_path': os.path.join(path, 'quantization_parameters_pipeline.npz'),
            'use_entropy_coding': self.use_entropy_coding,
            'xyz_codec_used': self.xyz_codec,
            'compression_method': self.compression_method,
            'compression_level': self.compression_level
        })

        if t_log:
            breakdown = ", ".join([f"{name}: {dt:.3f}s" for name, dt in t_log])
            print(f"  Quantization breakdown -> {breakdown}")
        return npz_data

    def dequantize(self, gaussians, npz_data):
        N = npz_data['N']
        num_segments = npz_data['num_segments']
        features_extra = np.zeros((N, 45), dtype=np.float32)
        
        # Check if entropy coding was used
        use_entropy = npz_data.get('use_entropy_coding', False)
        compression_method = npz_data.get('compression_method', 'zlib')

        if not npz_data['sh_all_zero']:
            if npz_data['bits_f_rest'] == 2:
                if use_entropy and 'M_rest_compressed' in npz_data:
                    M_rest = self.entropy_decode(npz_data['M_rest_compressed'], npz_data['M_rest_shape'], np.int8, compression_method)
                else:
                    M_rest = npz_data['M_rest']
                deq = self.asymmetric_dequantize_int2_segment_parallel(
                    M_rest, npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif npz_data['bits_f_rest'] == 4:
                if use_entropy and 'M_rest_compressed' in npz_data:
                    M_rest = self.entropy_decode(npz_data['M_rest_compressed'], npz_data['M_rest_shape'], np.int8, compression_method)
                else:
                    M_rest = npz_data['M_rest']
                deq = self.asymmetric_dequantize_int4_segment_parallel(
                    M_rest, npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif npz_data['bits_f_rest'] == 8:
                if use_entropy and 'M_rest_compressed' in npz_data:
                    M_rest = self.entropy_decode(npz_data['M_rest_compressed'], npz_data['M_rest_shape'], np.int8, compression_method)
                else:
                    M_rest = npz_data['M_rest']
                deq = self.asymmetric_dequantize_int8_per_segment_parallel(
                    M_rest, npz_data['S_rest'], npz_data['Z_rest'],
                    num_segments, M_rest.shape
                )
            else:
                deq = npz_data['f_rest'].astype(np.float32)

            features_extra[:npz_data['boundary_index'], :] = deq

        features_extra = features_extra.reshape((N, 3, (gaussians.max_sh_degree + 1) ** 2 - 1))
        length_map = {'f_dc': 3, 'opacities': 1, 'scale': 3, 'rotation': 4}
        new_var = {}

        for bits, key in [(2, 'int2'), (4, 'int4'), (8, 'int8'), (16, 'fp16')]:
            names = npz_data.get(f'{key}_names', [])
            if not names:
                continue

            if bits in [2, 4, 8]:
                if bits == 2:
                    if use_entropy and 'M_int2_compressed' in npz_data:
                        M_int2 = self.entropy_decode(npz_data['M_int2_compressed'], npz_data['M_int2_shape'], np.int8, compression_method)
                    else:
                        M_int2 = npz_data['M_int2']
                    deq = self.asymmetric_dequantize_int2_segment_parallel(
                        M_int2, npz_data['S_int2'], npz_data['Z_int2'],
                        npz_data['segment_size_int2'], npz_data['N_int2']
                    )
                elif bits == 4:
                    if use_entropy and 'M_int4_compressed' in npz_data:
                        M_int4 = self.entropy_decode(npz_data['M_int4_compressed'], npz_data['M_int4_shape'], np.int8, compression_method)
                    else:
                        M_int4 = npz_data['M_int4']
                    deq = self.asymmetric_dequantize_int4_segment_parallel(
                        M_int4, npz_data['S_int4'], npz_data['Z_int4'],
                        npz_data['segment_size_int4'], npz_data['N_int4']
                    )
                else:
                    if use_entropy and 'M_int8_compressed' in npz_data:
                        M_int8 = self.entropy_decode(npz_data['M_int8_compressed'], npz_data['M_int8_shape'], np.int8, compression_method)
                    else:
                        M_int8 = npz_data['M_int8']
                    deq = self.asymmetric_dequantize_int8_per_segment_parallel(
                        M_int8, npz_data['S_int8'], npz_data['Z_int8'],
                        num_segments, M_int8.shape
                    )

                start = 0
                for name in names:
                    width = length_map[name]
                    new_var[name] = deq[:, start:start + width]
                    start += width
            elif bits == 16:
                start = 0
                fp16 = npz_data['M_fp16'].astype(np.float32)
                for name in names:
                    width = length_map[name]
                    new_var[name] = fp16[:, start:start + width]
                    start += width

        features_dc = np.zeros((N, 3, 1), dtype=np.float32)
        features_dc[:, :, 0] = new_var['f_dc']

        # Decode / load XYZ depending on codec used
        xyz_codec = npz_data.get('xyz_codec_used', npz_data.get('xyz_codec', 'raw32'))
        if xyz_codec == 'morton16':
            xyz_qbits = int(npz_data.get('xyz_qbits', self.xyz_qbits))
            mins = np.array(npz_data['xyz_mins'], dtype=np.float32)
            maxs = np.array(npz_data['xyz_maxs'], dtype=np.float32)
            if npz_data.get('xyz_delta_compressed', None) is not None:
                zz_bytes = self.entropy_decode(npz_data['xyz_delta_compressed'], npz_data['xyz_delta_shape_bytes'], np.uint8, compression_method)
                zz_view = zz_bytes.reshape(npz_data['xyz_delta_view_shape'])
                zz = zz_view.view(np.uint32).reshape(npz_data['xyz_delta_shape_uint32'])
                base = np.array(npz_data['xyz_base'], dtype=np.uint16)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            elif 'xyz_delta_raw' in npz_data:
                base = np.array(npz_data['xyz_base'], dtype=np.uint16)
                zz = npz_data['xyz_delta_raw'].astype(np.uint32)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            else:
                if use_entropy and 'xyz_q16_compressed' in npz_data:
                    qbytes = self.entropy_decode(npz_data['xyz_q16_compressed'], npz_data['xyz_q16_shape_bytes'], np.uint8, compression_method)
                    q_view = qbytes.reshape(npz_data['xyz_q16_view_shape'])
                    qxyz = q_view.view(np.uint16).reshape(npz_data['xyz_q16_shape_uint16'])
                else:
                    qxyz = npz_data['xyz_q16'].astype(np.uint16)
            xyz_f32 = self._dequantize_xyz16(qxyz, mins, maxs, qbits=xyz_qbits)
        elif xyz_codec == 'morton8':
            xyz_qbits = int(npz_data.get('xyz_qbits', self.xyz_qbits))
            mins = np.array(npz_data['xyz_mins'], dtype=np.float32)
            maxs = np.array(npz_data['xyz_maxs'], dtype=np.float32)
            if npz_data.get('xyz_delta_compressed', None) is not None:
                zz_bytes = self.entropy_decode(npz_data['xyz_delta_compressed'], npz_data['xyz_delta_shape_bytes'], np.uint8, compression_method)
                zz_view = zz_bytes.reshape(npz_data['xyz_delta_view_shape'])
                zz = zz_view.view(np.uint32).reshape(npz_data['xyz_delta_shape_uint32'])
                base = np.array(npz_data['xyz_base'], dtype=np.uint16)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            elif 'xyz_delta_raw' in npz_data:
                base = np.array(npz_data['xyz_base'], dtype=np.uint16)
                zz = npz_data['xyz_delta_raw'].astype(np.uint32)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            else:
                if use_entropy and 'xyz_q8_compressed' in npz_data:
                    qbytes = self.entropy_decode(npz_data['xyz_q8_compressed'], npz_data['xyz_q8_shape_bytes'], np.uint8, compression_method)
                    q_view = qbytes.reshape(npz_data['xyz_q8_view_shape'])
                    qxyz = q_view.view(np.uint8).reshape(npz_data['xyz_q8_shape_uint8'])
                else:
                    qxyz = npz_data['xyz_q8'].astype(np.uint8)
            xyz_f32 = self._dequantize_xyz16(qxyz, mins, maxs, qbits=xyz_qbits)
        elif xyz_codec == 'morton32':
            xyz_qbits = int(npz_data.get('xyz_qbits', self.xyz_qbits))
            mins = np.array(npz_data['xyz_mins'], dtype=np.float32)
            maxs = np.array(npz_data['xyz_maxs'], dtype=np.float32)
            if npz_data.get('xyz_delta_compressed', None) is not None:
                zz_bytes = self.entropy_decode(npz_data['xyz_delta_compressed'], npz_data['xyz_delta_shape_bytes'], np.uint8, compression_method)
                zz_view = zz_bytes.reshape(npz_data['xyz_delta_view_shape'])
                zz_dtype = np.dtype(npz_data.get('xyz_delta_dtype', 'uint64'))
                zz = zz_view.view(zz_dtype).reshape(npz_data['xyz_delta_shape'])
                base = np.array(npz_data['xyz_base'], dtype=np.uint32)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            elif 'xyz_delta_raw' in npz_data:
                zz_dtype = np.dtype(npz_data.get('xyz_delta_dtype', 'uint64'))
                base = np.array(npz_data['xyz_base'], dtype=np.uint32)
                zz = npz_data['xyz_delta_raw'].astype(zz_dtype)
                qxyz = self._delta_decode(base, zz, qbits=xyz_qbits)
            else:
                if use_entropy and 'xyz_q_compressed' in npz_data:
                    qbytes = self.entropy_decode(npz_data['xyz_q_compressed'], npz_data['xyz_q_shape_bytes'], np.uint8, compression_method)
                    q_view = qbytes.reshape(npz_data['xyz_q_view_shape'])
                    q_dtype = np.dtype(npz_data.get('xyz_q_dtype', 'uint32'))
                    qxyz = q_view.view(q_dtype).reshape(npz_data['xyz_q_shape'])
                else:
                    qxyz = npz_data['xyz_q'].astype(np.uint32)
            xyz_f32 = self._dequantize_xyz16(qxyz, mins, maxs, qbits=xyz_qbits)
        elif xyz_codec == 'raw16':
            xyz_f32 = npz_data['xyz'].astype(np.float32)
        elif xyz_codec == 'raw8':
            xyz_qbits = int(npz_data.get('xyz_qbits', self.xyz_qbits))
            mins = np.array(npz_data['xyz_mins'], dtype=np.float32)
            maxs = np.array(npz_data['xyz_maxs'], dtype=np.float32)
            qxyz = npz_data['xyz_q8'].astype(np.uint8)
            xyz_f32 = self._dequantize_xyz16(qxyz, mins, maxs, qbits=xyz_qbits)
        elif xyz_codec == 'raw32' or xyz_codec == 'raw':
            # raw kept for backward compatibility; treat as raw32
            xyz_f32 = npz_data['xyz'].astype(np.float32)
        else:
            raise ValueError(f"Unsupported xyz_codec in npz: {xyz_codec}")

        gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz_f32, device="cuda").requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._opacity = torch.nn.Parameter(torch.tensor(new_var['opacities'], device="cuda").requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(torch.tensor(new_var['scale'], device="cuda").requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(torch.tensor(new_var['rotation'], device="cuda").requires_grad_(True))
        gaussians.active_sh_degree = gaussians.max_sh_degree


from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
import shutil
import copy

@dataclass(frozen=True)
class SearchConfig:
    pruning_rate: float
    sh_rate: float

class Searcher:
    def __init__(
        self,
        args,
        dataset,
        gaussians,
        pipe,
        scene,
        imp_score,
        filesize_input,
        search_space=None,
        target_psnr_drop=1.0,
        save_render=False
    ):
        self.args = args
        self.dataset = dataset
        self.gaussians = gaussians
        self.pipe = pipe
        self.scene = scene
        self.imp_score = imp_score
        self.filesize_input = filesize_input
        self.target_psnr_drop = target_psnr_drop
        self.save_render = save_render

        self.Pruner = Pruner()
        # Initialize Quantizer with entropy coding enabled by default
        compression_method = getattr(args, 'compression_method', None)
        use_entropy = getattr(args, 'use_entropy_coding', True)
        if compression_method is None:
            use_entropy = False
        segments = getattr(args, 'segments', None)
        xyz_codec = getattr(args, 'xyz_codec', 'raw')
        use_delta = getattr(args, 'use_delta_encoding', False)
        compression_level = getattr(args, 'compression_level', None)
        self.Quantizer = Quantizer(segments=segments, use_entropy_coding=use_entropy,
                                   xyz_codec=xyz_codec, use_delta_encoding=use_delta,
                                   compression_method=compression_method,
                                   compression_level=compression_level)

        self._explicit_search_space = None
        self._use_default_search = False
        if search_space == 'default':
            self._use_default_search = True
        else:
            self._explicit_search_space = self._init_search_space(search_space)
        self.search_space = None
        self.last_prune_psnr = None
        self.baseline_psnr = None
        self._best_mem_score = None
        self._best_psnr_drop = float('inf')
        self.neighbor_budget = max(0, int(getattr(args, 'search_neighbor_budget', 20)))
        self.greedy_budget = max(0, int(getattr(args, 'search_greedy_budget', 50)))
        # Track search phase timings
        self.stage1_time = 0.0  # seed/coarse search
        self.stage2_time = 0.0  # neighbor/refine search
        self.stage3_time = 0.0  # finalize best config (quant/dequant + final eval)
        self._original_attr_stats = None
        self._logged_original_sizes = False

    def _init_search_space(self, search_space):
        if search_space == 'default':
            return None
        return [SearchConfig(*cfg) if not isinstance(cfg, SearchConfig) else cfg for cfg in search_space]

    def _get_search_space(self, center_cfg=None):
        if self._explicit_search_space is not None:
            return self._explicit_search_space
        if not self._use_default_search:
            return []
        return list(self._generate_default_search_space(center_cfg))

    def _build_range(self, min_v, max_v, step, clamp_min=0.0, clamp_max=0.95):
        eps = 1e-9
        step = step if step and step > 0 else 0.05
        lo, hi = float(min(min_v, max_v)), float(max(min_v, max_v))
        values = []
        cur = lo
        while cur <= hi + eps:
            values.append(round(cur, 4))
            cur += step
        values.append(round(hi, 4))
        clamped = [min(max(v, clamp_min), clamp_max) for v in values]
        return sorted(set(clamped))

    def _prioritize_values(self, values, target):
        if target is None:
            return values
        if target not in values:
            values = sorted(set(values + [round(target, 4)]))
        return sorted(values, key=lambda v: (abs(v - target), v))

    def _generate_default_search_space(self, center_cfg=None):
        prune_vals = self._build_range(
            getattr(self.args, 'search_prune_min', 0.4),
            getattr(self.args, 'search_prune_max', 0.8),
            getattr(self.args, 'search_prune_step', 0.01),
            clamp_max=0.99
        )
        sh_vals = self._build_range(
            getattr(self.args, 'search_sh_min', 0.1),
            getattr(self.args, 'search_sh_max', 0.6),
            getattr(self.args, 'search_sh_step', 0.01),
            clamp_max=0.7
        )
        if center_cfg is not None:
            prune_vals = self._prioritize_values(prune_vals, center_cfg.pruning_rate)
            sh_vals = self._prioritize_values(sh_vals, center_cfg.sh_rate)
        for p in prune_vals:
            for s in sh_vals:
                yield SearchConfig(p, s)

    def _mem_saving_score(self, cfg: SearchConfig):
        return 1.8 * cfg.pruning_rate + cfg.sh_rate

    def _cfg_key(self, pruning_rate, sh_rate):
        return (round(pruning_rate, 4), round(sh_rate, 4))

    def _nearest_grid_cfg(self, pruning_rate, sh_rate, cfg_lookup):
        key = self._cfg_key(pruning_rate, sh_rate)
        if key in cfg_lookup:
            return cfg_lookup[key]
        best_cfg = None
        best_diff = float('inf')
        for cfg in cfg_lookup.values():
            diff = abs(cfg.pruning_rate - pruning_rate) + abs(cfg.sh_rate - sh_rate)
            if diff < best_diff:
                best_diff = diff
                best_cfg = cfg
        return best_cfg

    def _initial_seed_configs(self):
        """Return seed configurations, sorted by mem_score (descending) for better early stopping.
        
        Sorting by mem_score helps prioritize configurations that are more likely to meet
        the target PSNR drop, enabling more effective early stopping.
        """
        # seeds = [
        #     (0.4, 0.3), (0.4, 0.4), (0.4, 0.5), (0.6, 0.3), (0.7, 0.2)
        # ]
        seeds = [
            (0.4, 0.3), (0.4, 0.4), (0.4, 0.5), (0.6, 0.3), (0.7, 0.2)
        ]
        # Sort by mem_score (descending): mem_score = 1.8 * pruning_rate + sh_rate
        # This prioritizes configurations with higher compression potential
        seeds_sorted = sorted(seeds, key=lambda x: 1.8 * x[0] + x[1], reverse=True)
        return seeds_sorted
        # return [  
        #     (0.4, 0.3), (0.4, 0.4), (0.4, 0.5), (0.6, 0.3), (0.7, 0.2), (0.7, 0.3)
        # ]

    def _evaluate_config_once(self, cfg, threshold):
        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            current_psnr, _, _ = self.spin_once(cfg.pruning_rate, cfg.sh_rate, full_eval=False)
        logs = log_buffer.getvalue()
        prune_psnr = self.last_prune_psnr if self.last_prune_psnr is not None else current_psnr
        psnr_drop = max(0.0, self.baseline_psnr - prune_psnr)
        meets = prune_psnr >= threshold
        return {
            "cfg": cfg,
            "prune_psnr": prune_psnr,
            "psnr_drop": psnr_drop,
            "mem_score": self._mem_saving_score(cfg),
            "meets": meets,
            "logs": logs
        }


    def render_set_pipeline(self, views, gaussians, pipeline, background):
        psnrs, ssims, lpipss = [], [], []

        if self.save_render:
            render_path = os.path.join(self.tgt_path, "test/renders")
            gt_path = os.path.join(self.tgt_path, "test/gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)

        for idx, view in enumerate(views):
            rendering = render(view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            psnrs.append(psnr(rendering, gt).mean().item())
            # ssims.append(ssim(rendering, gt).mean().item())
            # lpipss.append(lpips(rendering, gt, net_type="vgg").mean().item())

            if self.save_render:
                torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
                torchvision.utils.save_image(gt, os.path.join(gt_path, f"{idx:05d}.png"))

        print ("psnr: ", np.mean(np.array(psnrs)))
        # print ("ssim: ", np.mean(np.array(ssims)))
        # print ("lpips: ", np.mean(np.array(lpipss)))

        return psnrs

    def render_sets_pipeline(self, gaussians : GaussianModel, skip_train : bool, skip_test : bool, load_iteration=None):
        with torch.no_grad():
            bg_color = torch.tensor([1, 1, 1] if self.dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            final_scores = []
            if not skip_train:
                final_scores += self.render_set_pipeline(self.scene.getTrainCameras(), gaussians, self.pipe, bg_color)
            if not skip_test:
                final_scores += self.render_set_pipeline(self.scene.getTestCameras(), gaussians, self.pipe, bg_color)
            return np.mean(final_scores)
    
    def spin_once(self, pruning_rate, sh_rate, save_render=False, full_eval=False):
        self.save_render = save_render
        configure = f"prune_{pruning_rate:.2f}_sh_{sh_rate:.2f}"
        out_path = os.path.join(self.tgt_path, "point_cloud", configure)
        # os.makedirs(out_path, exist_ok=True)

        if not self._logged_original_sizes:
            self._original_attr_stats = _log_attr_sizes_from_gaussians(
                self.gaussians,
                "Original PLY attribute sizes:",
                log=False
            )
            self._logged_original_sizes = True

        # Deep copy original gaussians
        start_time = time.time()
        gaussians = copy.deepcopy(self.gaussians)
        print(f"  Deep copy time:         {time.time() - start_time:.4f} s")

        start_time = time.time()
        self.Pruner.run(pruning_rate, sh_rate, torch.from_numpy(self.imp_score), gaussians)
        print(f"  Prune time:             {time.time() - start_time:.2f} s")

        # Render after pruning
        start_time = time.time()
        current_psnr = self.render_sets_pipeline(
            gaussians, skip_train=True, skip_test=False
        )
        print(f"  Render time: {time.time() - start_time:.2f}s")
        print(f"  PSNR after pruning: {current_psnr:.4f}")
        self.last_prune_psnr = current_psnr
        if self.baseline_psnr is not None:
            prune_drop = self.baseline_psnr - current_psnr
            print(f"  ΔPSNR after pruning: {prune_drop:.4f} dB")
            if prune_drop > self.target_psnr_drop:
                print(f"  Exceeds target drop ({self.target_psnr_drop:.4f} dB); skipping quantization for this config.")
                return current_psnr, None, None

        if not full_eval:
            return current_psnr, None, None

        # Quantize
        start_time = time.time()
        dic = self.Quantizer.quantize(
            gaussians, out_path, self.args.segments,
            bits_f_dc=8, bits_f_rest=4,
            bits_opacity=8, bits_scaling=8, bits_rotation=8,
            prune_rate=pruning_rate, sh_rate=sh_rate
        )
        # dic = self.Quantizer.quantize(
        #     gaussians, out_path, self.args.segments,
        #     bits_f_dc=8, bits_f_rest=4,
        #     bits_opacity=8, bits_scaling=8, bits_rotation=4,
        #     prune_rate=pruning_rate, sh_rate=sh_rate
        # )
        print(f"  Quantization time:      {time.time() - start_time:.2f} s")

        # Dequantize
        start_time = time.time()
        self.Quantizer.dequantize(gaussians, dic)
        print(f"  Dequantization time:    {time.time() - start_time:.2f} s")

        # Estimate size (dictionary-based)
        filesize_est = estimate_dict_size(dic)
        print(f"  Estimated size:         {filesize_est / (1024 * 1024):.2f} MB")
        if self.filesize_input:
            ratio = filesize_est / self.filesize_input * 100.0
            print(f"  Storage ratio vs input: {ratio:.2f}%")

        if getattr(self.args, 'print_storage_breakdown', True):
            log_top_storage_entries(
                dic,
                top_k=len(dic),
                original_attr_stats=self._original_attr_stats,
                current_attr_stats=self.Quantizer.latest_attr_stats
            )

        # Evaluate quality
        start_time = time.time()
        current_psnr = self.render_sets_pipeline(
            gaussians, skip_train=True, skip_test=False
        )
        print(f"  Final render time:      {time.time() - start_time:.2f} s")
        print(f"  Final PSNR:             {current_psnr:.4f} dB")
        print(f"  ΔPSNR from input:       {self.baseline_psnr - current_psnr:.4f} dB")
        return current_psnr, filesize_est, dic

    def run_search(self):
        search_start_time = time.time()
        self.stage1_time = 0.0
        self.stage2_time = 0.0
        self.stage3_time = 0.0
        stage2_start = None
        src_path = self.args.model_path
        scene_name = os.path.basename(os.path.normpath(src_path))
        self.tgt_path = os.path.join(self.args.output_path, scene_name)
        os.makedirs(self.tgt_path, exist_ok=True)

        print(f"\n[Baseline] ...")
        start_time = time.time()
        self.baseline_psnr = self.render_sets_pipeline(
            self.gaussians, skip_train=True, skip_test=False, load_iteration=30000
        )
        print(f"  Baseline render time:   {time.time() - start_time:.2f} s")
        print(f"  Baseline PSNR:          {self.baseline_psnr:.4f} dB")

        threshold = self.baseline_psnr - self.target_psnr_drop
        visited = set()
        selected_cfg = None
        selected_prune_psnr = None
        best_mem = -float('inf')
        best_drop = float('inf')
        neighbor_queue = []
        queued = set()
        evaluation_count = 0
        best_failed_cfg = None
        best_failed_drop = float('inf')
        success_frontier = []
        failure_frontier = []
        dominance_eps = 1e-6

        def record_success(cfg):
            nonlocal success_frontier
            pr, sh = cfg.pruning_rate, cfg.sh_rate
            for spr, ssh in success_frontier:
                if spr >= pr - dominance_eps and ssh >= sh - dominance_eps:
                    return
            success_frontier = [
                (spr, ssh)
                for spr, ssh in success_frontier
                if not (pr >= spr - dominance_eps and sh >= ssh - dominance_eps)
            ]
            success_frontier.append((pr, sh))

        def record_failure(cfg):
            nonlocal failure_frontier
            pr, sh = cfg.pruning_rate, cfg.sh_rate
            for fpr, fsh in failure_frontier:
                if fpr <= pr + dominance_eps and fsh <= sh + dominance_eps:
                    return
            failure_frontier = [
                (fpr, fsh)
                for fpr, fsh in failure_frontier
                if not (pr <= fpr + dominance_eps and sh <= fsh + dominance_eps)
            ]
            failure_frontier.append((pr, sh))

        def should_skip(cfg):
            pr, sh = cfg.pruning_rate, cfg.sh_rate
            for spr, ssh in success_frontier:
                if pr <= spr + dominance_eps and sh <= ssh + dominance_eps:
                    return f"dominated by success (pruning_rate={spr:.2f}, sh_rate={ssh:.2f})"
            for fpr, fsh in failure_frontier:
                if pr >= fpr - dominance_eps and sh >= fsh - dominance_eps:
                    return f"dominated by failure (pruning_rate={fpr:.2f}, sh_rate={fsh:.2f})"
            return None
       

        # Evaluate seeds first
        # Since seeds are sorted by mem_score (descending), we can stop at the first OK config
        seed_successful = False
        for pr, sh in self._initial_seed_configs():
            seed_cfg = SearchConfig(pr, sh)
            skip_reason = should_skip(seed_cfg)
            if skip_reason:
                key = self._cfg_key(seed_cfg.pruning_rate, seed_cfg.sh_rate)
                if key not in visited:
                    visited.add(key)
                mem_score = self._mem_saving_score(seed_cfg)
                print(f"\n[Search] [Seed] pruning_rate={pr:.2f}, sh_rate={sh:.2f}, "
                      f"mem_score={mem_score:.3f} [SKIP] {skip_reason}")
                continue
            result = self._evaluate_config_once(seed_cfg, threshold)
            evaluation_count += 1
            key = self._cfg_key(seed_cfg.pruning_rate, seed_cfg.sh_rate)
            visited.add(key)
            status = "OK" if result["meets"] else "FAIL"
            print(f"\n[Search] [Seed] pruning_rate={pr:.2f}, sh_rate={sh:.2f}, "
                  f"mem_score={result['mem_score']:.3f}, ΔPSNR={result['psnr_drop']:.4f} dB [{status}]")
            logs = result.get("logs")
            if logs:
                print(logs, end="")
            if result["meets"]:
                # Since seeds are sorted by mem_score (descending), the first OK config
                # has the highest mem_score among all OK configs, so we can use it directly
                selected_cfg = seed_cfg
                selected_prune_psnr = result["prune_psnr"]
                best_mem = result["mem_score"]
                best_drop = result["psnr_drop"]
                print(f"  [INFO] New best config: mem_score={best_mem:.3f}, ΔPSNR={best_drop:.4f} dB")
                record_success(seed_cfg)
                seed_successful = True
                # Break immediately since we're evaluating in descending mem_score order
                # The first OK config is the best one we can find in seed phase
                break
            else:
                record_failure(seed_cfg)
                if result["psnr_drop"] < best_failed_drop - 1e-6:
                    best_failed_drop = result["psnr_drop"]
                    best_failed_cfg = seed_cfg

        base_cfg_for_grid = selected_cfg if selected_cfg is not None else best_failed_cfg
        self.search_space = self._get_search_space(base_cfg_for_grid)

        # Seed phase done; stage1 time stops here
        self.stage1_time = time.time() - search_start_time
        stage2_start = time.time()

        # Build grid after seeds
        candidates = []
        cfg_lookup = {}
        grid_configs = self.search_space or []
        for idx, cfg in enumerate(grid_configs):
            total_rate = cfg.pruning_rate + cfg.sh_rate
            if total_rate > 1.0 + 1e-6:
                continue
            key = self._cfg_key(cfg.pruning_rate, cfg.sh_rate)
            cfg_lookup[key] = cfg
            mem_score = self._mem_saving_score(cfg)
            candidates.append((idx, cfg, mem_score, total_rate))

        if not candidates:
            print("\n[WARN] No valid search candidates (check constraints).")
            print(f"[Search] Total configurations evaluated: {evaluation_count}")
            return None

        def push_neighbors(cfg):
            grid_cfg = self._nearest_grid_cfg(cfg.pruning_rate, cfg.sh_rate, cfg_lookup)
            if grid_cfg is None:
                return
            base_step_pr = getattr(self.args, 'search_prune_step', 0.01)
            base_step_sh = getattr(self.args, 'search_sh_step', 0.01)
            
            # Two-tier fast mode based on current PSNR drop relative to target
            # First threshold: use 4x step size when drop < 50% of target (very fast)
            # Second threshold: use 2x step size when drop < 70% of target (fast)
            fast_threshold_1 = getattr(self.args, 'fast_threshold_1', 0.6)  # 4x step size
            fast_threshold_2 = getattr(self.args, 'fast_threshold_2', 0.7)  # 2x step size
            
            if selected_cfg is not None:
                drop_ratio = best_drop / self.target_psnr_drop if self.target_psnr_drop > 0 else 1.0
                if drop_ratio < fast_threshold_1:
                    step_multiplier = 4  # Very fast: 4x step size
                    use_fast_mode = True
                elif drop_ratio < fast_threshold_2:
                    step_multiplier = 2  # Fast: 2x step size
                    use_fast_mode = True
                else:
                    step_multiplier = 1  # Normal: 1x step size
                    use_fast_mode = False
            else:
                step_multiplier = 1
                use_fast_mode = False
            
            step_pr = base_step_pr * step_multiplier
            step_sh = base_step_sh * step_multiplier
            
            # In fast mode, only push neighbors in the direction of higher mem_score (positive steps)
            # to avoid exploring backwards
            if use_fast_mode:
                deltas = [(step_pr, 0.0), (0.0, step_sh)]
            else:
                deltas = [(step_pr, 0.0), (-step_pr, 0.0), (0.0, step_sh), (0.0, -step_sh)]
            
            for dp, ds in deltas:
                pr = round(grid_cfg.pruning_rate + dp, 4)
                sh = round(grid_cfg.sh_rate + ds, 4)
                key = self._cfg_key(pr, sh)
                if pr < 0 or sh < 0 or pr + sh > 1.0 + 1e-6:
                    continue
                if key in visited or key in queued:
                    continue
                neighbor_cfg = cfg_lookup.get(key)
                if neighbor_cfg is None:
                    continue
                if selected_cfg is not None and self._mem_saving_score(neighbor_cfg) <= best_mem + 1e-6:
                    continue
                # Pre-check if neighbor would be skipped (dominated) to avoid adding to queue
                if should_skip(neighbor_cfg):
                    continue
                heapq.heappush(neighbor_queue, (-self._mem_saving_score(neighbor_cfg), pr, sh))
                queued.add(key)

        def process_result(result, label="", allow_neighbor_push=True):
            nonlocal selected_cfg, selected_prune_psnr, best_mem, best_drop, best_failed_cfg, best_failed_drop
            if result is None:
                return False
            cfg = result["cfg"]
            key = self._cfg_key(cfg.pruning_rate, cfg.sh_rate)
            if key in visited:
                return False
            visited.add(key)
            mem_score = result["mem_score"]
            psnr_drop = result["psnr_drop"]
            status = "OK" if result["meets"] else "FAIL"
            print(f"\n[Search]{label} pruning_rate={cfg.pruning_rate:.2f}, sh_rate={cfg.sh_rate:.2f}, "
                  f"mem_score={mem_score:.3f}, ΔPSNR={psnr_drop:.4f} dB [{status}]")
            logs = result.get("logs")
            if logs:
                print(logs, end="")
            if result["meets"]:
                record_success(cfg)
                better = False
                if (selected_cfg is None) or (mem_score > best_mem + 1e-6):
                    better = True
                elif abs(mem_score - best_mem) <= 1e-6 and psnr_drop < best_drop - 1e-6:
                    better = True
                if better:
                    selected_cfg = cfg
                    selected_prune_psnr = result["prune_psnr"]
                    best_mem = mem_score
                    best_drop = psnr_drop
                    print(f"  [INFO] New best config: mem_score={mem_score:.3f}, ΔPSNR={psnr_drop:.4f} dB")
                    if allow_neighbor_push:
                        push_neighbors(cfg)
                    return True
            else:
                record_failure(cfg)
                if psnr_drop < best_failed_drop - 1e-6:
                    best_failed_drop = psnr_drop
                    best_failed_cfg = cfg
            return False

        def evaluate_cfg(cfg, label="", allow_neighbor_push=True):
            nonlocal evaluation_count
            skip_reason = should_skip(cfg)
            if skip_reason:
                key = self._cfg_key(cfg.pruning_rate, cfg.sh_rate)
                if key not in visited:
                    visited.add(key)
                mem_score = self._mem_saving_score(cfg)
                print(f"\n[Search]{label} pruning_rate={cfg.pruning_rate:.2f}, sh_rate={cfg.sh_rate:.2f}, "
                      f"mem_score={mem_score:.3f} [SKIP] {skip_reason}")
                return False
            evaluation_count += 1
            return process_result(self._evaluate_config_once(cfg, threshold), label, allow_neighbor_push)

        if base_cfg_for_grid is not None:
            push_neighbors(base_cfg_for_grid)

        # Neighbor exploration with early stopping optimizations
        neighbor_evals = 0
        consecutive_failures = 0
        max_consecutive_failures = getattr(self.args, 'max_consecutive_failures', 3)
        early_stop_threshold = getattr(self.args, 'neighbor_early_stop_threshold', 0.95)  # Stop if PSNR drop >= 95% of target
        max_queue_size = getattr(self.args, 'max_neighbor_queue_size', 50)  # Limit queue size
        min_improvement_threshold = getattr(self.args, 'min_improvement_threshold', 0.005)  # Minimum improvement per step (dB)
        
        # Track recent improvements for early stopping
        recent_improvements = []
        prev_best_drop = best_drop if selected_cfg is not None else None
        
        def cleanup_queue():
            """Remove low mem_score configs from queue when we find a better config"""
            if selected_cfg is None:
                return
            queue_size_before = len(neighbor_queue)
            if queue_size_before == 0:
                return
            
            # Keep only top candidates with mem_score > current best
            cleaned_queue = []
            removed_count = 0
            while neighbor_queue:
                neg_mem, pr, sh = heapq.heappop(neighbor_queue)
                key = self._cfg_key(pr, sh)
                cfg = cfg_lookup.get(key)
                if cfg and self._mem_saving_score(cfg) > best_mem + 1e-6:
                    heapq.heappush(cleaned_queue, (neg_mem, pr, sh))
                else:
                    queued.discard(key)
                    removed_count += 1
            neighbor_queue[:] = cleaned_queue
            
            # Log cleanup info if configs were removed
            if removed_count > 0:
                print(f"  [INFO] Queue cleanup: removed {removed_count} low mem_score configs (queue: {queue_size_before} -> {len(neighbor_queue)})")
        
        while neighbor_queue and neighbor_evals < self.neighbor_budget:
            # Limit queue size to avoid memory issues
            if len(neighbor_queue) > max_queue_size:
                # Keep only top candidates
                temp_queue = []
                for _ in range(min(max_queue_size, len(neighbor_queue))):
                    temp_queue.append(heapq.heappop(neighbor_queue))
                neighbor_queue[:] = temp_queue
                heapq.heapify(neighbor_queue)
            
            _, pr, sh = heapq.heappop(neighbor_queue)
            key = self._cfg_key(pr, sh)
            
            # Pre-check if should skip before getting cfg
            if key in visited:
                queued.discard(key)
                continue
            
            cfg = cfg_lookup.get(key)
            if cfg is None:
                queued.discard(key)
                continue
            
            # Pre-check dominance before evaluation
            if should_skip(cfg):
                visited.add(key)
                queued.discard(key)
                mem_score = self._mem_saving_score(cfg)
                print(f"\n[Search] [Neighbor] pruning_rate={cfg.pruning_rate:.2f}, sh_rate={cfg.sh_rate:.2f}, "
                      f"mem_score={mem_score:.3f} [SKIP] dominated")
                continue
            
            # Early stop: if current best config is already very close to target, stop
            if selected_cfg is not None and best_drop >= self.target_psnr_drop * early_stop_threshold:
                print(f"\n[INFO] Early stop in neighbor search: best PSNR drop ({best_drop:.4f} dB) >= {early_stop_threshold*100:.0f}% of target ({self.target_psnr_drop:.4f} dB)")
                break
            
            was_ok_before = (selected_cfg is not None)
            result_ok = evaluate_cfg(cfg, label=" [Neighbor]")
            neighbor_evals += 1
            
            if result_ok:
                consecutive_failures = 0
                # Track improvement rate
                if prev_best_drop is not None and best_drop > prev_best_drop:
                    improvement = best_drop - prev_best_drop
                    recent_improvements.append(improvement)
                    if len(recent_improvements) > 3:
                        recent_improvements.pop(0)
                    prev_best_drop = best_drop
                    
                    # If recent improvements are very small, we're likely at the boundary
                    if len(recent_improvements) >= 2 and all(d < min_improvement_threshold for d in recent_improvements[-2:]):
                        print(f"\n[INFO] Early stop in neighbor search: slow improvement rate (last improvements: {[f'{d:.4f}' for d in recent_improvements[-2:]]} dB)")
                        break
                    
                    # Clean up queue when we find a better config
                    cleanup_queue()
                elif prev_best_drop is None:
                    prev_best_drop = best_drop
                    # Also cleanup queue when we first find an OK config
                    cleanup_queue()
            else:
                consecutive_failures += 1
                # If we have a good config and hit multiple consecutive failures, 
                # we're likely at the boundary, so stop early
                if was_ok_before and consecutive_failures >= max_consecutive_failures:
                    print(f"\n[INFO] Early stop in neighbor search: {consecutive_failures} consecutive failures, likely at boundary")
                    break

        # fallback: greedy by mem score if no solution yet
        if selected_cfg is None:
            sorted_candidates = sorted(
                candidates,
                key=lambda item: (-item[2], -item[1].pruning_rate, -item[1].sh_rate)
            )
            greedy_used = 0
            for idx, cfg, _, _ in sorted_candidates:
                if self.greedy_budget > 0 and greedy_used >= self.greedy_budget:
                    break
                if evaluate_cfg(cfg, label=" [Greedy]", allow_neighbor_push=False):
                    break
                greedy_used += 1

        if selected_cfg is None:
            print("\nSearch complete.")
            print(f"[Search] Total configurations evaluated: {evaluation_count}")
            if stage2_start is None:
                stage2_start = time.time()
            self.stage2_time = time.time() - stage2_start
            self.stage3_time = 0.0
            return None

        print(f"\n[Search] Total configurations evaluated: {evaluation_count}")
        # Neighbor/refine finished; mark stage2 time
        self.stage2_time = time.time() - stage2_start if stage2_start is not None else 0.0
        print(f"\n[Finalize] Running quantization for pruning_rate={selected_cfg.pruning_rate:.2f}, sh_rate={selected_cfg.sh_rate:.2f}")
        finalize_start = time.time()
        final_psnr, filesize_est, dic = self.spin_once(
            selected_cfg.pruning_rate,
            selected_cfg.sh_rate,
            save_render=getattr(self.args, 'save_render', False),
            full_eval=True
        )
        self.stage3_time = time.time() - finalize_start

        if dic is None or final_psnr is None:
            print("[WARN] Final evaluation failed; no configuration stored.")
            return None

        dic.update({
            "pruning_rate": selected_cfg.pruning_rate,
            "sh_rate": selected_cfg.sh_rate,
            "psnr": final_psnr,
            "psnr_prune": selected_prune_psnr,
            "filesize": filesize_est
        })

        print("\nSearch complete.")
        return dic
