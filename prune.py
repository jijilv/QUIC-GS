#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
import random
import copy
import gc
import numpy as np
from collections import defaultdict
from typing import Tuple
from tqdm import tqdm
from fisher_pool_xyz_scaling import pool_fisher_cuda, pool_fisher_python

def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list

def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    for iteration in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
    return gaussian_list, imp_list


def calc_importance_fisher_logdet(gaussians, scene, pipe, background, resolution: int = 1, save_path: str = None):
    """
    Fisher-based importance (6x6 per Gaussian on xyz+scaling), aggregated over training cameras.
    - Accumulates Fisher with pool_fisher_python (fallback to Python implementation)
    - Computes log-determinant via SVD eigenvalues sum(log(sv)) as the pruning score
    - Optionally saves the Fisher tensor to save_path if provided

    Returns:
        fishers_log_dets: (N,) tensor of scores per Gaussian
    """
    with torch.enable_grad():
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        fishers = torch.zeros(N, 6, 6, device=device, dtype=torch.float32)
        for view_idx, view in tqdm(
            enumerate(scene.getTrainCameras()),
            total=len(scene.getTrainCameras()),
            desc="Computing Fisher...",
        ):
            # pool_fisher_python(
            #     view_idx, view, gaussians, pipe, background,
            #     fishers, resolution
            # )
            pool_fisher_cuda(
                view_idx, view, gaussians, pipe, background,
                fishers, resolution
            )

    if save_path is not None:
        torch.save(fishers, save_path)

    # Prune scores: log det ~ sum log singular values
    fishers_sv = torch.linalg.svdvals(fishers)
    fishers_log_dets = torch.log(torch.clamp_min(fishers_sv, 1e-12)).sum(dim=1)
    return fishers_log_dets


def calc_importance_fisher_fast(gaussians, scene, pipe, background):
    """
    Ultra-fast Fisher approximation using gradient magnitudes.
    Much faster than full Fisher matrix computation.
    """
    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    scores_sum = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    total_pixels = 0

    for view in cameras:
        pkg = render(view, gaussians, pipe, background)
        img = pkg["render"]

        loss = img.sum()
        grads_xyz, grads_scaling = torch.autograd.grad(
            loss, (gaussians._xyz, gaussians._scaling),
            retain_graph=False, create_graph=False, allow_unused=True
        )

        n_xyz = grads_xyz.norm(dim=1) if grads_xyz is not None else torch.zeros_like(scores_sum)
        n_scl = grads_scaling.norm(dim=1) if grads_scaling is not None else torch.zeros_like(scores_sum)
        scores_sum += (n_xyz + n_scl)

        total_pixels += img.shape[1] * img.shape[2]

        gc.collect()

    return scores_sum / max(1, total_pixels)

def calc_importance_fisher_fast_weighted(gaussians, scene, pipe, background):
    """
    Fast importance with visibility/opacity weighting using autograd.grad.
    - Does NOT modify or rely on .grad buffers
    - Weights per-Gaussian gradient norms by visibility_filter * opacity
    - Accumulates across training cameras, then normalizes by total pixels
    Returns: (N,) tensor per-Gaussian scores
    """
    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    scores_sum = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    total_pixels = 0

    for view in cameras:
        pkg = render(view, gaussians, pipe, background)
        img = pkg["render"]
        vis = pkg["visibility_filter"].float()
        opacity = gaussians.get_opacity.squeeze(1)

        loss = img.sum()
        grads_xyz, grads_scaling = torch.autograd.grad(
            loss, (gaussians._xyz, gaussians._scaling),
            retain_graph=False, create_graph=False, allow_unused=True
        )

        n_xyz = grads_xyz.norm(dim=1) if grads_xyz is not None else torch.zeros_like(scores_sum)
        n_scl = grads_scaling.norm(dim=1) if grads_scaling is not None else torch.zeros_like(scores_sum)
        weight = vis * opacity
        scores_sum += (n_xyz + n_scl) * weight

        total_pixels += img.shape[1] * img.shape[2]

        gc.collect()

    return scores_sum / max(1, total_pixels)


def calc_importance_fisher_fast_full(
    gaussians, scene, pipe, background,
    xyz_weight: float = 1.0,
    scale_weight: float = 1.0,
    rot_weight: float = 1.0,
    dc_weight: float = 1.0,
    sh_weight: float = 1.0,
    opacity_weight: float = 1.0,
    use_visibility_opacity: bool = False,
    print_stats: bool = False
):
    """
    Optimized Fisher-approx importance calculation.
    Includes gradients for XYZ, scaling, rotation, DC features, and opacity.
    Performance optimizations:
    - Conditional gradient computation based on weights
    - Progress bar for monitoring
    - Reduced memory allocations
    - Optimized garbage collection frequency

    Parameters:
        gaussians: Gaussian model with attributes _xyz, _scaling, _rotation, _features_dc, _opacity
        scene, pipe, background: rendering context
        xyz_weight: weight for position gradient
        scale_weight: weight for scaling gradient
        rot_weight: weight for rotation gradient
        dc_weight: weight for DC feature gradient
        sh_weight: weight for spherical harmonics gradient
        opacity_weight: weight for opacity gradient
        use_visibility_opacity: whether to weight by visibility * opacity
        print_stats: whether to print gradient statistics

    Returns:
        (N,) tensor of per-Gaussian importance scores normalized by total pixels.
    """
    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]
    scores_sum = torch.zeros(N, device=device)
    total_pixels = 0

    # Pre-allocate weight tensor if needed
    if use_visibility_opacity:
        opacity = gaussians.get_opacity.squeeze(1)
        weight = torch.ones(N, device=device)
    else:
        weight = torch.ones(N, device=device)
        opacity = None

    # Determine which gradients to compute based on weights
    need_xyz = xyz_weight > 0
    need_scaling = scale_weight > 0
    need_rotation = rot_weight > 0
    need_dc = dc_weight > 0
    need_sh = sh_weight > 0
    need_opacity = opacity_weight > 0

    # Build list of parameters to compute gradients for
    params_to_grad = []
    param_indices = []
    if need_xyz:
        params_to_grad.append(gaussians._xyz)
        param_indices.append('xyz')
    if need_scaling:
        params_to_grad.append(gaussians._scaling)
        param_indices.append('scaling')
    if need_rotation:
        params_to_grad.append(gaussians._rotation)
        param_indices.append('rotation')
    if need_dc:
        params_to_grad.append(gaussians._features_dc)
        param_indices.append('dc')
    if need_sh:
        params_to_grad.append(gaussians._features_rest)
        param_indices.append('sh')
    if need_opacity:
        params_to_grad.append(gaussians._opacity)
        param_indices.append('opacity')

    # Pre-allocate gradient norm tensors for statistics
    if print_stats:
        grad_norms = {
            'xyz': torch.zeros(N, device=device),
            'scaling': torch.zeros(N, device=device),
            'rotation': torch.zeros(N, device=device),
            'dc': torch.zeros(N, device=device),
            'sh': torch.zeros(N, device=device),
            'opacity': torch.zeros(N, device=device)
        }

    # Process views with progress bar
    num_cameras = len(cameras)
    gc_collect_interval = max(1, num_cameras // 10)  # Collect garbage every 10% of views
    # Ensure parameters track gradients (restore flags afterwards)
    orig_requires_grad = [p.requires_grad for p in params_to_grad]
    for p in params_to_grad:
        if not p.requires_grad:
            p.requires_grad_(True)

    for view_idx, view in enumerate(tqdm(cameras, desc="Computing importance scores")):
        pkg = render(view, gaussians, pipe, background)
        img = pkg["render"]

        # Update visibility weight if needed
        if use_visibility_opacity:
            vis = pkg.get("visibility_filter", None)
            if vis is not None:
                weight.copy_(vis.float() * opacity)
            else:
                weight.fill_(1.0)

        loss = img.sum()

        # Compute gradients only for needed parameters
        if len(params_to_grad) > 0:
            grads = torch.autograd.grad(
                loss,
                params_to_grad,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
        else:
            grads = []

        # Extract and compute norms for each gradient component
        # Accumulate scores directly to avoid creating intermediate tensors
        grad_idx = 0
        score_contrib = torch.zeros_like(scores_sum)

        if need_xyz and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_xyz = grad.norm(dim=1)
                score_contrib.add_(xyz_weight * n_xyz)
                if print_stats:
                    grad_norms['xyz'] += n_xyz
            grad_idx += 1

        if need_scaling and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_scl = grad.norm(dim=1)
                score_contrib.add_(scale_weight * n_scl)
                if print_stats:
                    grad_norms['scaling'] += n_scl
            grad_idx += 1

        if need_rotation and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_rot = grad.norm(dim=1)
                score_contrib.add_(rot_weight * n_rot)
                if print_stats:
                    grad_norms['rotation'] += n_rot
            grad_idx += 1

        if need_dc and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_dc = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(dc_weight * n_dc)
                if print_stats:
                    grad_norms['dc'] += n_dc
            grad_idx += 1

        if need_sh and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_sh = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(sh_weight * n_sh)
                if print_stats:
                    grad_norms['sh'] += n_sh
            grad_idx += 1

        if need_opacity and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_op = grad.norm(dim=1)
                score_contrib.add_(opacity_weight * n_op)
                if print_stats:
                    grad_norms['opacity'] += n_op
            grad_idx += 1

        # Accumulate weighted scores (single multiplication)
        score_contrib.mul_(weight)
        scores_sum.add_(score_contrib)
        total_pixels += img.shape[1] * img.shape[2]

        # Periodically clear cache
        if (view_idx + 1) % gc_collect_interval == 0:
            torch.cuda.empty_cache()

    # Restore original requires_grad flags
    for flag, p in zip(orig_requires_grad, params_to_grad):
        p.requires_grad_(flag)

    # Compute final scores
    final_scores = scores_sum / max(1, total_pixels)
    
    if print_stats:
        # Normalize statistics by total pixels and number of views
        num_views = max(1, num_cameras)
        grad_stats = {}
        for key in grad_norms:
            grad_stats[key] = (grad_norms[key] / (num_views * max(1, total_pixels / num_views))).mean().item()
        
        # Calculate relative importance
        total = sum(grad_stats.values())
        if total > 0:
            relative_importance = {k: v/total * 100 for k, v in grad_stats.items()}
        else:
            relative_importance = {k: 0.0 for k in grad_stats}
        
        print("\nGradient Component Statistics (Mean Values):")
        print("-" * 50)
        print(f"{'Component':<15} {'Raw Value':>12} {'Relative %':>12}")
        print("-" * 50)
        for component, value in grad_stats.items():
            rel_imp = relative_importance[component]
            print(f"{component:<15} {value:>12.6f} {rel_imp:>11.2f}%")
        print("-" * 50)
        
        # Print weighted contributions
        weighted_stats = {
            'xyz': grad_stats['xyz'] * xyz_weight,
            'scaling': grad_stats['scaling'] * scale_weight,
            'rotation': grad_stats['rotation'] * rot_weight,
            'dc': grad_stats['dc'] * dc_weight,
            'sh': grad_stats['sh'] * sh_weight,
            'opacity': grad_stats['opacity'] * opacity_weight
        }
        total_weighted = sum(weighted_stats.values())
        if total_weighted > 0:
            weighted_importance = {k: v/total_weighted * 100 for k, v in weighted_stats.items()}
        else:
            weighted_importance = {k: 0.0 for k in weighted_stats}
        
        print("\nWeighted Gradient Contributions:")
        print("-" * 50)
        print(f"{'Component':<15} {'Weight':>8} {'Contrib %':>12}")
        print("-" * 50)
        weights = {
            'xyz': xyz_weight,
            'scaling': scale_weight,
            'rotation': rot_weight,
            'dc': dc_weight,
            'sh': sh_weight,
            'opacity': opacity_weight
        }
        for component, weight in weights.items():
            contrib = weighted_importance[component]
            print(f"{component:<15} {weight:>8.2f} {contrib:>11.2f}%")
        print("-" * 50)
    
    return final_scores

def calc_importance_fisher_fast_full_v2(
    gaussians, scene, pipe, background,
    xyz_weight: float = 1.0,
    scale_weight: float = 1.0,
    rot_weight: float = 1.0,
    dc_weight: float = 1.0,
    sh_weight: float = 1.0,
    opacity_weight: float = 1.0,
    use_visibility_opacity: bool = False,
    print_stats: bool = False,
):
    """
    Variant of calc_importance_fisher_fast_full that uses count_render to get
    per-view gaussian_count and applies it as a visibility weighting on the
    final scores.
    """
    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]
    scores_sum = torch.zeros(N, device=device)
    total_pixels = 0

    if use_visibility_opacity:
        opacity = gaussians.get_opacity.squeeze(1)
        weight = torch.ones(N, device=device)
    else:
        weight = torch.ones(N, device=device)
        opacity = None

    need_xyz = xyz_weight > 0
    need_scaling = scale_weight > 0
    need_rotation = rot_weight > 0
    need_dc = dc_weight > 0
    need_sh = sh_weight > 0
    need_opacity = opacity_weight > 0

    params_to_grad = []
    if need_xyz:
        params_to_grad.append(gaussians._xyz)
    if need_scaling:
        params_to_grad.append(gaussians._scaling)
    if need_rotation:
        params_to_grad.append(gaussians._rotation)
    if need_dc:
        params_to_grad.append(gaussians._features_dc)
    if need_sh:
        params_to_grad.append(gaussians._features_rest)
    if need_opacity:
        params_to_grad.append(gaussians._opacity)

    if print_stats:
        grad_norms = {
            'xyz': torch.zeros(N, device=device),
            'scaling': torch.zeros(N, device=device),
            'rotation': torch.zeros(N, device=device),
            'dc': torch.zeros(N, device=device),
            'sh': torch.zeros(N, device=device),
            'opacity': torch.zeros(N, device=device),
        }

    num_cameras = len(cameras)
    gc_collect_interval = max(1, num_cameras // 10)
    orig_requires_grad = [p.requires_grad for p in params_to_grad]
    for p in params_to_grad:
        if not p.requires_grad:
            p.requires_grad_(True)

    for view_idx, view in enumerate(tqdm(cameras, desc="Computing importance scores (v2)")):
        # Single pass: use count_render to get differentiable render plus counts
        pkg = count_render(view, gaussians, pipe, background)
        img = pkg["render"]

        if use_visibility_opacity:
            vis = pkg.get("visibility_filter", None)
            if vis is not None:
                weight.copy_(vis.float() * opacity)
            else:
                weight.fill_(1.0)

        counts = pkg.get("gaussians_count", None)
        counts_mask = counts.gt(0).float() if counts is not None else None

        loss = img.sum()

        grads = torch.autograd.grad(
            loss,
            params_to_grad,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if params_to_grad else []

        grad_idx = 0
        score_contrib = torch.zeros_like(scores_sum)

        if need_xyz and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_xyz = grad.norm(dim=1)
                score_contrib.add_(xyz_weight * n_xyz)
                if print_stats:
                    grad_norms['xyz'] += n_xyz
            grad_idx += 1

        if need_scaling and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_scl = grad.norm(dim=1)
                score_contrib.add_(scale_weight * n_scl)
                if print_stats:
                    grad_norms['scaling'] += n_scl
            grad_idx += 1

        if need_rotation and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_rot = grad.norm(dim=1)
                score_contrib.add_(rot_weight * n_rot)
                if print_stats:
                    grad_norms['rotation'] += n_rot
            grad_idx += 1

        if need_dc and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_dc = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(dc_weight * n_dc)
                if print_stats:
                    grad_norms['dc'] += n_dc
            grad_idx += 1

        if need_sh and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_sh = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(sh_weight * n_sh)
                if print_stats:
                    grad_norms['sh'] += n_sh
            grad_idx += 1

        if need_opacity and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_op = grad.norm(dim=1)
                score_contrib.add_(opacity_weight * n_op)
                if print_stats:
                    grad_norms['opacity'] += n_op
            grad_idx += 1

        score_contrib.mul_(weight)
        if counts_mask is not None:
            score_contrib.mul_(counts_mask*10)
        scores_sum.add_(score_contrib)
        total_pixels += img.shape[1] * img.shape[2]

        if (view_idx + 1) % gc_collect_interval == 0:
            torch.cuda.empty_cache()

    for flag, p in zip(orig_requires_grad, params_to_grad):
        p.requires_grad_(flag)

    final_scores = scores_sum / max(1, total_pixels)

    if print_stats:
        num_views = max(1, num_cameras)
        grad_stats = {}
        for key in grad_norms:
            grad_stats[key] = (grad_norms[key] / (num_views * max(1, total_pixels / num_views))).mean().item()

        total = sum(grad_stats.values())
        if total > 0:
            relative_importance = {k: v / total * 100 for k, v in grad_stats.items()}
        else:
            relative_importance = {k: 0.0 for k in grad_stats}

        print("\nGradient Component Statistics (Mean Values):")
        print("-" * 50)
        print(f"{'Component':<15} {'Raw Value':>12} {'Relative %':>12}")
        print("-" * 50)
        for component, value in grad_stats.items():
            rel_imp = relative_importance[component]
            print(f"{component:<15} {value:>12.6f} {rel_imp:>11.2f}%")
        print("-" * 50)

        weighted_stats = {
            'xyz': grad_stats['xyz'] * xyz_weight,
            'scaling': grad_stats['scaling'] * scale_weight,
            'rotation': grad_stats['rotation'] * rot_weight,
            'dc': grad_stats['dc'] * dc_weight,
            'sh': grad_stats['sh'] * sh_weight,
            'opacity': grad_stats['opacity'] * opacity_weight,
        }
        total_weighted = sum(weighted_stats.values())
        if total_weighted > 0:
            weighted_importance = {k: v / total_weighted * 100 for k, v in weighted_stats.items()}
        else:
            weighted_importance = {k: 0.0 for k in weighted_stats}

        print("\nWeighted Gradient Contributions:")
        print("-" * 50)
        print(f"{'Component':<15} {'Weight':>8} {'Contrib %':>12}")
        print("-" * 50)
        weights = {
            'xyz': xyz_weight,
            'scaling': scale_weight,
            'rotation': rot_weight,
            'dc': dc_weight,
            'sh': sh_weight,
            'opacity': opacity_weight,
        }
        for component, weight_val in weights.items():
            contrib = weighted_importance[component]
            print(f"{component:<15} {weight_val:>8.2f} {contrib:>11.2f}%")
        print("-" * 50)

    return final_scores

def evaluate_view_importance_simple(cameras, gaussians, scene, pipe, background, num_sample=20):
    """
    Quickly evaluate view importance by sampling a subset of views.
    Uses image variance and gradient information as importance metrics.
    
    Args:
        cameras: List of camera views
        gaussians: Gaussian model
        scene, pipe, background: rendering context
        num_sample: Number of views to sample for evaluation
    
    Returns:
        view_importance: (len(cameras),) tensor of view importance scores
    """
    device = gaussians.get_xyz.device
    view_importance = torch.zeros(len(cameras), device=device)
    
    # Sample views uniformly for quick evaluation (or evaluate all if num_sample >= len(cameras))
    num_sample = min(num_sample, len(cameras))
    if num_sample >= len(cameras):
        # Evaluate all views
        sample_indices = list(range(len(cameras)))
        print(f"[V3] Evaluating view importance for all {len(cameras)} views...")
    else:
        # Sample subset of views
        sample_indices = torch.linspace(0, len(cameras) - 1, num_sample).long().tolist()
        print(f"[V3] Evaluating view importance on {num_sample}/{len(cameras)} sampled views...")
    
    # Accumulate contributions for overall statistics
    total_var_contrib = 0.0
    total_grad_contrib = 0.0
    total_vis_contrib = 0.0
    total_importance = 0.0
    
    for idx in sample_indices:
        view = cameras[idx]
        with torch.no_grad():
            pkg = render(view, gaussians, pipe, background)
            img = pkg["render"]
            
            # Compute importance metrics
            # 1. Image variance (detail richness)
            image_variance = img.var().item()
            
            # 2. Gradient magnitude (edge information)
            if img.shape[1] > 1 and img.shape[2] > 1:
                img_grad_x = torch.abs(img[:, 1:, :] - img[:, :-1, :])
                img_grad_y = torch.abs(img[:, :, 1:] - img[:, :, :-1])
                gradient_magnitude = (img_grad_x.mean() + img_grad_y.mean()).item()
            else:
                gradient_magnitude = 0.0
            
            # 3. Visibility (how many gaussians contribute)
            vis = pkg.get("visibility_filter", None)
            if vis is not None:
                visibility_score = vis.float().sum().item() / max(1, vis.numel())
            else:
                visibility_score = 1.0
            
            # Combined importance score
            var_contrib = image_variance * 0.05
            grad_contrib = gradient_magnitude * 0.1
            vis_contrib = visibility_score * 1.0
            importance = var_contrib + grad_contrib + vis_contrib
            
            # Accumulate contributions
            total_var_contrib += var_contrib
            total_grad_contrib += grad_contrib
            total_vis_contrib += vis_contrib
            total_importance += importance
            
            view_importance[idx] = importance
        
        # Clear cache
        del pkg, img
        if idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Print overall contribution statistics
    if total_importance > 0:
        var_pct = (total_var_contrib / total_importance) * 100
        grad_pct = (total_grad_contrib / total_importance) * 100
        vis_pct = (total_vis_contrib / total_importance) * 100
        print(f"\n[Overall] Importance contribution breakdown (averaged over {len(sample_indices)} views):")
        print(f"  image_variance contribution: {total_var_contrib:.6f} ({var_pct:.2f}%)")
        print(f"  gradient_magnitude contribution: {total_grad_contrib:.6f} ({grad_pct:.2f}%)")
        print(f"  visibility_score contribution: {total_vis_contrib:.6f} ({vis_pct:.2f}%)")
        print(f"  Total importance: {total_importance:.6f}")
    else:
        print(f"\n[Overall] Importance contribution breakdown (averaged over {len(sample_indices)} views):")
        print(f"  image_variance contribution: {total_var_contrib:.6f}")
        print(f"  gradient_magnitude contribution: {total_grad_contrib:.6f}")
        print(f"  visibility_score contribution: {total_vis_contrib:.6f}")
        print(f"  Total importance: {total_importance:.6f}")
    
    # Interpolate for non-sampled views using nearest neighbor
    if len(sample_indices) < len(cameras):
        sampled_values = view_importance[sample_indices]
        for i in range(len(cameras)):
            if i not in sample_indices:
                # Find nearest sampled view
                dists = [abs(i - sidx) for sidx in sample_indices]
                nearest_idx = dists.index(min(dists))
                view_importance[i] = sampled_values[nearest_idx]
    
    # Normalize
    view_importance = view_importance / max(1e-8, view_importance.sum())
    
    return view_importance



def calc_importance_fisher_fast_full_v3(
    gaussians, scene, pipe, background,
    xyz_weight: float = 1.0,
    scale_weight: float = 1.0,
    rot_weight: float = 1.0,
    dc_weight: float = 1.0,
    sh_weight: float = 1.0,
    opacity_weight: float = 1.0,
    use_visibility_opacity: bool = False,
    print_stats: bool = False,
    topk_views: int = None,
    view_importance_weight: float = 2.0
):
    """
    Fisher-approx importance calculation with top-k view importance weighting.
    
    Key innovation:
    1. Evaluate view importance quickly by sampling (fixed 20 views)
    2. Select top-k most important views
    3. Apply importance weight to top-k views during importance calculation
    
    Parameters:
        gaussians: Gaussian model
        scene, pipe, background: rendering context
        xyz_weight, scale_weight, etc.: weights for different gradient components
        use_visibility_opacity: whether to weight by visibility * opacity
        print_stats: whether to print statistics
        topk_views: number of top important views to weight (if None, use all views with weighting)
        view_importance_weight: weight multiplier for important views (default: 2.0)
    
    Returns:
        (N,) tensor of per-Gaussian importance scores
    """
    all_cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]
    
    # Step 1: Evaluate view importance for all views
    print("[V3] Step 1: Evaluating view importance for all views...")
    torch.cuda.empty_cache()
    view_importance = evaluate_view_importance_simple(all_cameras, gaussians, scene, pipe, background, 
                                                       num_sample=len(all_cameras))
    
    # Step 2: Determine view weights for all views
    # Use all views, but weight top-k important views more
    cameras = all_cameras
    view_weights = torch.ones(len(cameras), device=device)  # Default weight: 1.0
    
    if topk_views is not None and topk_views < len(all_cameras):
        # Select top-k views and apply weight multiplier
        _, topk_indices = torch.topk(view_importance, min(topk_views, len(all_cameras)))
        topk_indices = topk_indices.cpu().tolist()
        # Apply weight multiplier to top-k views
        for idx in topk_indices:
            view_weights[idx] = view_importance_weight
        print(f"[V3] Using all {len(cameras)} views, with top-{len(topk_indices)} important views "
              f"weighted {view_importance_weight}x")
    else:
        # Use all views with importance-based weighting
        # Normalize importance and apply weight multiplier
        max_importance = view_importance.max()
        view_weights = (view_importance / max_importance) * view_importance_weight
        # Ensure minimum weight of 1.0 for all views
        view_weights = torch.clamp(view_weights, min=1.0)
        print(f"[V3] Using all {len(cameras)} views with importance-based weighting "
              f"(weight range: [{view_weights.min():.2f}, {view_weights.max():.2f}]x)")
    
    # Normalize view weights (optional, to maintain similar total weight)
    # view_weights = view_weights / max(1e-8, view_weights.mean())
    
    # Step 3: Compute importance scores with view weighting
    scores_sum = torch.zeros(N, device=device)
    total_pixels = 0
    
    # Pre-allocate weight tensor if needed
    if use_visibility_opacity:
        opacity = gaussians.get_opacity.squeeze(1)
        weight = torch.ones(N, device=device)
    else:
        weight = torch.ones(N, device=device)
        opacity = None
    
    # Determine which gradients to compute
    need_xyz = xyz_weight > 0
    need_scaling = scale_weight > 0
    need_rotation = rot_weight > 0
    need_dc = dc_weight > 0
    need_sh = sh_weight > 0
    need_opacity = opacity_weight > 0
    
    params_to_grad = []
    if need_xyz:
        params_to_grad.append(gaussians._xyz)
    if need_scaling:
        params_to_grad.append(gaussians._scaling)
    if need_rotation:
        params_to_grad.append(gaussians._rotation)
    if need_dc:
        params_to_grad.append(gaussians._features_dc)
    if need_sh:
        params_to_grad.append(gaussians._features_rest)
    if need_opacity:
        params_to_grad.append(gaussians._opacity)
    
    # Pre-allocate gradient norm tensors for statistics
    if print_stats:
        grad_norms = {
            'xyz': torch.zeros(N, device=device),
            'scaling': torch.zeros(N, device=device),
            'rotation': torch.zeros(N, device=device),
            'dc': torch.zeros(N, device=device),
            'sh': torch.zeros(N, device=device),
            'opacity': torch.zeros(N, device=device)
        }
    
    # Ensure parameters track gradients
    orig_requires_grad = [p.requires_grad for p in params_to_grad]
    for p in params_to_grad:
        if not p.requires_grad:
            p.requires_grad_(True)
    
    # Process views with importance weighting
    num_cameras = len(cameras)
    gc_collect_interval = max(1, num_cameras // 10)
    
    for view_idx, (view, view_weight) in enumerate(tqdm(zip(cameras, view_weights), 
                                                          desc="Computing importance (V3)",
                                                          total=len(cameras))):
        pkg = render(view, gaussians, pipe, background)
        img = pkg["render"]
        
        if use_visibility_opacity:
            vis = pkg.get("visibility_filter", None)
            if vis is not None:
                weight.copy_(vis.float() * opacity)
            else:
                weight.fill_(1.0)
        
        loss = img.sum()
        
        if len(params_to_grad) > 0:
            grads = torch.autograd.grad(
                loss,
                params_to_grad,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
        else:
            grads = []
        
        # Extract and compute norms for each gradient component
        grad_idx = 0
        score_contrib = torch.zeros_like(scores_sum)
        
        if need_xyz and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_xyz = grad.norm(dim=1)
                score_contrib.add_(xyz_weight * n_xyz)
                if print_stats:
                    grad_norms['xyz'] += n_xyz * view_weight
            grad_idx += 1
        
        if need_scaling and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_scl = grad.norm(dim=1)
                score_contrib.add_(scale_weight * n_scl)
                if print_stats:
                    grad_norms['scaling'] += n_scl * view_weight
            grad_idx += 1
        
        if need_rotation and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_rot = grad.norm(dim=1)
                score_contrib.add_(rot_weight * n_rot)
                if print_stats:
                    grad_norms['rotation'] += n_rot * view_weight
            grad_idx += 1
        
        if need_dc and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_dc = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(dc_weight * n_dc)
                if print_stats:
                    grad_norms['dc'] += n_dc * view_weight
            grad_idx += 1
        
        if need_sh and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_sh = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(sh_weight * n_sh)
                if print_stats:
                    grad_norms['sh'] += n_sh * view_weight
            grad_idx += 1
        
        if need_opacity and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_op = grad.norm(dim=1)
                score_contrib.add_(opacity_weight * n_op)
                if print_stats:
                    grad_norms['opacity'] += n_op * view_weight
            grad_idx += 1
        
        # Apply visibility weight and view importance weight
        score_contrib.mul_(weight)
        scores_sum.add_(score_contrib * view_weight)
        total_pixels += img.shape[1] * img.shape[2] * view_weight.item()
        
        if (view_idx + 1) % gc_collect_interval == 0:
            torch.cuda.empty_cache()
    
    # Restore original requires_grad flags
    for flag, p in zip(orig_requires_grad, params_to_grad):
        p.requires_grad_(flag)
    
    # Compute final scores
    final_scores = scores_sum / max(1, total_pixels)
    
    if print_stats:
        # Normalize statistics
        num_views = max(1, num_cameras)
        grad_stats = {}
        for key in grad_norms:
            grad_stats[key] = (grad_norms[key] / (num_views * max(1, total_pixels / num_views))).mean().item()
        
        total = sum(grad_stats.values())
        if total > 0:
            relative_importance = {k: v/total * 100 for k, v in grad_stats.items()}
        else:
            relative_importance = {k: 0.0 for k in grad_stats}
        
        print("\n[V3] Gradient Component Statistics (Mean Values):")
        print("-" * 50)
        print(f"{'Component':<15} {'Raw Value':>12} {'Relative %':>12}")
        print("-" * 50)
        for component, value in grad_stats.items():
            rel_imp = relative_importance[component]
            print(f"{component:<15} {value:>12.6f} {rel_imp:>11.2f}%")
        print("-" * 50)
        
        # Print weighted contributions
        weighted_stats = {
            'xyz': grad_stats['xyz'] * xyz_weight,
            'scaling': grad_stats['scaling'] * scale_weight,
            'rotation': grad_stats['rotation'] * rot_weight,
            'dc': grad_stats['dc'] * dc_weight,
            'sh': grad_stats['sh'] * sh_weight,
            'opacity': grad_stats['opacity'] * opacity_weight
        }
        total_weighted = sum(weighted_stats.values())
        if total_weighted > 0:
            weighted_importance = {k: v/total_weighted * 100 for k, v in weighted_stats.items()}
        else:
            weighted_importance = {k: 0.0 for k in weighted_stats}
        
        print("\n[V3] Weighted Gradient Contributions:")
        print("-" * 50)
        print(f"{'Component':<15} {'Weight':>8} {'Contrib %':>12}")
        print("-" * 50)
        weights = {
            'xyz': xyz_weight,
            'scaling': scale_weight,
            'rotation': rot_weight,
            'dc': dc_weight,
            'sh': sh_weight,
            'opacity': opacity_weight
        }
        for component, weight in weights.items():
            contrib = weighted_importance[component]
            print(f"{component:<15} {weight:>8.2f} {contrib:>11.2f}%")
        print("-" * 50)
    
    return final_scores


def calc_importance_fisher_fast_full_v4(
    gaussians, scene, pipe, background,
    xyz_weight: float = 1.0,
    scale_weight: float = 1.0,
    rot_weight: float = 1.0,
    dc_weight: float = 1.0,
    sh_weight: float = 1.0,
    opacity_weight: float = 1.0,
    use_visibility_opacity: bool = False,
    print_stats: bool = False,
    topk_views_percent: float = None,
    view_importance_weight: float = 2.0
):
    """
    Fisher-approx importance calculation with top-percent view importance weighting.
    
    Key innovation:
    1. Evaluate view importance quickly by sampling (fixed 20 views)
    2. Select top-percent most important views (e.g., top 20% of views)
    3. Apply importance weight to top-percent views during importance calculation
    
    Parameters:
        gaussians: Gaussian model
        scene, pipe, background: rendering context
        xyz_weight, scale_weight, etc.: weights for different gradient components
        use_visibility_opacity: whether to weight by visibility * opacity
        print_stats: whether to print statistics
        topk_views_percent: percentage of top important views to weight (0.0-1.0, e.g., 0.2 for top 20%)
                           if None, use all views with weighting
        view_importance_weight: weight multiplier for important views (default: 2.0)
    
    Returns:
        (N,) tensor of per-Gaussian importance scores
    """
    all_cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]
    
    # Step 1: Evaluate view importance for all views
    print("[V4] Step 1: Evaluating view importance for all views...")
    torch.cuda.empty_cache()
    view_importance = evaluate_view_importance_simple(all_cameras, gaussians, scene, pipe, background, 
                                                       num_sample=len(all_cameras))
    
    # Step 2: Determine view weights for all views
    # Use all views, but weight top-percent important views more
    cameras = all_cameras
    view_weights = torch.ones(len(cameras), device=device)  # Default weight: 1.0
    
    if topk_views_percent is not None and 0.0 < topk_views_percent <= 1.0:
        # Calculate number of views to select based on percentage
        num_topk = max(1, int(len(all_cameras) * topk_views_percent))
        num_topk = min(num_topk, len(all_cameras))
        
        # Select top-percent views and apply weight multiplier
        _, topk_indices = torch.topk(view_importance, num_topk)
        topk_indices = topk_indices.cpu().tolist()
        # Apply weight multiplier to top-percent views
        for idx in topk_indices:
            view_weights[idx] = view_importance_weight
        print(f"[V4] Using all {len(cameras)} views, with top-{num_topk} ({topk_views_percent*100:.1f}%) "
              f"important views weighted {view_importance_weight}x")
    else:
        # Use all views with importance-based weighting
        # Normalize importance and apply weight multiplier
        max_importance = view_importance.max()
        view_weights = (view_importance / max_importance) * view_importance_weight
        # Ensure minimum weight of 1.0 for all views
        view_weights = torch.clamp(view_weights, min=1.0)
        print(f"[V4] Using all {len(cameras)} views with importance-based weighting "
              f"(weight range: [{view_weights.min():.2f}, {view_weights.max():.2f}]x)")
    
    # Normalize view weights (optional, to maintain similar total weight)
    # view_weights = view_weights / max(1e-8, view_weights.mean())
    
    # Step 3: Compute importance scores with view weighting
    scores_sum = torch.zeros(N, device=device)
    total_pixels = 0
    
    # Pre-allocate weight tensor if needed
    if use_visibility_opacity:
        opacity = gaussians.get_opacity.squeeze(1)
        weight = torch.ones(N, device=device)
    else:
        weight = torch.ones(N, device=device)
        opacity = None
    
    # Determine which gradients to compute
    need_xyz = xyz_weight > 0
    need_scaling = scale_weight > 0
    need_rotation = rot_weight > 0
    need_dc = dc_weight > 0
    need_sh = sh_weight > 0
    need_opacity = opacity_weight > 0
    
    params_to_grad = []
    if need_xyz:
        params_to_grad.append(gaussians._xyz)
    if need_scaling:
        params_to_grad.append(gaussians._scaling)
    if need_rotation:
        params_to_grad.append(gaussians._rotation)
    if need_dc:
        params_to_grad.append(gaussians._features_dc)
    if need_sh:
        params_to_grad.append(gaussians._features_rest)
    if need_opacity:
        params_to_grad.append(gaussians._opacity)
    
    # Pre-allocate gradient norm tensors for statistics
    if print_stats:
        grad_norms = {
            'xyz': torch.zeros(N, device=device),
            'scaling': torch.zeros(N, device=device),
            'rotation': torch.zeros(N, device=device),
            'dc': torch.zeros(N, device=device),
            'sh': torch.zeros(N, device=device),
            'opacity': torch.zeros(N, device=device)
        }
    
    # Ensure parameters track gradients
    orig_requires_grad = [p.requires_grad for p in params_to_grad]
    for p in params_to_grad:
        if not p.requires_grad:
            p.requires_grad_(True)
    
    # Process views with importance weighting
    num_cameras = len(cameras)
    gc_collect_interval = max(1, num_cameras // 10)
    
    for view_idx, (view, view_weight) in enumerate(tqdm(zip(cameras, view_weights), 
                                                          desc="Computing importance (V4)",
                                                          total=len(cameras))):
        pkg = render(view, gaussians, pipe, background)
        img = pkg["render"]
        
        if use_visibility_opacity:
            vis = pkg.get("visibility_filter", None)
            if vis is not None:
                weight.copy_(vis.float() * opacity)
            else:
                weight.fill_(1.0)
        
        loss = img.sum()
        
        if len(params_to_grad) > 0:
            grads = torch.autograd.grad(
                loss,
                params_to_grad,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
        else:
            grads = []
        
        # Extract and compute norms for each gradient component
        grad_idx = 0
        score_contrib = torch.zeros_like(scores_sum)
        
        if need_xyz and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_xyz = grad.norm(dim=1)
                score_contrib.add_(xyz_weight * n_xyz)
                if print_stats:
                    grad_norms['xyz'] += n_xyz * view_weight
            grad_idx += 1
        
        if need_scaling and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_scl = grad.norm(dim=1)
                score_contrib.add_(scale_weight * n_scl)
                if print_stats:
                    grad_norms['scaling'] += n_scl * view_weight
            grad_idx += 1
        
        if need_rotation and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_rot = grad.norm(dim=1)
                score_contrib.add_(rot_weight * n_rot)
                if print_stats:
                    grad_norms['rotation'] += n_rot * view_weight
            grad_idx += 1
        
        if need_dc and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_dc = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(dc_weight * n_dc)
                if print_stats:
                    grad_norms['dc'] += n_dc * view_weight
            grad_idx += 1
        
        if need_sh and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_sh = grad.flatten(start_dim=1).norm(dim=1)
                score_contrib.add_(sh_weight * n_sh)
                if print_stats:
                    grad_norms['sh'] += n_sh * view_weight
            grad_idx += 1
        
        if need_opacity and grad_idx < len(grads):
            grad = grads[grad_idx]
            if grad is not None:
                n_op = grad.norm(dim=1)
                score_contrib.add_(opacity_weight * n_op)
                if print_stats:
                    grad_norms['opacity'] += n_op * view_weight
            grad_idx += 1
        
        # Apply visibility weight and view importance weight
        score_contrib.mul_(weight)
        scores_sum.add_(score_contrib * view_weight)
        total_pixels += img.shape[1] * img.shape[2] * view_weight.item()
        
        if (view_idx + 1) % gc_collect_interval == 0:
            torch.cuda.empty_cache()
    
    # Restore original requires_grad flags
    for flag, p in zip(orig_requires_grad, params_to_grad):
        p.requires_grad_(flag)
    
    # Compute final scores
    final_scores = scores_sum / max(1, total_pixels)
    
    if print_stats:
        # Normalize statistics
        num_views = max(1, num_cameras)
        grad_stats = {}
        for key in grad_norms:
            grad_stats[key] = (grad_norms[key] / (num_views * max(1, total_pixels / num_views))).mean().item()
        
        total = sum(grad_stats.values())
        if total > 0:
            relative_importance = {k: v/total * 100 for k, v in grad_stats.items()}
        else:
            relative_importance = {k: 0.0 for k in grad_stats}
        
        print("\n[V4] Gradient Component Statistics (Mean Values):")
        print("-" * 50)
        print(f"{'Component':<15} {'Raw Value':>12} {'Relative %':>12}")
        print("-" * 50)
        for component, value in grad_stats.items():
            rel_imp = relative_importance[component]
            print(f"{component:<15} {value:>12.6f} {rel_imp:>11.2f}%")
        print("-" * 50)
        
        # Print weighted contributions
        weighted_stats = {
            'xyz': grad_stats['xyz'] * xyz_weight,
            'scaling': grad_stats['scaling'] * scale_weight,
            'rotation': grad_stats['rotation'] * rot_weight,
            'dc': grad_stats['dc'] * dc_weight,
            'sh': grad_stats['sh'] * sh_weight,
            'opacity': grad_stats['opacity'] * opacity_weight
        }
        total_weighted = sum(weighted_stats.values())
        if total_weighted > 0:
            weighted_importance = {k: v/total_weighted * 100 for k, v in weighted_stats.items()}
        else:
            weighted_importance = {k: 0.0 for k in weighted_stats}
        
        print("\n[V4] Weighted Gradient Contributions:")
        print("-" * 50)
        print(f"{'Component':<15} {'Weight':>8} {'Contrib %':>12}")
        print("-" * 50)
        weights = {
            'xyz': xyz_weight,
            'scaling': scale_weight,
            'rotation': rot_weight,
            'dc': dc_weight,
            'sh': sh_weight,
            'opacity': opacity_weight
        }
        for component, weight in weights.items():
            contrib = weighted_importance[component]
            print(f"{component:<15} {weight:>8.2f} {contrib:>11.2f}%")
        print("-" * 50)
    
    return final_scores


def calc_importance_hessian_zero_shot(
    gaussians, scene, pipe, background,
    xyz_weight: float = 1.0,
    scale_weight: float = 1.0,
    rot_weight: float = 1.0,
    dc_weight: float = 1.0,
    sh_weight: float = 1.0,
    opacity_weight: float = 1.0,
    epsilon: float = 1e-3,
    use_visibility_opacity: bool = False,
    print_stats: bool = False
):
    """
    Zero-shot Hessian-aware Importance Estimation
    ----------------------------------------------
    通过有限差分(second-order finite difference)近似 Hessian 对角项，
    不使用反向传播、不计算真实梯度，实现训练-free的重要性估计。

    重要性分数:
        H_i = (L(x_i+eps) - 2*L(x_i) + L(x_i-eps)) / eps^2

    优点:
        - Zero-shot（无需训练）
        - Hessian-aware（二阶敏感度，更精确）
        - 不依赖 autograd.grad
        - 对每个 Gaussian 独立估计 importance

    返回:
        (N,) tensor，每个 Gaussian 的二阶重要性分数
    """

    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]

    if N == 0:
        return torch.zeros(0, device=device)

    scores_sum = torch.zeros(N, device=device)
    total_pixels = 0
    eps_sq = epsilon ** 2

    # Visibility weighting buffer
    if use_visibility_opacity:
        opacity = gaussians.get_opacity.squeeze(1)
        weight_buf = torch.ones(N, device=device)
    else:
        opacity = None
        weight_buf = torch.ones(N, device=device)

    components = []

    def register_axis_component(name, tensor, weight):
        if weight <= 0 or tensor is None or tensor.numel() == 0:
            return
        flat = tensor.view(N, -1)
        axis_norm = tensor.detach().view(N, -1).abs() + 1e-8
        axis_mean = axis_norm.mean(dim=0, keepdim=True).clamp_min(1e-8)
        axis_norm = axis_norm / axis_mean
        components.append(
            {
                "name": name,
                "weight": weight,
                "flat": flat,
                "norm": axis_norm,
                "axis_count": axis_norm.shape[1],
                "mode": "axis",
            }
        )

    def register_block_component(name, tensor, weight):
        if weight <= 0 or tensor is None or tensor.numel() == 0:
            return
        flat = tensor.view(N, -1)
        block = tensor.detach().view(N, -1)
        if block.shape[1] == 0:
            return
        block_norm = block.norm(dim=1) + 1e-8
        block_mean = block_norm.mean().clamp_min(1e-8)
        components.append(
            {
                "name": name,
                "weight": weight,
                "flat": flat,
                "norm": block_norm / block_mean,
                "mode": "block",
            }
        )

    register_axis_component("xyz", gaussians._xyz, xyz_weight)
    register_axis_component("scaling", gaussians._scaling, scale_weight)
    register_axis_component("rotation", gaussians._rotation, rot_weight)
    register_axis_component("dc", gaussians._features_dc, dc_weight)
    register_block_component("sh", gaussians._features_rest, sh_weight)
    register_axis_component("opacity", gaussians._opacity, opacity_weight)

    if len(components) == 0:
        return scores_sum

    component_stats = (
        {comp["name"]: {"raw": 0.0, "weighted": 0.0} for comp in components}
        if print_stats
        else None
    )

    num_cameras = len(cameras)
    gc_collect_interval = max(1, num_cameras // 10)

    for view_idx, view in enumerate(tqdm(cameras, desc="Zero-shot Hessian Importance")):
        with torch.no_grad():
            pkg0 = render(view, gaussians, pipe, background)
            img0 = pkg0["render"]
            base_loss = img0.sum()
        total_pixels += img0.shape[1] * img0.shape[2]

        if use_visibility_opacity:
            vis = pkg0.get("visibility_filter", None)
            if vis is not None:
                weight_buf.copy_(vis.float() * opacity)
            else:
                weight_buf.fill_(1.0)
        else:
            weight_buf.fill_(1.0)

        view_scores = torch.zeros_like(scores_sum)

        for comp in components:
            if comp["mode"] == "axis":
                axis_norm = comp["norm"]
                flat = comp["flat"]
                for axis_idx in range(comp["axis_count"]):
                    column = flat[:, axis_idx]
                    with torch.no_grad():
                        column.add_(epsilon)
                        loss_pos = render(view, gaussians, pipe, background)["render"].sum()
                        column.add_(-2 * epsilon)
                        loss_neg = render(view, gaussians, pipe, background)["render"].sum()
                        column.add_(epsilon)
                    h_val = (loss_pos - 2 * base_loss + loss_neg) / eps_sq
                    abs_h = torch.abs(h_val)
                    view_scores.add_(comp["weight"] * abs_h * axis_norm[:, axis_idx])
                    if component_stats is not None:
                        component_stats[comp["name"]]["raw"] += abs_h.item()
                        component_stats[comp["name"]]["weighted"] += comp["weight"] * abs_h.item()
            else:
                flat = comp["flat"]
                with torch.no_grad():
                    flat.add_(epsilon)
                    loss_pos = render(view, gaussians, pipe, background)["render"].sum()
                    flat.add_(-2 * epsilon)
                    loss_neg = render(view, gaussians, pipe, background)["render"].sum()
                    flat.add_(epsilon)
                h_val = (loss_pos - 2 * base_loss + loss_neg) / eps_sq
                abs_h = torch.abs(h_val)
                view_scores.add_(comp["weight"] * abs_h * comp["norm"])
                if component_stats is not None:
                    component_stats[comp["name"]]["raw"] += abs_h.item()
                    component_stats[comp["name"]]["weighted"] += comp["weight"] * abs_h.item()

        scores_sum.add_(view_scores * weight_buf)

        if (view_idx + 1) % gc_collect_interval == 0:
            torch.cuda.empty_cache()

    final_scores = scores_sum / max(1, total_pixels)

    if component_stats is not None:
        num_views = max(1, num_cameras)
        avg_raw = {k: v["raw"] / num_views for k, v in component_stats.items()}
        avg_weighted = {k: v["weighted"] / num_views for k, v in component_stats.items()}
        total_raw = sum(avg_raw.values())
        total_weighted = sum(avg_weighted.values())

        print("\nHessian Component Statistics (per-view averages):")
        print("-" * 60)
        print(f"{'Component':<15} {'Raw':>12} {'Raw %':>10} {'Weighted %':>14}")
        print("-" * 60)
        for name in component_stats.keys():
            raw_val = avg_raw[name]
            raw_pct = (raw_val / total_raw * 100.0) if total_raw > 0 else 0.0
            weighted_pct = (
                avg_weighted[name] / total_weighted * 100.0 if total_weighted > 0 else 0.0
            )
            print(f"{name:<15} {raw_val:>12.6f} {raw_pct:>9.2f}% {weighted_pct:>13.2f}%")
        print("-" * 60)

    return final_scores


def calc_importance2(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    color_importance, dc_importance, sh_importance, cov_grad, opacity_contribution = None, None, None, None, None
    
    # Register hooks to capture gradients
    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    
    # Clear existing gradients
    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None

    # Accumulate raw gradient sums, divide once at the end
    total_pixels = 0
    viewpoint_cam = viewpoint_stack.pop()
    rendering = render(viewpoint_cam, gaussians, pipe, background)["render"]
    loss = rendering.sum()
    loss.backward()
    total_pixels += rendering.shape[1] * rendering.shape[2]
    
    # Build per-view contributions or zero tensors with correct flattened shapes
    if gaussians._features_dc.grad is not None and gaussians._features_rest.grad is not None:
        color_sum = torch.cat(
            [gaussians._features_dc.grad, gaussians._features_rest.grad],
            1,
        ).flatten(-2)
        dc_sum = torch.cat([gaussians._features_dc.grad], 1).flatten(-2)
        sh_sum = torch.cat([gaussians._features_rest.grad], 1).flatten(-2)
    else:
        n = gaussians._features_dc.shape[0]
        color_dim = gaussians._features_dc.flatten(-2).shape[1] + gaussians._features_rest.flatten(-2).shape[1]
        dc_dim = gaussians._features_dc.flatten(-2).shape[1]
        sh_dim = gaussians._features_rest.flatten(-2).shape[1]
        color_sum = torch.zeros(n, color_dim, device=gaussians._features_dc.device, dtype=gaussians._features_dc.dtype)
        dc_sum = torch.zeros(n, dc_dim, device=gaussians._features_dc.device, dtype=gaussians._features_dc.dtype)
        sh_sum = torch.zeros(n, sh_dim, device=gaussians._features_rest.device, dtype=gaussians._features_rest.dtype)
    

    for iteration in range(len(viewpoint_stack)):
        # Zero grads so that each view contributes its own gradient (no double counting)
        gaussians._features_dc.grad = None
        gaussians._features_rest.grad = None
        viewpoint_cam = viewpoint_stack.pop()
        rendering = render(viewpoint_cam, gaussians, pipe, background)["render"]
        loss = rendering.sum()
        loss.backward()
        total_pixels += rendering.shape[1] * rendering.shape[2]
        
        # Check if gradients exist before using them
        if gaussians._features_dc.grad is not None and gaussians._features_rest.grad is not None:
            color_sum += torch.cat(
                [gaussians._features_dc.grad, gaussians._features_rest.grad],
                1,
            ).flatten(-2)
            dc_sum += torch.cat([gaussians._features_dc.grad], 1).flatten(-2)
            sh_sum += torch.cat([gaussians._features_rest.grad], 1).flatten(-2)
        
        gc.collect()
    
    # Clean up hooks and cache
    h1.remove()
    h2.remove()
    torch.cuda.empty_cache()

    # Normalize by total pixels and then reduce
    color_importance = color_sum / max(1, total_pixels)
    dc_importance = dc_sum / max(1, total_pixels)
    sh_importance = sh_sum / max(1, total_pixels)


    color_contribution = color_importance.amax(-1)
    dc_contribution = dc_importance.amax(-1)
    sh_contribution = sh_importance.amax(-1)
    
    return color_contribution.detach(), dc_contribution.detach(), sh_contribution.detach()

def calc_importance_nfip(gaussians, scene, pipe, background,
                         rot_weight=1.0, dc_weight=1.0, opacity_weight=1.0,
                         batch_size=32, eps=1e-8):
    """
    Normalized Fisher Information Proxy (NFIP)
    理论基础更强，可解释性更好，计算复杂度 ~ O(N*V)，与原版相同。

    数学含义：
        score_i ≈ ||∇_θ_i L||^2 / (L^2 + ε)
    表示每个高斯在单位误差下的敏感度贡献。

    参数含义同上。
    """
    cameras = scene.getTrainCameras().copy()
    device = gaussians.get_xyz.device
    scores_sum = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    total_weight = torch.zeros_like(scores_sum)

    for i in range(0, len(cameras), batch_size):
        batch_cams = cameras[i:i+batch_size]
        torch.cuda.empty_cache(); gc.collect()

        for view in batch_cams:
            pkg = render(view, gaussians, pipe, background)
            img = pkg["render"]
            loss = (img ** 2).sum()  # 相当于平方误差的能量项
            loss_energy = loss.detach()

            grads = torch.autograd.grad(
                loss,
                (gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._features_dc, gaussians._opacity),
                retain_graph=False, create_graph=False, allow_unused=True
            )

            grads_xyz, grads_scaling, grads_rot, grads_dc, grads_op = grads
            # n_xyz = grads_xyz.norm(dim=1) ** 2 if grads_xyz is not None else torch.zeros_like(scores_sum)
            # n_scl = grads_scaling.norm(dim=1) ** 2 if grads_scaling is not None else torch.zeros_like(scores_sum)
            # n_rot = grads_rot.norm(dim=1) ** 2 if grads_rot is not None else torch.zeros_like(scores_sum)
            # n_dc = grads_dc.flatten(start_dim=1).norm(dim=1) ** 2 if grads_dc is not None else torch.zeros_like(scores_sum)
            # n_op = grads_op.norm(dim=1) ** 2 if grads_op is not None else torch.zeros_like(scores_sum)

            n_xyz = grads_xyz.norm(dim=1)  if grads_xyz is not None else torch.zeros_like(scores_sum)
            n_scl = grads_scaling.norm(dim=1)  if grads_scaling is not None else torch.zeros_like(scores_sum)
            n_rot = grads_rot.norm(dim=1)  if grads_rot is not None else torch.zeros_like(scores_sum)
            n_dc = grads_dc.flatten(start_dim=1).norm(dim=1)  if grads_dc is not None else torch.zeros_like(scores_sum)
            n_op = grads_op.norm(dim=1)  if grads_op is not None else torch.zeros_like(scores_sum)

            score = n_xyz + n_scl + rot_weight * n_rot + dc_weight * n_dc + opacity_weight * n_op

            # Fisher 归一化：梯度平方 / 损失平方
            norm_factor = (loss_energy.item() ** 2 + eps)
            scores_sum += score / norm_factor
            # scores_sum += score
            total_weight += 1

            del grads, grads_xyz, grads_scaling, grads_rot, grads_dc, grads_op
            torch.cuda.empty_cache()

    return scores_sum / (total_weight + eps)


def pre_volume(volume, beta):
    # volume = torch.tensor(volume)
    index = int(volume.shape[0] * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, beta)
    return v_list

def cal_mesongs_imp(
        gaussians,
        views,
        pipeline, 
        background
    ):
    
    full_opa_imp = None 

    # Use count_render to get per-Gaussian importance scores, similar to prune_list
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = count_render(view, gaussians, pipeline, background)
        important_score = render_pkg["important_score"].detach()
        
        if full_opa_imp is not None:
            full_opa_imp += important_score
        else:
            full_opa_imp = important_score
            
        del render_pkg
        
    volume = torch.prod(gaussians.get_scaling, dim=1)

    v_list = pre_volume(volume, 0.01)
    imp = v_list * full_opa_imp
    
    return imp.detach()