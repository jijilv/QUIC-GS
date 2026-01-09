#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:05:28 2024

@author: bytian
"""

import os
from re import T
import sys
import torch
import time
import numpy as np
import shutil
import copy
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from prune import cal_mesongs_imp, calc_importance2, calc_importance_fisher_fast_full, calc_importance_fisher_fast_full_v2, calc_importance_fisher_fast_full_v3, calc_importance_fisher_fast_full_v4, calc_importance_fisher_logdet, calc_importance_hessian_zero_shot, calculate_v_imp_score, prune_list
# from modules import Pruner, Quantizer, Searcher
from modules import Pruner, Quantizer
from modules_ablate import AblationSearcher, AblationConfig, DEFAULT_ABLATIONS
from modules_ablate import Pruner, Quantizer, AblationSearcher as Searcher

def timed_function(func):
    """Decorator to time a function, can be toggled on/off via self.enable_timing."""
    def wrapper(self, *args, **kwargs):
        if self.enable_timing:
            torch.cuda.synchronize()
            start_time = time.time()
        result = func(self, *args, **kwargs)
        if self.enable_timing:
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = f"\033[2;32m{end_time - start_time:.4f} seconds\033[0m"  # Darker green
            print(f"\033[2;32m[TIMING]\033[0m {func.__name__}: {elapsed_time}")  # Darker green for entire line
        return result
    return wrapper

class Pipeline:
    def __init__(self, args, model_params, pipeline_params):
        self.args = args
        self.dataset = model_params.extract(args)
        self.pipe = pipeline_params.extract(args)
        self.gaussians = None
        self.scene = None
        self.enable_timing = False  # Set to False to disable timing
        self.imp_score = None
        self.filesize_input = 0
        self.filesize_output = 0
        self.closest_dic = None
        self.baseline_psnr = 0

    @timed_function
    def load_model_cameras(self):
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.filesize_input = os.path.getsize(
            os.path.join(self.args.model_path, 'point_cloud/iteration_30000/point_cloud.ply'))

    @timed_function
    def compute_importance_scores(self):
        if hasattr(self.args, "imp_score_path") and self.args.imp_score_path and os.path.exists(self.args.imp_score_path):
            print(f"[INFO] Loading importance scores from: {self.args.imp_score_path}")
            data = np.load(self.args.imp_score_path)
            # Automatically get the first (and only) array
            if len(data.files) != 1:
                raise ValueError(f"Expected one array in {self.args.imp_score_path}, but found: {data.files}")
            self.imp_score = data[data.files[0]]

            print (self.imp_score.shape)

        else:
            print("[INFO] Computing importance scores...")
            v_pow=0.1
            bg_color = torch.tensor(
                [1, 1, 1] if self.dataset.white_background else [0, 0, 0],
                dtype=torch.float32,
                device="cuda"
            )
            # _, imp_list = prune_list(self.gaussians, self.scene, self.pipe, bg_color)
            # v_list = calculate_v_imp_score(self.gaussians, imp_list, v_pow)
            # self.imp_score = v_list.cpu().detach().numpy()

            # _, imp_score_tensor, _ = calc_importance2(self.gaussians, self.scene, self.pipe, bg_color)
            # self.imp_score = imp_score_tensor.cpu().detach().numpy()

            # Use MesonGS-style importance (per-view opacity-based)
            # train_views = self.scene.getTrainCameras().copy()
            # imp_tensor = cal_mesongs_imp(self.gaussians, train_views, self.pipe, bg_color)
            # self.imp_score = imp_tensor.cpu().detach().numpy()
            
            # self.imp_score = calc_importance_fisher_logdet(self.gaussians, self.scene, self.pipe, bg_color).cpu().detach().numpy()

            # Choose importance calculation method
            use_v4 = getattr(self.args, 'use_importance_v4', False)
            
            if use_v4:
                scores = calc_importance_fisher_fast_full_v4(
                    self.gaussians,
                    self.scene,
                    self.pipe,
                    bg_color,
                    xyz_weight=getattr(self.args, 'xyz_weight', 0.02),
                    scale_weight=getattr(self.args, 'scale_weight', 1.0),
                    rot_weight=getattr(self.args, 'rot_weight', 0.2),
                    dc_weight=getattr(self.args, 'dc_weight', 0.1),
                    sh_weight=getattr(self.args, 'sh_weight', 0.2),
                    opacity_weight=getattr(self.args, 'opacity_weight', 0.1),
                    use_visibility_opacity=False,
                    print_stats=True,
                    topk_views_percent=getattr(self.args, 'topk_views_percent', None),
                    view_importance_weight=getattr(self.args, 'view_importance_weight', 2.0)
                )
            else:
                scores = calc_importance_fisher_fast_full(
                    self.gaussians,
                    self.scene,
                    self.pipe,
                    bg_color,
                    xyz_weight=0.02,
                    scale_weight=1.0,
                    rot_weight=0.2,
                    dc_weight=0.1,
                    sh_weight=0.2,
                    opacity_weight=0.1,
                    use_visibility_opacity=False,
                    print_stats=True
                )
            # scores = calc_importance_fisher_fast_full_v2(
            #     self.gaussians,
            #     self.scene,
            #     self.pipe,
            #     bg_color,
            #     xyz_weight=0.02,
            #     scale_weight=1.0,
            #     rot_weight=0.2,
            #     dc_weight=0.1,
            #     sh_weight=0.2,
            #     opacity_weight=0.1,
            #     use_visibility_opacity=False,
            #     print_stats=True
            # )
            self.imp_score = scores.cpu().detach().numpy()


    @timed_function
    def store_model(self):
        if not self.closest_dic:
            print("[INFO] No valid configuration found within PSNR threshold.")
            return

        pruning_rate = self.closest_dic['pruning_rate']
        sh_rate = self.closest_dic['sh_rate']
        best_psnr = self.closest_dic['psnr']
        best_size = self.closest_dic['filesize'] / (1024 ** 2)  # in MB
        base_psnr = self.baseline_psnr
        input_mb = self.filesize_input / (1024 ** 2)
        compression_ratio = self.filesize_input / self.closest_dic['filesize']

        print()
        print("=" * 60)
        print(f"Best combination found: pruning={pruning_rate:.3f}, sh={sh_rate:.3f}")
        print()
        print(f"[INFO] Best config:           pruning_rate = {pruning_rate:.3f}, sh_rate = {sh_rate:.3f}")
        print(f"[INFO] Base PSNR:             {base_psnr:.4f} dB")
        print(f"[INFO] Best PSNR:             {best_psnr:.4f} dB")
        print(f"[INFO] PSNR drop:             {base_psnr - best_psnr:.4f} dB")
        print(f"[INFO] Input file size:       {input_mb:.2f} MB")
        print(f"[INFO] Output file size:      {best_size:.2f} MB")
        print(f"[INFO] Compression ratio:     {compression_ratio:.2f}x")
        print("=" * 60)

        # Extra ablation metrics
        neval = self.closest_dic.get("neval", "-")
        stime = self.closest_dic.get("search_time", "-")
        psnr_drop = self.closest_dic.get("psnr_drop", base_psnr - best_psnr if best_psnr else None)
        target_drop = self.closest_dic.get("target_psnr_drop", getattr(self.args, "quality_target_diff", None))
        diff = "-"
        if psnr_drop is not None and target_drop is not None:
            # diff = target_drop - psnr_drop (positive means margin left before target)
            diff = target_drop - psnr_drop
            diff = f"{diff:+.4f}"
        print(f"[INFO] Neval:                 {neval}")
        print(f"[INFO] Search Time(s):        {stime}")
        print(f"[INFO] diff (target - drop):  {diff}")

        # Save compressed model
        npz_path = self.closest_dic['npz_path']
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **self.closest_dic)

        if self.args.save_render:
            print(f" Saving render results ...")
            self.optimizer.spin_once(self.closest_dic['pruning_rate'], self.closest_dic['sh_rate'], save_render=True)

    @timed_function
    def run(self):
        t0 = time.time()

        self.load_model_cameras()

        t1 = time.time()

        self.compute_importance_scores()

        t2 = time.time()

        if args.ablation_setting:
            ab_cfg_map = {cfg.name: cfg for cfg in DEFAULT_ABLATIONS}
            ab_cfg = ab_cfg_map[args.ablation_setting]
            print(f"[INFO] Using ablation setting: {ab_cfg.name} "
                  f"(two_stage={ab_cfg.flags['two_stage']}, adaptive_step={ab_cfg.use_adaptive_step}, "
                  f"dominance_pruning={ab_cfg.use_dominance_pruning}, early_stop={ab_cfg.use_early_stop})")
            self.optimizer = AblationSearcher(
                args=args,
                dataset=self.dataset,
                gaussians=self.gaussians,
                pipe=self.pipe,
                scene=self.scene,
                imp_score=self.imp_score,
                filesize_input=self.filesize_input,
                search_space='default',
                target_psnr_drop=args.quality_target_diff,
                ablation_cfg=ab_cfg
            )
        else:
            self.optimizer = Searcher(
                args=args,
                dataset=self.dataset,
                gaussians=self.gaussians,
                pipe=self.pipe,
                scene=self.scene,
                imp_score=self.imp_score,
                filesize_input=self.filesize_input,
                search_space='default',
                target_psnr_drop=args.quality_target_diff
            )
        
        # Check if fixed configuration is provided
        if args.fixed_pruning_rate is not None and args.fixed_sh_rate is not None:
            print(f"[INFO] Using fixed configuration: pruning_rate={args.fixed_pruning_rate:.3f}, sh_rate={args.fixed_sh_rate:.3f}")
            print(f"[INFO] Skipping search and using fixed configuration directly.")
            
            # Initialize baseline PSNR first
            src_path = args.model_path
            scene_name = os.path.basename(os.path.normpath(src_path))
            self.optimizer.tgt_path = os.path.join(args.output_path, scene_name)
            os.makedirs(self.optimizer.tgt_path, exist_ok=True)
            
            print(f"\n[Baseline] ...")
            self.baseline_psnr = self.optimizer.render_sets_pipeline(
                self.gaussians, skip_train=True, skip_test=False, load_iteration=30000
            )
            # Also set optimizer's baseline_psnr so spin_once can use it
            self.optimizer.baseline_psnr = self.baseline_psnr
            print(f"  Baseline PSNR:          {self.baseline_psnr:.4f} dB")
            
            # Use fixed configuration directly
            final_psnr, filesize_est, dic = self.optimizer.spin_once(
                args.fixed_pruning_rate,
                args.fixed_sh_rate,
                save_render=getattr(args, 'save_render', False),
                full_eval=True
            )
            
            if dic is None or final_psnr is None:
                print("[ERROR] Failed to process fixed configuration.")
                self.closest_dic = None
            else:
                dic.update({
                    "pruning_rate": args.fixed_pruning_rate,
                    "sh_rate": args.fixed_sh_rate,
                    "psnr": final_psnr,
                    "psnr_prune": self.optimizer.last_prune_psnr,
                    "filesize": filesize_est
                })
                self.closest_dic = dic
        else:
            # Use search as before
            self.closest_dic = self.optimizer.run_search()
            self.baseline_psnr = self.optimizer.baseline_psnr

        t3 = time.time()

        self.store_model()

        t4 = time.time()

        print("\n" + "=" * 60)
        print("[STATS] Pipeline Time Breakdown:")
        print(f"  Load time:    {t1 - t0:>6.2f} s")
        print(f"  Score time:   {t2 - t1:>6.2f} s")
        print(f"  Search time:  {t3 - t2:>6.2f} s")
        print(f"  Store time:   {t4 - t3:>6.2f} s")
        print(f"  Total time:   {t4 - t0:>6.2f} s")
        print("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--segments", default=None, type=int,
                        help="Number of segments for per-segment quantization; omit/0/negative to disable segmentation (use one segment for all points)")
    parser.add_argument("--save_render", default=False, type=bool)
    parser.add_argument("--quality_target_diff", default=1.0, type=float)
    parser.add_argument("--imp_score_path", type=str)
    parser.add_argument("--use_entropy_coding", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Enable entropy coding; ignored when compression_method is not set.")
    parser.add_argument("--compression_method", default=None, type=str,
                        choices=['zlib', 'lzma', 'bz2', 'zstd', 'zopfli', 'xz', 'brotli', 'lz4'])
    parser.add_argument("--compression_level", default=None, type=int)
    parser.add_argument("--use_delta_encoding", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--xyz_codec", default='raw32', type=str,
                        choices=['raw32', 'raw16', 'raw8', 'morton16', 'morton8', 'morton32'],
                        help="raw32: store xyz as float32; raw16: store xyz as float16; raw8: 8-bit uniform quantization; morton16: 16-bit Morton-ordered quantization; morton8: 8-bit Morton-ordered quantization; morton32: 32-bit Morton-ordered quantization (default raw32)")
    parser.add_argument("--use_progressive_pruning", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Enable progressive pruning: prune in multiple stages with importance recomputation")
    parser.add_argument("--progressive_pruning_stages", default=3, type=int,
                        help="Number of stages for progressive pruning (default: 3)")
    parser.add_argument("--use_importance_v4", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Use calc_importance_fisher_fast_full_v4 with top-percent view importance weighting")
    parser.add_argument("--topk_views_percent", default=None, type=float,
                        help="Percentage of top important views to weight (0.0-1.0, e.g., 0.2 for top 20%%, None = use all views with weighting, default: None)")
    parser.add_argument("--view_importance_weight", default=2.0, type=float,
                        help="Weight multiplier for important views (default: 2.0)")
    parser.add_argument("--xyz_weight", default=0.02, type=float,
                        help="Weight for xyz gradient component (default: 0.02)")
    parser.add_argument("--scale_weight", default=1.0, type=float,
                        help="Weight for scale gradient component (default: 1.0)")
    parser.add_argument("--rot_weight", default=0.2, type=float,
                        help="Weight for rotation gradient component (default: 0.2)")
    parser.add_argument("--dc_weight", default=0.1, type=float,
                        help="Weight for DC (diffuse color) gradient component (default: 0.1)")
    parser.add_argument("--sh_weight", default=0.2, type=float,
                        help="Weight for SH (spherical harmonics) gradient component (default: 0.2)")
    parser.add_argument("--opacity_weight", default=0.1, type=float,
                        help="Weight for opacity gradient component (default: 0.1)")
    parser.add_argument("--fixed_pruning_rate", default=None, type=float,
                        help="Use fixed pruning rate instead of searching (e.g., 0.5)")
    parser.add_argument("--fixed_sh_rate", default=None, type=float,
                        help="Use fixed sh_rate instead of searching (e.g., 0.5)")
    parser.add_argument("--ablation_setting", default=None, type=str,
                        choices=[cfg.name for cfg in DEFAULT_ABLATIONS],
                        help="Use ablation searcher with a specific setting (see modules_ablate.DEFAULT_ABLATIONS)")

    args = get_combined_args(parser)
    args.data_device = "cuda"
    print("[ARGS] Device:", args.data_device)
    print("[ARGS] Input Model Path:", args.model_path)
    print("[ARGS] Save Render:", args.save_render)

    # Ensure new optional args exist even if absent in cfg_args
    for attr, default in [
        ("fixed_pruning_rate", None),
        ("fixed_sh_rate", None),
        ("ablation_setting", None),
        ("segments", None),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Run pipeline
    Pipeline(args, model_params, pipeline_params).run()
