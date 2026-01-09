"""
Ablation-friendly search wrapper.

This module reuses the main Searcher logic from ``modules.py`` but exposes
switches to disable or enable components for ablation (two-stage flow,
adaptive step size, dominance pruning, early stopping). It also provides a
helper to run a suite of ablations and print a compact table of results.
"""

import time
import os
import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from modules import Pruner, Quantizer, Searcher as BaseSearcher, SearchConfig


@dataclass(frozen=True)
class AblationConfig:
    name: str
    evaluate_seeds: bool = True
    run_neighbor_stage: bool = True
    use_adaptive_step: bool = True
    use_dominance_pruning: bool = True
    use_early_stop: bool = True

    @property
    def flags(self) -> Dict[str, bool]:
        return {
            "two_stage": self.evaluate_seeds and self.run_neighbor_stage,
            "adaptive_step": self.use_adaptive_step,
            "dominance_pruning": self.use_dominance_pruning,
            "early_stop": self.use_early_stop,
        }


# Predefined ablation settings following the provided table.
DEFAULT_ABLATIONS: List[AblationConfig] = [
    AblationConfig("Fixed step", evaluate_seeds=True, run_neighbor_stage=True, use_adaptive_step=False,
                   use_dominance_pruning=True, use_early_stop=True),
    AblationConfig("w/o Stage II", evaluate_seeds=False, run_neighbor_stage=False, use_adaptive_step=False,
                   use_dominance_pruning=True, use_early_stop=True),
    AblationConfig("w/o dominance", evaluate_seeds=True, run_neighbor_stage=True, use_adaptive_step=True,
                   use_dominance_pruning=False, use_early_stop=True),
    AblationConfig("w/o early stopping", evaluate_seeds=True, run_neighbor_stage=True, use_adaptive_step=True,
                   use_dominance_pruning=True, use_early_stop=False),
    AblationConfig("Stage II only", evaluate_seeds=True, run_neighbor_stage=True, use_adaptive_step=True,
                   use_dominance_pruning=False, use_early_stop=False),
    AblationConfig("full", evaluate_seeds=True, run_neighbor_stage=True, use_adaptive_step=True,
                   use_dominance_pruning=True, use_early_stop=True),
]


class AblationSearcher(BaseSearcher):
    """Searcher with switches for ablation experiments."""

    def __init__(self, *args, ablation_cfg: AblationConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_cfg = ablation_cfg

    def _should_skip_wrapper(self, cfg, success_frontier, failure_frontier, dominance_eps):
        if not self.ablation_cfg.use_dominance_pruning:
            return None
        pr, sh = cfg.pruning_rate, cfg.sh_rate
        for spr, ssh in success_frontier:
            if pr <= spr + dominance_eps and sh <= ssh + dominance_eps:
                return f"dominated by success (pruning_rate={spr:.2f}, sh_rate={ssh:.2f})"
        for fpr, fsh in failure_frontier:
            if pr >= fpr - dominance_eps and sh >= fsh - dominance_eps:
                return f"dominated by failure (pruning_rate={fpr:.2f}, sh_rate={fsh:.2f})"
        return None

    def run_search(self):  # noqa: C901 - keep logic explicit for clarity
        search_start_time = time.time()

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
        best_mem = -float("inf")
        best_drop = float("inf")
        neighbor_queue = []
        queued = set()
        evaluation_count = 0
        best_failed_cfg = None
        best_failed_drop = float("inf")
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
            return self._should_skip_wrapper(cfg, success_frontier, failure_frontier, dominance_eps)

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

        # ----- Single-stage sweep when Stage II is disabled -----
        if not self.ablation_cfg.run_neighbor_stage:
            # Mimic the simple midpoint sweep from modules_flex: fixed step, no neighbor expansion.
            self.search_space = self._get_search_space(None)
            grid_configs = [cfg for cfg in (self.search_space or []) if cfg.pruning_rate + cfg.sh_rate <= 1.0 + 1e-6]
            if not grid_configs:
                print("\n[WARN] No valid search candidates (check constraints).")
                print(f"[Search] Total configurations evaluated: {evaluation_count}")
                return None

            start_index = len(grid_configs) // 2
            direction = 1
            index = start_index
            threshold = self.baseline_psnr - self.target_psnr_drop
            best_cfg = None
            best_prune_psnr = None
            best_psnr = None

            print(f"\n[Single-Stage] Start index {start_index} -> (prune, sh)=({grid_configs[start_index].pruning_rate:.2f}, {grid_configs[start_index].sh_rate:.2f})")

            while 0 <= index < len(grid_configs):
                cfg = grid_configs[index]
                print(f"\n[Search] pruning_rate={cfg.pruning_rate:.2f}, sh_rate={cfg.sh_rate:.2f}")
                evaluation_count += 1
                current_psnr, _, _ = self.spin_once(
                    cfg.pruning_rate, cfg.sh_rate,
                    save_render=getattr(self.args, "save_render", False),
                    full_eval=False
                )
                if index == start_index:
                    direction = 1 if current_psnr >= threshold else -1

                if current_psnr >= threshold:
                    best_cfg = cfg
                    best_prune_psnr = self.last_prune_psnr
                    best_psnr = current_psnr
                    best_drop = self.baseline_psnr - current_psnr
                    if direction == -1:
                        break
                else:
                    if direction == 1:
                        break

                index += direction

            if best_cfg is None:
                print("\nSearch complete.")
                print(f"[Search] Total configurations evaluated: {evaluation_count}")
                return None

            print(f"\n[Finalize] Running quantization for pruning_rate={best_cfg.pruning_rate:.2f}, sh_rate={best_cfg.sh_rate:.2f}")
            final_psnr, filesize_est, dic = self.spin_once(
                best_cfg.pruning_rate,
                best_cfg.sh_rate,
                save_render=getattr(self.args, "save_render", False),
                full_eval=True,
            )

            if dic is None or final_psnr is None:
                print("[WARN] Final evaluation failed; no configuration stored.")
                return None

            dic.update({
                "pruning_rate": best_cfg.pruning_rate,
                "sh_rate": best_cfg.sh_rate,
                "psnr": final_psnr,
                "psnr_prune": best_prune_psnr,
                # Use pre-quantization drop (prune-stage PSNR) for reporting
                "psnr_drop": self.baseline_psnr - (best_prune_psnr if best_prune_psnr is not None else final_psnr),
                "target_psnr_drop": self.target_psnr_drop,
                "filesize": filesize_est,
                "neval": evaluation_count,
                "search_time": time.time() - search_start_time,
                "ablation_name": self.ablation_cfg.name,
            })

            print("\nSearch complete.")
            print(f"[Search] Total configurations evaluated: {evaluation_count}")
            return dic

        # Stage I: seeds
        if self.ablation_cfg.evaluate_seeds:
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
                    selected_cfg = seed_cfg
                    selected_prune_psnr = result["prune_psnr"]
                    best_mem = result["mem_score"]
                    best_drop = result["psnr_drop"]
                    print(f"  [INFO] New best config: mem_score={best_mem:.3f}, ΔPSNR={best_drop:.4f} dB")
                    record_success(seed_cfg)
                    break
                else:
                    record_failure(seed_cfg)
                    if result["psnr_drop"] < best_failed_drop - 1e-6:
                        best_failed_drop = result["psnr_drop"]
                        best_failed_cfg = seed_cfg

        base_cfg_for_grid = selected_cfg if selected_cfg is not None else best_failed_cfg
        self.search_space = self._get_search_space(base_cfg_for_grid)

        # Stage II: grid + neighbors
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

        def push_neighbors(cfg):
            grid_cfg = self._nearest_grid_cfg(cfg.pruning_rate, cfg.sh_rate, cfg_lookup)
            if grid_cfg is None:
                return
            base_step_pr = getattr(self.args, "search_prune_step", 0.01)
            base_step_sh = getattr(self.args, "search_sh_step", 0.01)

            if self.ablation_cfg.use_adaptive_step and selected_cfg is not None:
                drop_ratio = best_drop / self.target_psnr_drop if self.target_psnr_drop > 0 else 1.0
                fast_threshold_1 = getattr(self.args, "fast_threshold_1", 0.6)
                fast_threshold_2 = getattr(self.args, "fast_threshold_2", 0.7)
                if drop_ratio < fast_threshold_1:
                    step_multiplier = 4
                elif drop_ratio < fast_threshold_2:
                    step_multiplier = 2
                else:
                    step_multiplier = 1
            else:
                step_multiplier = 1

            step_pr = base_step_pr * step_multiplier
            step_sh = base_step_sh * step_multiplier

            deltas = [(step_pr, 0.0), (0.0, step_sh)] if (self.ablation_cfg.use_adaptive_step and selected_cfg is not None and step_multiplier > 1) else [
                (step_pr, 0.0), (-step_pr, 0.0), (0.0, step_sh), (0.0, -step_sh)
            ]

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
                if should_skip(neighbor_cfg):
                    continue
                heapq.heappush(neighbor_queue, (-self._mem_saving_score(neighbor_cfg), pr, sh))
                queued.add(key)

        # If Stage II is disabled, finish early.
        if not self.ablation_cfg.run_neighbor_stage:
            if selected_cfg is None:
                print("\n[WARN] No valid config found in seed stage; neighbor stage disabled.")
                print(f"[Search] Total configurations evaluated: {evaluation_count}")
                return None
            print(f"\n[Search] Total configurations evaluated: {evaluation_count}")
            print(f"\n[Finalize] (seed-only) pruning_rate={selected_cfg.pruning_rate:.2f}, sh_rate={selected_cfg.sh_rate:.2f}")
            final_psnr, filesize_est, dic = self.spin_once(
                selected_cfg.pruning_rate,
                selected_cfg.sh_rate,
                save_render=getattr(self.args, "save_render", False),
                full_eval=True,
            )
            if dic is None or final_psnr is None:
                print("[WARN] Final evaluation failed; no configuration stored.")
                return None
            dic.update({
                "pruning_rate": selected_cfg.pruning_rate,
                "sh_rate": selected_cfg.sh_rate,
                "psnr": final_psnr,
                "psnr_prune": selected_prune_psnr,
                # Use pre-quantization drop (prune-stage PSNR) for reporting
                "psnr_drop": self.baseline_psnr - (selected_prune_psnr if selected_prune_psnr is not None else final_psnr),
                "target_psnr_drop": self.target_psnr_drop,
                "filesize": filesize_est,
                "neval": evaluation_count,
                "search_time": time.time() - search_start_time,
                "ablation_name": self.ablation_cfg.name,
            })
            return dic

        if base_cfg_for_grid is not None:
            push_neighbors(base_cfg_for_grid)

        # If Stage I is skipped, warm up the queue with the best grid config.
        if not self.ablation_cfg.evaluate_seeds and candidates and not neighbor_queue:
            best_idx, best_cfg, _, _ = max(candidates, key=lambda item: item[2])
            if evaluate_cfg(best_cfg, label=" [GridSeed]"):
                pass
            if selected_cfg is not None:
                push_neighbors(best_cfg)

        neighbor_evals = 0
        consecutive_failures = 0
        max_consecutive_failures = getattr(self.args, "max_consecutive_failures", 3)
        early_stop_threshold = getattr(self.args, "neighbor_early_stop_threshold", 0.95)
        max_queue_size = getattr(self.args, "max_neighbor_queue_size", 50)
        min_improvement_threshold = getattr(self.args, "min_improvement_threshold", 0.005)
        recent_improvements = []
        prev_best_drop = best_drop if selected_cfg is not None else None

        def cleanup_queue():
            if selected_cfg is None:
                return
            if not neighbor_queue:
                return
            cleaned_queue = []
            while neighbor_queue:
                neg_mem, pr, sh = heapq.heappop(neighbor_queue)
                key = self._cfg_key(pr, sh)
                cfg = cfg_lookup.get(key)
                if cfg and self._mem_saving_score(cfg) > best_mem + 1e-6:
                    heapq.heappush(cleaned_queue, (neg_mem, pr, sh))
                else:
                    queued.discard(key)
            neighbor_queue[:] = cleaned_queue

        while neighbor_queue and neighbor_evals < self.neighbor_budget:
            if len(neighbor_queue) > max_queue_size:
                temp_queue = []
                for _ in range(min(max_queue_size, len(neighbor_queue))):
                    temp_queue.append(heapq.heappop(neighbor_queue))
                neighbor_queue[:] = temp_queue
                heapq.heapify(neighbor_queue)

            _, pr, sh = heapq.heappop(neighbor_queue)
            key = self._cfg_key(pr, sh)
            if key in visited:
                queued.discard(key)
                continue
            cfg = cfg_lookup.get(key)
            if cfg is None:
                queued.discard(key)
                continue
            if should_skip(cfg):
                visited.add(key)
                queued.discard(key)
                mem_score = self._mem_saving_score(cfg)
                print(f"\n[Search] [Neighbor] pruning_rate={cfg.pruning_rate:.2f}, sh_rate={cfg.sh_rate:.2f}, "
                      f"mem_score={mem_score:.3f} [SKIP] dominated")
                continue

            if self.ablation_cfg.use_early_stop and selected_cfg is not None and best_drop >= self.target_psnr_drop * early_stop_threshold:
                print(f"\n[INFO] Early stop in neighbor search: best PSNR drop ({best_drop:.4f} dB) >= {early_stop_threshold*100:.0f}% of target ({self.target_psnr_drop:.4f} dB)")
                break

            was_ok_before = selected_cfg is not None
            result_ok = evaluate_cfg(cfg, label=" [Neighbor]")
            neighbor_evals += 1

            if result_ok:
                consecutive_failures = 0
                if prev_best_drop is not None and best_drop > prev_best_drop:
                    improvement = best_drop - prev_best_drop
                    recent_improvements.append(improvement)
                    if len(recent_improvements) > 3:
                        recent_improvements.pop(0)
                    prev_best_drop = best_drop
                    if self.ablation_cfg.use_early_stop and len(recent_improvements) >= 2 and all(d < min_improvement_threshold for d in recent_improvements[-2:]):
                        print(f"\n[INFO] Early stop in neighbor search: slow improvement rate (last improvements: {[f'{d:.4f}' for d in recent_improvements[-2:]]} dB)")
                        break
                    cleanup_queue()
                elif prev_best_drop is None:
                    prev_best_drop = best_drop
                    cleanup_queue()
            else:
                consecutive_failures += 1
                if self.ablation_cfg.use_early_stop and was_ok_before and consecutive_failures >= max_consecutive_failures:
                    print(f"\n[INFO] Early stop in neighbor search: {consecutive_failures} consecutive failures, likely at boundary")
                    break

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
            return None

        print(f"\n[Search] Total configurations evaluated: {evaluation_count}")
        print(f"\n[Finalize] Running quantization for pruning_rate={selected_cfg.pruning_rate:.2f}, sh_rate={selected_cfg.sh_rate:.2f}")
        final_psnr, filesize_est, dic = self.spin_once(
            selected_cfg.pruning_rate,
            selected_cfg.sh_rate,
            save_render=getattr(self.args, "save_render", False),
            full_eval=True,
        )

        if dic is None or final_psnr is None:
            print("[WARN] Final evaluation failed; no configuration stored.")
            return None

        dic.update({
            "pruning_rate": selected_cfg.pruning_rate,
            "sh_rate": selected_cfg.sh_rate,
            "psnr": final_psnr,
            "psnr_prune": selected_prune_psnr,
            # Use pre-quantization drop (prune-stage PSNR) for reporting
            "psnr_drop": self.baseline_psnr - (selected_prune_psnr if selected_prune_psnr is not None else final_psnr),
            "target_psnr_drop": self.target_psnr_drop,
            "filesize": filesize_est,
            "neval": evaluation_count,
            "search_time": time.time() - search_start_time,
            "ablation_name": self.ablation_cfg.name,
        })

        print("\nSearch complete.")
        return dic


def run_ablation_experiments(args, dataset, gaussians, pipe, scene, imp_score, filesize_input,
                             search_space=None, target_psnr_drop=1.0, save_render=False,
                             settings: Optional[Iterable[AblationConfig]] = None):
    """
    Run a suite of ablation experiments and print a compact summary.

    Parameters mirror those of ``modules.Searcher``; only the searcher class is swapped.
    """

    settings = list(settings or DEFAULT_ABLATIONS)
    results = []
    for cfg in settings:
        print(f"\n================ Ablation: {cfg.name} ================")
        searcher = AblationSearcher(
            args,
            dataset,
            gaussians,
            pipe,
            scene,
            imp_score,
            filesize_input,
            search_space=search_space,
            target_psnr_drop=target_psnr_drop,
            save_render=save_render,
            ablation_cfg=cfg,
        )
        res = searcher.run_search()
        if res is not None:
            results.append(res)

    if not results:
        print("\n[WARN] No ablation runs produced results.")
        return results

    print("\n===== Ablation Summary =====")
    header = ["Setting", "Two Stage", "adaptive step", "Dominance pruning", "Early-stop", "Neval", "Search Time(s)", "diff"]
    print("\t".join(header))
    for cfg in settings:
        match = next((r for r in results if r.get("ablation_name") == cfg.name), None)
        if match is None:
            row = [cfg.name, "-", "-", "-", "-", "-", "-", "-"]
            print("\t".join(map(str, row)))
            continue
        two_stage = "√" if cfg.flags["two_stage"] else "×"
        adaptive = "√" if cfg.use_adaptive_step else "×"
        dominance = "√" if cfg.use_dominance_pruning else "×"
        early = "√" if cfg.use_early_stop else "×"
        neval = match.get("neval", "-")
        stime = match.get("search_time", 0.0)
        psnr_drop = match.get("psnr_drop")
        diff = "-"
        target = match.get("target_psnr_drop")
        if target is not None and psnr_drop is not None:
            # diff = target - psnr_drop (positive means still under target drop)
            diff = f"{target - psnr_drop:+.4f}"
        row = [cfg.name, two_stage, adaptive, dominance, early, neval, f"{stime:.2f}", diff]
        print("\t".join(map(str, row)))

    return results


__all__ = [
    "AblationConfig",
    "DEFAULT_ABLATIONS",
    "AblationSearcher",
    "run_ablation_experiments",
    "Pruner",
    "Quantizer",
]
