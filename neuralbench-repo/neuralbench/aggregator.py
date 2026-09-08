# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestrate multiple benchmark experiments and collect/plot results."""

from __future__ import annotations

import json
import logging
import typing as tp
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from exca import ConfDict
from pydantic import Field
from tqdm import tqdm

import neuralset as ns

from .plots._constants import AdaptationMode
from .plots.benchmark import plot_all_results
from .plots.tables import print_skip_table

if tp.TYPE_CHECKING:
    from .main import Experiment

LOGGER = logging.getLogger(__name__)


def _infer_eval_mode(experiment: "Experiment") -> str:
    """Classify *experiment* by adaptation strategy, as an :class:`AdaptationMode` tag."""
    # derived rather than an Experiment field, which would change the config hash
    wrapper = experiment.downstream_model_wrapper
    if wrapper is None:
        return AdaptationMode("finetune").tag
    aggregation = "" if wrapper.aggregation is None else str(wrapper.aggregation)
    if wrapper.lora_config is not None:
        return AdaptationMode("lora", aggregation, wrapper.lora_config.r).tag
    if wrapper.layers_to_unfreeze == [""]:
        strategy = (
            "attentive_probe" if wrapper.probe_config == "attention" else "linear_probe"
        )
        return AdaptationMode(strategy, aggregation).tag
    return AdaptationMode("finetune", aggregation).tag


def _default_output_dir() -> str:
    """Resolve the default output directory under the user-configured ``SAVE_DIR``.

    Imported lazily so that ``BenchmarkAggregator`` can be imported before
    ``neuralbench.config_manager`` has been initialised (avoids triggering
    interactive setup at import time).
    """
    from neuralbench.config_manager import get_config

    return str(Path(get_config()["SAVE_DIR"]) / "outputs")


def _experiment_variant(experiment: "Experiment") -> str:
    """Signature of config axes that vary *independently* of ``brain_model_name``.

    ``brain_model_name`` alone does not uniquely identify a configuration: the
    same backbone can be run with different downstream wrappers (``-w``),
    checkpoints (``--checkpoint``), probe layers, or model hyperparameters (a
    grid sweep, or two YAMLs sharing a config class).  Aggregation groups by
    model name and collapses seeds, so without an extra discriminator these
    distinct configs would be silently averaged together.

    The tag is a ``;``-separated set of ``key=value`` components:

    - ``ckpt=<stem>`` -- the downstream checkpoint, when set;
    - ``cfg=<uid>`` -- exca's own UID for the full ``brain_model_config``
      (``ConfDict.from_model(...).to_uid()``), which generally disambiguates
      *any* model-hyperparameter difference without enumerating fields.  Reusing
      exca's UID keeps this consistent with how exca canonicalises and hashes
      configs for caching, rather than maintaining a parallel scheme;
    - the downstream wrapper's non-default fields, rendered readably (the common
      ``-w`` / probe-layer / aggregation knobs).

    ``build_results_df`` surfaces only the components that actually differ
    between a model's variants, so the ``cfg`` UID stays invisible unless a
    model hyperparameter is the thing that varies.
    """
    parts: list[str] = []
    checkpoint = getattr(experiment, "pretrained_weights_fname", None)
    if checkpoint:
        parts.append(f"ckpt={Path(checkpoint).stem}")
    config = getattr(experiment, "brain_model_config", None)
    if config is not None:
        parts.append(f"cfg={ConfDict.from_model(config).to_uid()}")
    wrapper = getattr(experiment, "downstream_model_wrapper", None)
    if wrapper is not None:
        # ``exclude_defaults`` keeps only fields overridden from the wrapper's
        # defaults, so the canonical single-wrapper case stays minimal while any
        # sweep over wrapper knobs (aggregation, probe_layer, freezing, ...)
        # produces a distinct tag.
        overrides = wrapper.model_dump(exclude_defaults=True, exclude_none=True)
        parts.extend(f"{key}={overrides[key]}" for key in sorted(overrides))
    return ";".join(parts)


class BenchmarkAggregator(ns.BaseModel):
    """Orchestrate multiple :class:`Experiment` runs and visualise results.

    Experiments are submitted (possibly via Slurm) with :meth:`prepare`,
    collected with :meth:`_collect_results`, and plotted/tabled via the
    functions in :mod:`neuralbench.plots.benchmark`.
    """

    experiments: list["Experiment"]
    max_workers: int = 256
    collect_max_workers: int = 32
    debug: bool = False
    wandb_paper_summary: bool = False

    output_dir: str = Field(default_factory=_default_output_dir)

    # Each unique loss name picks one headline metric for plots/tables.  This
    # is a per-loss (not per-task) mapping: when two tasks would share a loss
    # but need different headline metrics, give one of them a dedicated loss
    # class (or, as the sleep-onset task does, wrap it in ``MultiLoss``).
    loss_to_metric_mapping: dict[str, str] = {
        "CrossEntropyLoss": "test/bal_acc",
        "BCEWithLogitsLoss": "test/f1_score_macro",
        "MSELoss": "test/pearsonr",
        "L1Loss": "test/mae",  # currently emg/pose only
        "MultiLoss": "test/bmae",  # currently sleep-onset only
        "ClipLoss": "test/full_retrieval/top5_acc_subject-agg",
        "CTCLoss": "test/CER",  # currently emg/typing only
    }

    def prepare(self) -> None:
        n_total = len(self.experiments)
        statuses = [exp.infra.status() for exp in self.experiments]
        n_completed = statuses.count("completed")
        n_running = statuses.count("running")
        n_cached = n_completed + n_running
        is_force = self.experiments[0].infra.mode == "force"
        if n_cached == n_total:
            parts = []
            if n_completed:
                parts.append(f"{n_completed} completed")
            if n_running:
                parts.append(f"{n_running} still running")
            if is_force:
                LOGGER.info(
                    "All %d experiment(s) already cached/running (%s). "
                    "Re-running with --force.",
                    n_total,
                    ", ".join(parts),
                )
            else:
                LOGGER.info(
                    "All %d experiment(s) already cached/running (%s). "
                    "Nothing to launch. Use --force to re-run.",
                    n_total,
                    ", ".join(parts),
                )
                return

        if self.debug:
            for experiment in self.experiments:
                experiment.run()
        else:
            tmp = self.experiments[0].infra.clone_obj()
            with tmp.infra.job_array(max_workers=self.max_workers) as tasks:
                tasks.extend(self.experiments)

    @staticmethod
    def _process_one_experiment(
        experiment: "Experiment", cached_only: bool
    ) -> tuple[dict[str, tp.Any] | None, str, str, str]:
        """Process one experiment, returning ``(result_or_None, task, model, status)``.

        The status distinguishes an experiment that errored out from one that
        simply has not run yet, which otherwise look identical in the table.
        """
        task = experiment.task_name
        model = experiment.brain_model_name
        if cached_only and (status := experiment.infra.status()) != "completed":
            return None, task, model, status
        out = experiment.run()
        out["task_name"] = experiment.task_name
        study = experiment.data.study
        if isinstance(study, ns.Chain):
            out["dataset_name"] = type(study[0]).__name__
        else:
            out["dataset_name"] = type(study).__name__
        out["brain_model_name"] = experiment.brain_model_name
        out["model_variant"] = _experiment_variant(experiment)
        out["loss"] = {"name": type(experiment.loss).__name__}
        out["seed"] = experiment.seed
        out["eval_mode"] = _infer_eval_mode(experiment)
        return out, task, model, "completed"

    def _collect_results(self, cached_only: bool = False) -> list[dict[str, tp.Any]]:
        """Gather experiment results, optionally skipping uncached ones.

        When *cached_only* is ``True`` the work is purely I/O-bound (pickle
        loads), so experiments are processed in parallel using threads.
        """
        if cached_only:
            return self._collect_results_parallel()
        return self._collect_results_sequential(cached_only=False)

    def _tally(
        self, outcomes: tp.Iterable[tuple[dict[str, tp.Any] | None, str, str, str]]
    ) -> list[dict[str, tp.Any]]:
        """Keep the results out of *outcomes*, tabling what was skipped."""
        results: list[dict[str, tp.Any]] = []
        total: Counter[tuple[str, str]] = Counter()
        skipped: Counter[tuple[str, str]] = Counter()
        failed: Counter[tuple[str, str]] = Counter()
        for out, task, model, status in outcomes:
            total[task, model] += 1
            if out is None:
                skipped[task, model] += 1
                if status == "failed":
                    failed[task, model] += 1
            else:
                results.append(out)
        if skipped:
            print_skip_table(total, skipped, failed)
        return results

    def _collect_results_sequential(
        self, cached_only: bool = False
    ) -> list[dict[str, tp.Any]]:
        return self._tally(
            self._process_one_experiment(exp, cached_only) for exp in self.experiments
        )

    def _collect_results_parallel(self) -> list[dict[str, tp.Any]]:
        n_workers = min(self.collect_max_workers, len(self.experiments))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self._process_one_experiment, exp, True)
                for exp in self.experiments
            ]
            return self._tally(
                future.result()
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Collecting cached results",
                )
            )

    def _save_computational_stats(self, results: list[dict[str, tp.Any]]) -> Path:
        """Write per-experiment computational stats to JSON for later analysis."""
        _COMP_KEYS = (
            "training_time_s",
            "peak_gpu_memory_mb",
            "peak_cpu_memory_mb",
            "n_total_params",
            "n_trainable_params",
        )
        stats = []
        for r in results:
            entry: dict[str, tp.Any] = {
                "task_name": r.get("task_name"),
                "dataset_name": r.get("dataset_name"),
                "model": r.get("brain_model_name", "unknown"),
                "seed": r.get("seed"),
            }
            entry.update({k: r.get(k) for k in _COMP_KEYS})
            stats.append(entry)
        out_path = Path(self.output_dir) / "other" / "computational_stats.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        LOGGER.info("Saved computational stats to %s", out_path)
        return out_path

    def collect(self, cached_only: bool = True) -> list[dict[str, tp.Any]]:
        """Gather results, without the plots, tables and stats files :meth:`run` writes."""
        return self._collect_results(cached_only=cached_only)
    def _log_wandb_paper_summary(self, results: list[dict[str, tp.Any]]) -> None:
        """Upload mean/std test metrics across seeds as a quiet W&B table."""

        def _is_number(value: tp.Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        metric_keys = sorted(
            {
                key
                for row in results
                for key, value in row.items()
                if key.startswith("test/") and _is_number(value)
            }
        )
        if not metric_keys:
            return

        group_keys = ["task_name", "dataset_name", "brain_model_name", "model_variant"]
        grouped: dict[tuple[tp.Any, ...], list[dict[str, tp.Any]]] = defaultdict(list)
        for row in results:
            grouped[tuple(row.get(key) for key in group_keys)].append(row)

        import math
        import os
        import statistics

        summary_rows: list[dict[str, tp.Any]] = []
        for group, rows in sorted(grouped.items(), key=lambda item: item[0]):
            seeds = sorted(row.get("seed") for row in rows)
            base = dict(zip(group_keys, group))
            for metric in metric_keys:
                values = [
                    float(row[metric])
                    for row in rows
                    if _is_number(row.get(metric)) and math.isfinite(float(row[metric]))
                ]
                if not values:
                    continue
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                summary_rows.append(
                    {
                        **base,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n_seeds": len(values),
                        "seeds": ",".join(str(seed) for seed in seeds),
                        "paper_value": f"{mean:.2f} +/- {std:.2f}",
                    }
                )
        if not summary_rows:
            return

        wandb_config = next(
            (
                experiment.wandb_config
                for experiment in self.experiments
                if experiment.wandb_config is not None
            ),
            None,
        )
        if wandb_config is None:
            LOGGER.info("W&B is disabled; skipping paper summary upload.")
            return

        old_silent = os.environ.get("WANDB_SILENT")
        os.environ["WANDB_SILENT"] = "true"
        try:
            import wandb

            if not wandb_config.offline:
                wandb.login(host=wandb_config.host)
            columns = [
                *group_keys,
                "metric",
                "mean",
                "std",
                "n_seeds",
                "seeds",
                "paper_value",
            ]
            with wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                group="paper_summary",
                name=self._paper_summary_run_name(results),
                job_type="paper_summary",
                settings=wandb.Settings(silent=True),
            ) as run:
                table = wandb.Table(
                    columns=columns,
                    data=[[row.get(column) for column in columns] for row in summary_rows],
                )
                run.log({"paper/summary": table})
                for row in summary_rows:
                    metric_name = str(row["metric"]).replace("/", "_")
                    run.summary[f"paper/{metric_name}/mean"] = row["mean"]
                    run.summary[f"paper/{metric_name}/std"] = row["std"]
                    run.summary[f"paper/{metric_name}/value"] = row["paper_value"]
                run.summary["paper/seeds"] = summary_rows[0]["seeds"]
        finally:
            if old_silent is None:
                os.environ.pop("WANDB_SILENT", None)
            else:
                os.environ["WANDB_SILENT"] = old_silent

    def _paper_summary_run_name(self, results: list[dict[str, tp.Any]]) -> str:
        tasks = sorted({str(row.get("task_name", "unknown")) for row in results})
        models = sorted({str(row.get("brain_model_name", "unknown")) for row in results})
        return f"paper_summary/{'+'.join(tasks)}/{'+'.join(models)}"

    def run(self, cached_only: bool = False) -> list[dict[str, tp.Any]]:
        results = self._collect_results(cached_only=cached_only)
        if not results:
            if cached_only:
                # --plot-cached only finds canonical (non-debug) runs: --debug
                # uses a reduced config (fewer epochs, smaller batch, subset
                # query) and is therefore cached under a different key.
                LOGGER.info(
                    "No cached results found. Nothing to plot.\n"
                    "  --plot-cached looks for completed canonical runs only; "
                    "--debug runs are cached separately and are not picked up.\n"
                    "  To populate the cache, run the canonical command first "
                    "(drop --debug), e.g.:\n"
                    "      neuralbench <device> <task> -m <model>\n"
                    "  then re-run with --plot-cached:\n"
                    "      neuralbench <device> <task> -m <model> --plot-cached"
                )
            else:
                LOGGER.info("No results found. Nothing to plot.")
            return []
        self._save_computational_stats(results)
        if self.wandb_paper_summary:
            self._log_wandb_paper_summary(results)
        try:
            plot_all_results(results, self.loss_to_metric_mapping, self.output_dir)
        except ValueError as exc:
            if "requires at least 2 models" not in str(exc):
                raise
            LOGGER.info("Skipping global model-comparison plots: %s", exc)
        return results
