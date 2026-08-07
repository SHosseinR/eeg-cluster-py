"""Within-band validity and cross-band stability analysis for optimization results."""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


EPSILON = 1e-10
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_RANDOM_SEED = 42


def _normalized_gaps(values: Sequence[float], healthy: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    healthy = np.asarray(healthy, dtype=float)
    denominator = np.where(np.abs(healthy) > EPSILON, np.abs(healthy), 1.0)
    return np.abs(values - healthy) / denominator


def _rms(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def _bootstrap_median_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(values, size=(int(n_resamples), values.size), replace=True)
    medians = np.median(samples, axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def _paired_rank_biserial(before: Sequence[float], after: Sequence[float]) -> float:
    differences = np.asarray(before, dtype=float) - np.asarray(after, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    denominator = float(np.sum(ranks))
    return float((np.sum(ranks[differences > 0]) - np.sum(ranks[differences < 0])) / denominator)


def _safe_wilcoxon(before: Sequence[float], after: Sequence[float]) -> Tuple[float, float]:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    finite = np.isfinite(before) & np.isfinite(after)
    before = before[finite]
    after = after[finite]
    if before.size == 0 or np.allclose(before, after):
        return 0.0, 1.0
    result = stats.wilcoxon(before, after, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def _stimulation_polarity(amplitude: float) -> str:
    if amplitude < -EPSILON:
        return "suppression"
    if amplitude > EPSILON:
        return "enhancement"
    return "zero"


def _as_float(value, default=np.nan) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _solution_feasibility(solution: Mapping) -> Tuple[bool, float]:
    constraint_values = np.asarray(solution.get("constraint_values", []), dtype=float)
    if constraint_values.size:
        violation = float(np.sum(np.maximum(constraint_values, 0.0)))
        return violation <= EPSILON, violation
    return bool(solution.get("feasible", True)), float(solution.get("constraint_violation", 0.0))


def build_analysis_tables(
    results_by_band: Mapping[str, Mapping[str, Mapping]],
    require_matched_subjects: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert saved per-band result dictionaries into subject and metric tables."""
    if not results_by_band:
        raise ValueError("No per-band optimization results were provided.")

    subject_sets = {band: set(results) for band, results in results_by_band.items()}
    if require_matched_subjects:
        first_band = next(iter(subject_sets))
        reference = subject_sets[first_band]
        mismatches = {
            band: sorted(reference.symmetric_difference(subjects))
            for band, subjects in subject_sets.items()
            if subjects != reference
        }
        if mismatches:
            raise ValueError(
                "Cross-band analysis requires identical subject IDs in every band; "
                f"mismatches: {json.dumps(mismatches)}"
            )

    subject_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    for band, band_results in results_by_band.items():
        for subject_id, result in band_results.items():
            solution = result.get("best_solution")
            initial = result.get("initial_metrics")
            final = result.get("final_metrics")
            if final is None and solution is not None:
                final = solution.get("measure_values")
            measures = list(result.get("optimization_measures", []))
            healthy_map = result.get("healthy_measure_baselines", {})
            if solution is None or initial is None or final is None or not measures:
                continue
            if any(measure not in healthy_map for measure in measures):
                continue

            initial = np.asarray(initial, dtype=float)
            final = np.asarray(final, dtype=float)
            healthy = np.asarray([healthy_map[measure] for measure in measures], dtype=float)
            if initial.size != len(measures) or final.size != len(measures):
                continue
            initial_gaps = _normalized_gaps(initial, healthy)
            final_gaps = _normalized_gaps(final, healthy)
            initial_distance = _rms(initial_gaps)
            final_distance = _rms(final_gaps)
            relative_improvement = (
                (initial_distance - final_distance) / max(initial_distance, EPSILON)
            )

            amplitude = _as_float(solution.get("stimulation_amplitude"), default=0.0)
            total_change = _as_float(
                solution.get("stimulation_total_change"), default=amplitude
            )
            activation_amount = _as_float(
                solution.get("stimulation_activation_amount"),
                default=amplitude,
            )
            log_gain = _as_float(
                solution.get("stimulation_log_gain"),
                default=amplitude,
            )
            feasible, violation = _solution_feasibility(solution)
            node = int(solution.get("node", -1))
            labels = result.get("channel_display_names") or result.get("channel_names") or []
            target = labels[node] if 0 <= node < len(labels) else str(node)
            raw_min = solution.get("raw_activation_ratio_min", np.nan)
            raw_max = solution.get("raw_activation_ratio_max", np.nan)
            subject_rows.append({
                "subject_id": str(subject_id),
                "band": str(band),
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "relative_improvement": float(relative_improvement),
                "responded": bool(relative_improvement > 0),
                "target_node": node,
                "target_label": target,
                "stimulation_amplitude": amplitude,
                "stimulation_total_change": total_change,
                "stimulation_activation_amount": activation_amount,
                "stimulation_log_gain": log_gain,
                "stimulation_model": solution.get(
                    "stimulation_model",
                    result.get("stimulation_model", "state_space"),
                ),
                "stimulation_polarity": _stimulation_polarity(amplitude),
                "stimulation_duration": _as_float(solution.get("stimulation_duration")),
                "leak": _as_float(solution.get("leak")),
                "feasible": feasible,
                "feasibility_recorded": any(
                    key in solution
                    for key in ("constraint_values", "constraint_violation", "feasible")
                ),
                "constraint_violation": violation,
                "raw_activation_ratio_min": _as_float(raw_min),
                "raw_activation_ratio_max": _as_float(raw_max),
                "n_nodes": int(result.get("n_nodes", len(labels))),
            })
            for idx, measure in enumerate(measures):
                metric_rows.append({
                    "subject_id": str(subject_id),
                    "band": str(band),
                    "measure": measure,
                    "healthy_value": float(healthy[idx]),
                    "initial_value": float(initial[idx]),
                    "final_value": float(final[idx]),
                    "initial_gap": float(initial_gaps[idx]),
                    "final_gap": float(final_gaps[idx]),
                    "gap_improvement": float(initial_gaps[idx] - final_gaps[idx]),
                })

    subject_df = pd.DataFrame(subject_rows)
    metric_df = pd.DataFrame(metric_rows)
    if subject_df.empty:
        raise ValueError("No complete subject results were available for band analysis.")
    return subject_df, metric_df


def _normalized_target_entropy(group: pd.DataFrame) -> float:
    counts = group["target_node"].value_counts().to_numpy(dtype=float)
    n_nodes = int(group["n_nodes"].max())
    if counts.size <= 1 or n_nodes <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(n_nodes))


def compute_band_summary(
    subject_df: pd.DataFrame,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    for band_idx, (band, group) in enumerate(subject_df.groupby("band", sort=False)):
        improvements = group["relative_improvement"].to_numpy(dtype=float)
        median, ci_low, ci_high = _bootstrap_median_ci(
            improvements, n_resamples=n_resamples, random_seed=random_seed + band_idx
        )
        successes = int(np.sum(improvements > 0))
        response_low, response_high = proportion_confint(
            successes, len(group), alpha=0.05, method="wilson"
        )
        polarity = group["stimulation_polarity"].value_counts(normalize=True)
        wilcoxon_stat, wilcoxon_p = _safe_wilcoxon(
            group["initial_distance"], group["final_distance"]
        )
        rows.append({
            "band": band,
            "n_subjects": int(len(group)),
            "median_relative_improvement": median,
            "median_improvement_ci_low": ci_low,
            "median_improvement_ci_high": ci_high,
            "improvement_mad": float(stats.median_abs_deviation(improvements, nan_policy="omit")),
            "improvement_iqr": float(np.nanquantile(improvements, 0.75) - np.nanquantile(improvements, 0.25)),
            "response_count": successes,
            "response_rate": float(successes / len(group)),
            "response_rate_ci_low": float(response_low),
            "response_rate_ci_high": float(response_high),
            "target_entropy_normalized": _normalized_target_entropy(group),
            "enhancement_fraction": float(polarity.get("enhancement", 0.0)),
            "suppression_fraction": float(polarity.get("suppression", 0.0)),
            "zero_fraction": float(polarity.get("zero", 0.0)),
            "polarity_consistency": float(polarity.max()) if not polarity.empty else np.nan,
            "within_band_wilcoxon_statistic": wilcoxon_stat,
            "within_band_wilcoxon_pvalue": wilcoxon_p,
            "within_band_rank_biserial": _paired_rank_biserial(
                group["initial_distance"], group["final_distance"]
            ),
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["within_band_wilcoxon_pvalue_holm"] = multipletests(
            summary["within_band_wilcoxon_pvalue"].to_numpy(), method="holm"
        )[1]
    return summary


def compute_metric_tests(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, measure), group in metric_df.groupby(["band", "measure"], sort=False):
        statistic, pvalue = _safe_wilcoxon(group["initial_gap"], group["final_gap"])
        rows.append({
            "band": band,
            "measure": measure,
            "n_subjects": int(len(group)),
            "median_initial_gap": float(group["initial_gap"].median()),
            "median_final_gap": float(group["final_gap"].median()),
            "wilcoxon_statistic": statistic,
            "wilcoxon_pvalue": pvalue,
            "rank_biserial": _paired_rank_biserial(group["initial_gap"], group["final_gap"]),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_pvalue_holm"] = multipletests(
            result["wilcoxon_pvalue"].to_numpy(), method="holm"
        )[1]
    return result


def compute_cross_band_tests(subject_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    pivot = subject_df.pivot(index="subject_id", columns="band", values="relative_improvement")
    pivot = pivot.dropna(axis=0, how="any")
    bands = list(pivot.columns)
    if len(bands) < 2:
        raise ValueError("At least two bands are required for cross-band analysis.")

    friedman_statistic = np.nan
    friedman_pvalue = np.nan
    if len(bands) >= 3:
        arrays = [pivot[band].to_numpy(dtype=float) for band in bands]
        if all(np.allclose(arrays[0], values) for values in arrays[1:]):
            friedman_statistic, friedman_pvalue = 0.0, 1.0
        else:
            friedman = stats.friedmanchisquare(*arrays)
            friedman_statistic = float(friedman.statistic)
            friedman_pvalue = float(friedman.pvalue)

    rows = []
    for band_a, band_b in combinations(bands, 2):
        values_a = pivot[band_a].to_numpy(dtype=float)
        values_b = pivot[band_b].to_numpy(dtype=float)
        statistic, pvalue = _safe_wilcoxon(values_a, values_b)
        rows.append({
            "band_a": band_a,
            "band_b": band_b,
            "n_paired_subjects": int(len(pivot)),
            "median_difference_a_minus_b": float(np.median(values_a - values_b)),
            "wilcoxon_statistic": statistic,
            "pvalue": pvalue,
        })
    pairwise = pd.DataFrame(rows)
    pairwise["pvalue_holm"] = multipletests(pairwise["pvalue"].to_numpy(), method="holm")[1]
    omnibus = {
        "n_paired_subjects": int(len(pivot)),
        "bands": bands,
        "friedman_statistic": friedman_statistic,
        "friedman_pvalue": friedman_pvalue,
    }
    return pairwise, omnibus


def compute_bootstrap_rank_probabilities(
    subject_df: pd.DataFrame,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    pivot = subject_df.pivot(index="subject_id", columns="band", values="relative_improvement").dropna()
    bands = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("No matched subject rows are available for bootstrap ranking.")
    rng = np.random.default_rng(random_seed)
    rank_counts = np.zeros((len(bands), len(bands)), dtype=float)
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        medians = np.median(values[indices], axis=0)
        order = np.argsort(-medians, kind="mergesort")
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and np.isclose(medians[order[end]], medians[order[start]]):
                end += 1
            probability = 1.0 / (end - start)
            for band_index in order[start:end]:
                rank_counts[band_index, start:end] += probability
            start = end
    rows = []
    for band_index, band in enumerate(bands):
        row = {"band": band}
        for rank_index in range(len(bands)):
            row[f"rank_{rank_index + 1}_probability"] = rank_counts[band_index, rank_index] / n_resamples
        rows.append(row)
    return pd.DataFrame(rows)


def choose_conservative_winner(
    band_summary: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
    omnibus: Mapping,
    alpha: float = 0.05,
) -> Tuple[str, str]:
    ranked = band_summary.sort_values("median_improvement_ci_low", ascending=False)
    candidate = str(ranked.iloc[0]["band"])
    if len(ranked) == 1:
        return candidate, "Only one band was analyzed."
    if not np.isfinite(omnibus.get("friedman_pvalue", np.nan)) or omnibus["friedman_pvalue"] >= alpha:
        return "inconclusive", "The paired omnibus test was not significant."

    for other in ranked["band"].astype(str):
        if other == candidate:
            continue
        row = pairwise_tests[
            ((pairwise_tests["band_a"] == candidate) & (pairwise_tests["band_b"] == other))
            | ((pairwise_tests["band_a"] == other) & (pairwise_tests["band_b"] == candidate))
        ]
        if row.empty or float(row.iloc[0]["pvalue_holm"]) >= alpha:
            return "inconclusive", f"{candidate} was not significantly different from every alternative."
        signed_difference = float(row.iloc[0]["median_difference_a_minus_b"])
        if str(row.iloc[0]["band_a"]) != candidate:
            signed_difference *= -1
        if signed_difference <= 0:
            return "inconclusive", f"{candidate} did not outperform every alternative."
    return candidate, f"{candidate} had the strongest conservative improvement bound and beat every alternative."


def _save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_validity_dashboard(
    band: str,
    subject_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    summary_row: Mapping,
    metric_tests: pd.DataFrame,
    output_path: str,
) -> None:
    group = subject_df[subject_df["band"] == band]
    metrics = metric_df[metric_df["band"] == band].copy()
    long_distance = group.melt(
        id_vars=["subject_id"],
        value_vars=["initial_distance", "final_distance"],
        var_name="state",
        value_name="healthy_distance",
    )
    long_distance["state"] = long_distance["state"].map(
        {"initial_distance": "Before", "final_distance": "After"}
    )
    metric_long = metrics.melt(
        id_vars=["subject_id", "measure"],
        value_vars=["initial_gap", "final_gap"],
        var_name="state",
        value_name="normalized_gap",
    )
    metric_long["state"] = metric_long["state"].map({"initial_gap": "Before", "final_gap": "After"})

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for _, row in group.iterrows():
        axes[0, 0].plot([0, 1], [row["initial_distance"], row["final_distance"]], color="0.75", alpha=0.35)
    sns.boxplot(data=long_distance, x="state", y="healthy_distance", ax=axes[0, 0], color="#8ecae6")
    axes[0, 0].set_title("Paired RMS distance to healthy target")

    sns.violinplot(data=group, x="relative_improvement", ax=axes[0, 1], inner="quartile", color="#90be6d")
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Relative improvement (positive is better)")

    sns.boxplot(data=metric_long, x="measure", y="normalized_gap", hue="state", ax=axes[1, 0])
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("Per-metric normalized healthy-target gaps")

    axes[1, 1].axis("off")
    metric_lines = []
    for _, row in metric_tests[metric_tests["band"] == band].iterrows():
        metric_lines.append(
            f"{row['measure']}: p(Holm)={row['wilcoxon_pvalue_holm']:.3g}, "
            f"r={row['rank_biserial']:.2f}"
        )
    text = (
        f"n = {int(summary_row['n_subjects'])}\n"
        f"Median improvement = {summary_row['median_relative_improvement']:.3f}\n"
        f"95% bootstrap CI = [{summary_row['median_improvement_ci_low']:.3f}, "
        f"{summary_row['median_improvement_ci_high']:.3f}]\n"
        f"Response rate = {summary_row['response_rate']:.1%} "
        f"[{summary_row['response_rate_ci_low']:.1%}, {summary_row['response_rate_ci_high']:.1%}]\n"
        f"Aggregate p(Holm) = {summary_row['within_band_wilcoxon_pvalue_holm']:.3g}\n"
        f"Aggregate rank-biserial = {summary_row['within_band_rank_biserial']:.2f}\n\n"
        + "\n".join(metric_lines)
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", family="monospace", fontsize=11)
    axes[1, 1].set_title("Statistical summary")
    fig.suptitle(f"{band.capitalize()} outcome validity", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, output_path)


def _all_solution_audit(band_results: Mapping[str, Mapping]) -> Tuple[int, int, int]:
    total = 0
    recorded = 0
    infeasible = 0
    for result in band_results.values():
        for solution in result.get("all_solutions", []) or []:
            total += 1
            has_metadata = any(
                key in solution
                for key in ("constraint_values", "constraint_violation", "feasible")
            )
            if not has_metadata:
                continue
            recorded += 1
            feasible, _ = _solution_feasibility(solution)
            infeasible += int(not feasible)
    return total, recorded, infeasible


def _stable_histogram_spec(
    values,
    requested_bins: int = 20,
) -> Tuple[int, Optional[Tuple[float, float]], Optional[float]]:
    """Choose finite histogram bins for constant or near-constant parameters.

    Optimizers often return values within a few floating-point ULPs of a hard
    bound. Dividing that tiny representable range into a fixed number of bins
    can produce duplicate bin edges in NumPy. Such values are scientifically
    indistinguishable here, so they are displayed as one bin around their
    median rather than as numerical optimizer noise.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1, None, None
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    center = float(np.median(finite))
    scale = max(1.0, abs(lower), abs(upper))
    if upper - lower <= 1e-9 * scale:
        padding = max(0.05, 0.02 * scale)
        return 1, (center - padding, center + padding), center
    return max(1, int(requested_bins)), None, None


def plot_stimulation_profile_dashboard(
    band: str,
    subject_df: pd.DataFrame,
    band_results: Mapping[str, Mapping],
    summary_row: Mapping,
    output_path: str,
) -> None:
    group = subject_df[subject_df["band"] == band].copy()
    is_static = (
        "stimulation_model" in group
        and group["stimulation_model"].eq("static_adjacency").all()
    )
    is_adjacency_activation = (
        "stimulation_model" in group
        and group["stimulation_model"].eq("adjacency_activation").all()
    )
    is_log_gain = (
        "stimulation_model" in group
        and group["stimulation_model"].eq("adjacency_activation_log_gain").all()
    )
    is_dynamics_free = is_static or is_adjacency_activation or is_log_gain
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if is_static:
        strength_column = "stimulation_total_change"
    elif is_adjacency_activation:
        strength_column = "stimulation_activation_amount"
    elif is_log_gain:
        strength_column = "stimulation_log_gain"
    else:
        strength_column = "stimulation_amplitude"
    histogram_bins, histogram_range, constant_center = _stable_histogram_spec(
        group[strength_column],
        requested_bins=20,
    )
    histogram_kwargs = {"bins": histogram_bins}
    if histogram_range is not None:
        histogram_kwargs["binrange"] = histogram_range
    sns.histplot(
        data=group,
        x=strength_column,
        hue="stimulation_polarity",
        ax=axes[0, 0],
        **histogram_kwargs,
    )
    axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    if constant_center is not None:
        axes[0, 0].text(
            0.98,
            0.94,
            f"All selected values ≈ {constant_center:.4g}",
            transform=axes[0, 0].transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
    axes[0, 0].set_title(
        "Selected signed total adjacency changes"
        if is_static
        else "Selected signed direct activation amounts"
        if is_adjacency_activation
        else "Selected signed log gains"
        if is_log_gain
        else "Selected signed amplitudes"
    )

    if is_dynamics_free:
        scatter = axes[0, 1].scatter(
            group["target_node"],
            group[strength_column],
            c=group["relative_improvement"],
            s=55,
            cmap="coolwarm",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
        axes[0, 1].set_xlabel("Target node index")
        axes[0, 1].set_ylabel(
            "Signed total adjacency change"
            if is_static
            else "Signed direct activation amount"
            if is_adjacency_activation
            else "Signed log gain"
        )
        axes[0, 1].set_title("Dynamics-free optimization variables")
        colorbar_label = "Relative improvement"
    else:
        scatter = axes[0, 1].scatter(
            group["stimulation_duration"], group["stimulation_amplitude"],
            c=group["leak"], s=35 + 90 * np.clip(group["relative_improvement"], 0, 1),
            cmap="viridis", alpha=0.8, edgecolors="white", linewidths=0.4,
        )
        axes[0, 1].set_xlabel("Duration")
        axes[0, 1].set_ylabel("Signed amplitude")
        axes[0, 1].set_title("Protocol parameters (color = leak, size = improvement)")
        colorbar_label = "Leak"
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    fig.colorbar(scatter, ax=axes[0, 1], label=colorbar_label)

    target_counts = pd.crosstab(group["target_label"], group["stimulation_polarity"])
    target_counts = target_counts.loc[target_counts.sum(axis=1).sort_values(ascending=False).index]
    sns.heatmap(target_counts, annot=True, fmt="g", cmap="Blues", ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title("Target selection by stimulation polarity")
    axes[1, 0].set_xlabel("Polarity")
    axes[1, 0].set_ylabel("Target")

    axes[1, 1].axis("off")
    total_solutions, recorded_solutions, infeasible_solutions = _all_solution_audit(band_results)
    audit_rate = infeasible_solutions / recorded_solutions if recorded_solutions else np.nan
    selected_recorded = group[group["feasibility_recorded"]]
    selected_infeasible = int((~selected_recorded["feasible"]).sum())
    if recorded_solutions:
        feasibility_lines = (
            f"Solutions with feasibility metadata = {recorded_solutions:,}/{total_solutions:,}\n"
            f"Infeasible recorded solutions = {infeasible_solutions:,} ({audit_rate:.1%})\n"
            f"Infeasible selected solutions = {selected_infeasible}\n"
        )
        if is_static:
            feasibility_text = (
                feasibility_lines
                + "Activation-ratio bounds = not applicable (static model)"
            )
        else:
            feasibility_text = (
                feasibility_lines
                + f"Selected raw-ratio minimum = "
                f"{group['raw_activation_ratio_min'].min():.3f}\n"
                + f"Selected raw-ratio maximum = "
                f"{group['raw_activation_ratio_max'].max():.3f}"
            )
    else:
        feasibility_text = (
            f"Evaluated solutions present = {total_solutions:,}\n"
            "Feasibility metadata = not recorded by source optimizer\n"
            "Raw activation-ratio bounds = not recorded"
        )
    audit_text = (
        f"Enhancement = {summary_row['enhancement_fraction']:.1%}\n"
        f"Suppression = {summary_row['suppression_fraction']:.1%}\n"
        f"Zero = {summary_row['zero_fraction']:.1%}\n"
        f"Polarity consistency = {summary_row['polarity_consistency']:.3f}\n"
        f"Normalized target entropy = {summary_row['target_entropy_normalized']:.3f}\n\n"
        + feasibility_text
    )
    axes[1, 1].text(0.02, 0.98, audit_text, va="top", family="monospace", fontsize=11)
    axes[1, 1].set_title("Protocol stability and available metadata")
    axes[0, 0].set_xlabel(
        "Signed total adjacency change"
        if is_static
        else "Signed direct activation amount"
        if is_adjacency_activation
        else "Signed log gain"
        if is_log_gain
        else "Signed amplitude"
    )
    fig.suptitle(f"{band.capitalize()} stimulation profile", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_cross_band_dashboard(
    subject_df: pd.DataFrame,
    band_summary: pd.DataFrame,
    output_path: str,
) -> None:
    band_order = list(band_summary.sort_values("median_improvement_ci_low", ascending=False)["band"])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    pivot = subject_df.pivot(index="subject_id", columns="band", values="relative_improvement")[band_order]
    for _, row in pivot.iterrows():
        axes[0, 0].plot(range(len(band_order)), row.to_numpy(), color="0.75", alpha=0.25)
    sns.violinplot(data=subject_df, x="band", y="relative_improvement", order=band_order, inner="quartile", ax=axes[0, 0])
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Paired relative improvement by band")

    ordered = band_summary.set_index("band").loc[band_order]
    medians = ordered["median_relative_improvement"].to_numpy()
    errors = np.vstack([
        medians - ordered["median_improvement_ci_low"].to_numpy(),
        ordered["median_improvement_ci_high"].to_numpy() - medians,
    ])
    axes[0, 1].errorbar(band_order, medians, yerr=errors, fmt="o", capsize=5, color="#277da1")
    axes[0, 1].set_title("Median improvement with 95% bootstrap CI")

    stability = ordered.reset_index().melt(
        id_vars="band", value_vars=["improvement_mad", "improvement_iqr"],
        var_name="metric", value_name="value"
    )
    sns.barplot(data=stability, x="band", y="value", hue="metric", order=band_order, ax=axes[1, 0])
    axes[1, 0].set_title("Between-subject outcome variability (lower is more stable)")

    response = ordered.reset_index().melt(
        id_vars="band", value_vars=["response_rate", "enhancement_fraction", "suppression_fraction"],
        var_name="metric", value_name="fraction"
    )
    sns.barplot(data=response, x="band", y="fraction", hue="metric", order=band_order, ax=axes[1, 1])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Response and selected polarity fractions")
    fig.suptitle("Cross-band efficacy and between-subject stability", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_rank_probabilities(
    rank_probabilities: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
    output_path: str,
) -> None:
    bands = rank_probabilities["band"].tolist()
    rank_columns = [column for column in rank_probabilities if column.startswith("rank_")]
    rank_matrix = rank_probabilities.set_index("band")[rank_columns]
    rank_matrix.columns = [f"Rank {index + 1}" for index in range(len(rank_columns))]
    p_matrix = pd.DataFrame(np.ones((len(bands), len(bands))), index=bands, columns=bands)
    for _, row in pairwise_tests.iterrows():
        p_matrix.loc[row["band_a"], row["band_b"]] = row["pvalue_holm"]
        p_matrix.loc[row["band_b"], row["band_a"]] = row["pvalue_holm"]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(rank_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Bootstrap rank probabilities")
    axes[0].set_xlabel("Rank")
    sns.heatmap(p_matrix, annot=True, fmt=".3g", cmap="magma_r", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Holm-adjusted paired p-values")
    fig.tight_layout()
    _save_figure(fig, output_path)


def append_band_analysis_report(
    report_path: str,
    winner: str,
    winner_reason: str,
    omnibus: Mapping,
    band_summary: pd.DataFrame,
) -> None:
    with open(report_path, "a", encoding="utf-8") as report:
        report.write("\n" + "=" * 80 + "\n")
        report.write("CROSS-BAND EFFICACY AND STABILITY ANALYSIS\n")
        report.write("=" * 80 + "\n")
        report.write(f"Paired subjects: {omnibus['n_paired_subjects']}\n")
        report.write(
            f"Friedman statistic: {omnibus['friedman_statistic']:.6g}; "
            f"p={omnibus['friedman_pvalue']:.6g}\n"
        )
        report.write(f"Conservative winner: {winner}\n")
        report.write(f"Decision: {winner_reason}\n\n")
        for _, row in band_summary.sort_values("median_improvement_ci_low", ascending=False).iterrows():
            report.write(
                f"{row['band']}: median improvement={row['median_relative_improvement']:.6f} "
                f"(95% CI {row['median_improvement_ci_low']:.6f} to "
                f"{row['median_improvement_ci_high']:.6f}), "
                f"MAD={row['improvement_mad']:.6f}, response={row['response_rate']:.1%}, "
                f"target entropy={row['target_entropy_normalized']:.6f}\n"
            )
        report.write("\nInterpretation limitations:\n")
        report.write("- Stability here means between-subject outcome/protocol stability.\n")
        report.write("- One NSGA-II run per subject does not establish stochastic optimizer reproducibility.\n")
        report.write("- Bands optimize different metric sets; comparisons use dimensionless relative improvement.\n")
        report.write("- These results provide within-dataset statistical support, not clinical validation.\n")


def run_band_stability_analysis(
    results_by_band: Mapping[str, Mapping[str, Mapping]],
    output_dir: str,
    figures_dir: str,
    report_path: str | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict:
    """Run the complete analysis, save CSVs/plots, and return the decision summary."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    subject_df, metric_df = build_analysis_tables(results_by_band, require_matched_subjects=True)
    band_summary = compute_band_summary(subject_df, n_resamples, random_seed)
    metric_tests = compute_metric_tests(metric_df)
    pairwise_tests, omnibus = compute_cross_band_tests(subject_df)
    rank_probabilities = compute_bootstrap_rank_probabilities(subject_df, n_resamples, random_seed)
    winner, winner_reason = choose_conservative_winner(band_summary, pairwise_tests, omnibus)

    paths = {
        "subject_table": os.path.join(output_dir, "subject_band_analysis.csv"),
        "metric_table": os.path.join(output_dir, "subject_metric_analysis.csv"),
        "band_summary": os.path.join(output_dir, "band_comparison_summary.csv"),
        "metric_tests": os.path.join(output_dir, "within_band_metric_tests.csv"),
        "pairwise_tests": os.path.join(output_dir, "band_pairwise_tests.csv"),
        "rank_probabilities": os.path.join(output_dir, "band_bootstrap_rank_probabilities.csv"),
        "decision": os.path.join(output_dir, "band_selection_decision.json"),
    }
    subject_df.to_csv(paths["subject_table"], index=False)
    metric_df.to_csv(paths["metric_table"], index=False)
    band_summary.to_csv(paths["band_summary"], index=False)
    metric_tests.to_csv(paths["metric_tests"], index=False)
    pairwise_tests.to_csv(paths["pairwise_tests"], index=False)
    rank_probabilities.to_csv(paths["rank_probabilities"], index=False)

    decision = {"winner": winner, "reason": winner_reason, **omnibus}
    with open(paths["decision"], "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)

    for band, band_results in results_by_band.items():
        summary_row = band_summary[band_summary["band"] == band].iloc[0]
        plot_outcome_validity_dashboard(
            band, subject_df, metric_df, summary_row, metric_tests,
            os.path.join(figures_dir, f"{band}_outcome_validity_dashboard.png"),
        )
        plot_stimulation_profile_dashboard(
            band, subject_df, band_results, summary_row,
            os.path.join(figures_dir, f"{band}_stimulation_profile_dashboard.png"),
        )
    plot_cross_band_dashboard(
        subject_df, band_summary,
        os.path.join(figures_dir, "cross_band_efficacy_stability_dashboard.png"),
    )
    plot_rank_probabilities(
        rank_probabilities, pairwise_tests,
        os.path.join(figures_dir, "cross_band_rank_probabilities.png"),
    )
    if report_path:
        append_band_analysis_report(report_path, winner, winner_reason, omnibus, band_summary)
    return {"decision": decision, "paths": paths}
