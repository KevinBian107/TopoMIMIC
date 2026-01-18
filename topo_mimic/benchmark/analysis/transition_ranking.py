"""Transition variant ranking and statistics."""

import os
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .behavior_parsing import (
    TurnDirection,
    turn_direction_to_string,
    format_transition_with_turns,
)
from .ground_truth import TransitionSegment, find_all_transitions


@dataclass
class TransitionVariant:
    from_behavior: str
    to_behavior: str
    from_turn: TurnDirection
    to_turn: TurnDirection
    count: int

    @property
    def base_transition(self) -> tuple[str, str]:
        return tuple(sorted([self.from_behavior, self.to_behavior]))

    @property
    def formatted_name(self) -> str:
        return format_transition_with_turns(
            self.from_behavior,
            self.to_behavior,
            self.from_turn,
            self.to_turn,
        )

    @property
    def canonical_key(self) -> tuple:
        base = self.base_transition
        if base[0] == self.from_behavior:
            return (base[0], base[1], self.from_turn, self.to_turn)
        else:
            return (base[0], base[1], self.to_turn, self.from_turn)


def count_transition_variants(
    transitions: list[TransitionSegment],
) -> list[TransitionVariant]:
    """
    Count transition variants including turn directions.

    Returns:
        List of TransitionVariant sorted by count (descending)
    """
    counter = Counter()

    for t in transitions:
        base = tuple(sorted([t.from_behavior, t.to_behavior]))
        if base[0] == t.from_behavior:
            key = (base[0], base[1], t.from_turn, t.to_turn)
        else:
            key = (base[0], base[1], t.to_turn, t.from_turn)
        counter[key] += 1

    variants = []
    for key, count in counter.most_common():
        variants.append(TransitionVariant(
            from_behavior=key[0],
            to_behavior=key[1],
            from_turn=key[2],
            to_turn=key[3],
            count=count,
        ))

    return variants


def count_base_transitions(
    transitions: list[TransitionSegment],
) -> Counter:
    """Count base transitions (ignoring turn directions)."""
    counter = Counter()
    for t in transitions:
        key = tuple(sorted([t.from_behavior, t.to_behavior]))
        counter[key] += 1
    return counter


def print_transition_ranking(
    variants: list[TransitionVariant],
    top_k: int = 50,
    show_base_totals: bool = True,
) -> None:
    """Print transition variant ranking to console."""
    print("\n" + "=" * 70)
    print("TRANSITION VARIANT RANKING (with turn directions)")
    print("=" * 70)
    print(f"{'Rank':<6}{'From':<25}{'To':<25}{'Count':>8}")
    print("-" * 70)

    for i, v in enumerate(variants[:top_k]):
        from_str = f"{turn_direction_to_string(v.from_turn)}-{v.from_behavior}"
        to_str = f"{turn_direction_to_string(v.to_turn)}-{v.to_behavior}"
        print(f"{i+1:<6}{from_str:<25}{to_str:<25}{v.count:>8}")

    total = sum(v.count for v in variants)
    print("-" * 70)
    print(f"Total transitions: {total}")
    print(f"Unique variants: {len(variants)}")

    if show_base_totals:
        print("\n" + "=" * 70)
        print("BASE TRANSITIONS (ignoring turn directions)")
        print("=" * 70)

        base_counter = Counter()
        for v in variants:
            base_counter[v.base_transition] += v.count

        print(f"{'Rank':<6}{'Transition':<45}{'Count':>8}")
        print("-" * 70)

        for i, (key, count) in enumerate(base_counter.most_common()):
            transition_str = f"{key[0]} <-> {key[1]}"
            print(f"{i+1:<6}{transition_str:<45}{count:>8}")


def save_transition_ranking(
    variants: list[TransitionVariant],
    output_path: str,
) -> pd.DataFrame:
    """Save transition variant ranking to CSV."""
    data = []
    for i, v in enumerate(variants):
        data.append({
            "Rank": i + 1,
            "From_Behavior": v.from_behavior,
            "To_Behavior": v.to_behavior,
            "From_Turn": str(v.from_turn),
            "To_Turn": str(v.to_turn),
            "Count": v.count,
            "Formatted": v.formatted_name,
            "Base_Transition": f"{v.base_transition[0]} <-> {v.base_transition[1]}",
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved transition ranking to {output_path}")
    return df


def save_base_transition_counts(
    variants: list[TransitionVariant],
    output_path: str,
) -> pd.DataFrame:
    """Save base transition counts (ignoring turns) to CSV."""
    base_counter = Counter()
    for v in variants:
        base_counter[v.base_transition] += v.count

    data = []
    for i, (key, count) in enumerate(base_counter.most_common()):
        data.append({
            "Rank": i + 1,
            "Behavior_1": key[0],
            "Behavior_2": key[1],
            "Transition": f"{key[0]} <-> {key[1]}",
            "Count": count,
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved base transition counts to {output_path}")
    return df


def generate_transition_matrix(
    variants: list[TransitionVariant],
    include_turns: bool = False,
) -> pd.DataFrame:
    """
    Generate a transition matrix.

    Args:
        variants: List of TransitionVariant
        include_turns: If True, expand behaviors by turn direction

    Returns:
        DataFrame with from-behavior as rows, to-behavior as columns
    """
    if include_turns:
        behaviors = set()
        for v in variants:
            behaviors.add(f"{turn_direction_to_string(v.from_turn)}-{v.from_behavior}")
            behaviors.add(f"{turn_direction_to_string(v.to_turn)}-{v.to_behavior}")
        behaviors = sorted(behaviors)

        matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
        for v in variants:
            from_key = f"{turn_direction_to_string(v.from_turn)}-{v.from_behavior}"
            to_key = f"{turn_direction_to_string(v.to_turn)}-{v.to_behavior}"
            matrix.loc[from_key, to_key] += v.count
            matrix.loc[to_key, from_key] += v.count
    else:
        behaviors = set()
        for v in variants:
            behaviors.add(v.from_behavior)
            behaviors.add(v.to_behavior)
        behaviors = sorted(behaviors)

        matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
        for v in variants:
            matrix.loc[v.from_behavior, v.to_behavior] += v.count
            matrix.loc[v.to_behavior, v.from_behavior] += v.count

    return matrix


def run_transition_ranking_analysis(
    csv_path: str,
    output_dir: str,
    short_threshold: int = 10,
    max_gap: int = 10,
    merge_rear: bool = True,
    top_k: int = 50,
) -> tuple[list[TransitionVariant], pd.DataFrame]:
    """
    Run complete transition ranking analysis.

    Args:
        csv_path: Path to behavior_labels.csv
        output_dir: Directory to save output files
        short_threshold: Max frames for intermediate behavior bridging
        max_gap: Max gap between consecutive segments
        merge_rear: Merge Supported/Unsupported Rear
        top_k: Number of top variants to display

    Returns:
        Tuple of (variants, ranking_dataframe)
    """
    os.makedirs(output_dir, exist_ok=True)

    transitions = find_all_transitions(
        csv_path,
        short_threshold=short_threshold,
        max_gap=max_gap,
        merge_rear=merge_rear,
    )

    print(f"\nFound {len(transitions)} total transitions in ground truth data")

    variants = count_transition_variants(transitions)

    print_transition_ranking(variants, top_k=top_k)

    df_ranking = save_transition_ranking(
        variants,
        os.path.join(output_dir, "transition_variants_ranking.csv"),
    )

    save_base_transition_counts(
        variants,
        os.path.join(output_dir, "base_transition_counts.csv"),
    )

    matrix = generate_transition_matrix(variants, include_turns=False)
    matrix.to_csv(os.path.join(output_dir, "transition_matrix.csv"))
    print(f"Saved transition matrix to {os.path.join(output_dir, 'transition_matrix.csv')}")

    matrix_turns = generate_transition_matrix(variants, include_turns=True)
    matrix_turns.to_csv(os.path.join(output_dir, "transition_matrix_with_turns.csv"))
    print(f"Saved transition matrix (with turns) to {os.path.join(output_dir, 'transition_matrix_with_turns.csv')}")

    return variants, df_ranking
