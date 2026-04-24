"""Gradio-based GUI for manual label assignment with recommendations."""

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio_ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

from topo_mimic.labeling.core import LabelRecommendation


class GridVideoConfig:
    """Configuration for grid video layout matching render_grid_clusters_side_by_side defaults."""
    target_size: tuple[int, int] = (320, 240)
    gutter: int = 10
    pad_x: int = 24
    pad_y: int = 24
    label_h: int = 24
    title_h: int = 34

    @classmethod
    def cell_height(cls) -> int:
        return cls.label_h + cls.target_size[1]

    @classmethod
    def row_height_with_gutter(cls) -> int:
        return cls.cell_height() + cls.gutter

    @classmethod
    def content_start_y(cls) -> int:
        return cls.pad_y + cls.title_h

    @classmethod
    def get_row_bounds(cls, row_idx: int) -> tuple[int, int]:
        """Get (y_start, y_end) pixel bounds for a specific row."""
        y0 = cls.content_start_y()
        y_start = y0 + row_idx * cls.row_height_with_gutter()
        y_end = y_start + cls.cell_height()
        return y_start, y_end


def extract_cluster_row_video(
    grid_video_path: str,
    cluster_idx: int,
    output_dir: str | None = None,
    horizontal_padding: int = 40,
    padding_color: tuple[int, int, int] = (255, 255, 255),
) -> str | None:
    """
    Extract a single cluster row from a grid video with optional padding.

    Args:
        grid_video_path: Path to the grid video
        cluster_idx: Index of the cluster row to extract (0-indexed)
        output_dir: Directory to save the extracted video (uses temp dir if None)
        horizontal_padding: Padding to add on left and right sides (pixels)
        padding_color: RGB color for padding (default white)

    Returns:
        Path to the extracted row video, or None if extraction failed
    """
    if not HAS_CV2:
        return None

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    os.makedirs(output_dir, exist_ok=True)

    video_name = Path(grid_video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_row{cluster_idx}.mp4")

    if os.path.exists(output_path):
        source_mtime = os.path.getmtime(grid_video_path)
        cached_mtime = os.path.getmtime(output_path)
        if cached_mtime >= source_mtime:
            return output_path
        os.remove(output_path)

    cap = cv2.VideoCapture(grid_video_path)
    if not cap.isOpened():
        print(f"[GUI] Failed to open video: {grid_video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y_start, y_end = GridVideoConfig.get_row_bounds(cluster_idx)
    y_start = max(0, y_start)
    y_end = min(frame_height, y_end)

    print(f"[GUI] Extracting row {cluster_idx}: y=[{y_start}, {y_end}] from {frame_height}px video")

    if y_end <= y_start:
        print(f"[GUI] Invalid row bounds for cluster {cluster_idx}")
        cap.release()
        return None

    row_height = y_end - y_start
    padded_width = frame_width + 2 * horizontal_padding
    padding_bgr = (padding_color[2], padding_color[1], padding_color[0])

    if HAS_FFMPEG:
        writer = imageio_ffmpeg.write_frames(
            output_path,
            (padded_width, row_height),
            fps=fps,
            codec="h264",
            macro_block_size=1,
        )
        writer.send(None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            row_frame = frame[y_start:y_end, :, :]
            padded_frame = cv2.copyMakeBorder(
                row_frame,
                top=0, bottom=0,
                left=horizontal_padding, right=horizontal_padding,
                borderType=cv2.BORDER_CONSTANT,
                value=padding_bgr,
            )
            padded_frame_rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
            writer.send(padded_frame_rgb)

        cap.release()
        writer.close()
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (padded_width, row_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            row_frame = frame[y_start:y_end, :, :]
            padded_frame = cv2.copyMakeBorder(
                row_frame,
                top=0, bottom=0,
                left=horizontal_padding, right=horizontal_padding,
                borderType=cv2.BORDER_CONSTANT,
                value=padding_bgr,
            )
            writer.write(padded_frame)

        cap.release()
        writer.release()

    if os.path.exists(output_path):
        print(f"[GUI] Extracted row video: {output_path}")
        return output_path
    return None


def parse_grid_video_info(video_path: str) -> tuple[int | None, int | None]:
    """
    Parse cluster count and seed from grid video filename.

    Expected format: clusters_k{n_clusters}_seed{seed}.mp4

    Returns:
        (n_clusters, seed) or (None, None) if not a grid video
    """
    basename = Path(video_path).name
    match = re.match(r'clusters_k(\d+)_seed(\d+)\.mp4', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class LabelingGUI:
    """GUI for cluster label assignment with recommendations."""

    def __init__(
        self,
        method_dir: str,
        seed: int,
        recommendations: dict[int, LabelRecommendation],
        cluster_ids: list[int],
    ):
        self.method_dir = Path(method_dir)
        self.seed = seed
        self.recommendations = recommendations
        self.cluster_ids = sorted(cluster_ids)
        self.current_idx = 0
        self.labels: dict[int, str] = {}
        self.turn_directions: dict[int, str] = {}

        self._grid_video_path: str | None = None
        self._row_video_cache_dir = self.method_dir / "row_videos"
        self._detect_grid_video()
        self._load_existing_labels()

    def _detect_grid_video(self) -> None:
        """Detect and store the grid video path if it exists."""
        patterns = [f"clusters_k*_seed{self.seed}.mp4"]
        for pattern in patterns:
            matches = list(self.method_dir.glob(pattern))
            if matches:
                self._grid_video_path = str(matches[0])
                self._row_video_cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"[GUI] Found grid video: {self._grid_video_path}")
                print(f"[GUI] Row cache dir: {self._row_video_cache_dir}")
                break
        if not self._grid_video_path:
            print(f"[GUI] No grid video found in {self.method_dir}")

    def _load_existing_labels(self) -> None:
        """Load existing labels and turn directions from YAML if present."""
        labels_path = self.method_dir / f"labels_seed_{self.seed}.yaml"
        if labels_path.exists():
            with open(labels_path, "r") as f:
                data = yaml.safe_load(f)
            existing = data.get("labels", {})
            for k, v in existing.items():
                if v:
                    self.labels[int(k)] = v
            existing_turns = data.get("turn_directions", {})
            for k, v in existing_turns.items():
                if v:
                    self.turn_directions[int(k)] = v
            print(f"[GUI] Loaded {len(self.labels)} labels and {len(self.turn_directions)} turn directions from {labels_path}")
        else:
            print(f"[GUI] No existing labels found at {labels_path}")

    def _get_video_path(self, cluster_id: int) -> str | None:
        """Get path to cluster video if it exists, extracting row from grid video if needed."""
        individual_pattern = f"cluster_{cluster_id}_seed{self.seed}.mp4"
        matches = list(self.method_dir.glob(individual_pattern))
        if matches:
            return str(matches[0])

        if self._grid_video_path and HAS_CV2:
            cluster_idx = self.cluster_ids.index(cluster_id)
            print(f"[GUI] Getting video for cluster {cluster_id} (index {cluster_idx})")
            row_video = extract_cluster_row_video(
                self._grid_video_path,
                cluster_idx,
                output_dir=str(self._row_video_cache_dir),
            )
            if row_video:
                return row_video
            print(f"[GUI] Row extraction failed, falling back to grid video")

        if self._grid_video_path:
            return self._grid_video_path

        print(f"[GUI] No video found for cluster {cluster_id}")
        return None

    def get_cluster_info(self, idx: int) -> tuple[int, str, list[tuple[str, float]], str, str]:
        """Get info for cluster at index."""
        if not self.cluster_ids:
            return -1, "No clusters", [], "", ""

        idx = max(0, min(idx, len(self.cluster_ids) - 1))
        cid = self.cluster_ids[idx]
        rec = self.recommendations.get(cid, LabelRecommendation(cid, []))
        current_label = self.labels.get(cid, "")
        current_turn = self.turn_directions.get(cid, "")
        return cid, f"Cluster {cid} ({idx + 1}/{len(self.cluster_ids)})", rec.labels, current_label, current_turn

    def assign_label(self, cluster_id: int, label: str) -> None:
        """Assign label to cluster."""
        if label.strip():
            self.labels[cluster_id] = label.strip()
        elif cluster_id in self.labels:
            del self.labels[cluster_id]

    def assign_turn_direction(self, cluster_id: int, direction: str) -> None:
        """Assign turn direction to cluster."""
        if direction and direction.strip():
            self.turn_directions[cluster_id] = direction.strip()
        elif cluster_id in self.turn_directions:
            del self.turn_directions[cluster_id]

    def save_labels(self) -> str:
        """Save labels and turn directions to YAML file."""
        labels_path = self.method_dir / f"labels_seed_{self.seed}.yaml"

        if labels_path.exists():
            with open(labels_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = {
                "method": self.method_dir.name,
                "seed": self.seed,
                "n_clusters": len(self.cluster_ids),
            }

        data["labels"] = {int(k): v for k, v in self.labels.items()}
        data["turn_directions"] = {int(k): v for k, v in self.turn_directions.items()}

        with open(labels_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return f"Saved labels to {labels_path}"

    def get_summary_df(self) -> list[list]:
        """Get summary data for display."""
        rows = []
        for cid in self.cluster_ids:
            assigned = self.labels.get(cid, "(unlabeled)")
            turn = self.turn_directions.get(cid, "-")
            rec = self.recommendations.get(cid)
            top_rec = rec.labels[0][0] if rec and rec.labels else "N/A"
            top_dist = f"{rec.labels[0][1]:.4f}" if rec and rec.labels else "N/A"
            rows.append([cid, assigned, turn, top_rec, top_dist])
        return rows

    def build_interface(self) -> "gr.Blocks":
        """Build and return Gradio interface."""
        if not HAS_GRADIO:
            raise ImportError("Gradio not installed. Run: pip install gradio")

        with gr.Blocks(title="Cluster Label Assignment", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Cluster Label Assignment Tool")

            with gr.Row():
                with gr.Column(scale=1):
                    cluster_nav = gr.Slider(
                        minimum=0,
                        maximum=max(0, len(self.cluster_ids) - 1),
                        step=1,
                        value=0,
                        label="Cluster Index",
                    )
                    with gr.Row():
                        prev_btn = gr.Button("< Previous")
                        next_btn = gr.Button("Next >")

                with gr.Column(scale=2):
                    cluster_title = gr.Markdown("### Cluster 0")

            with gr.Row():
                with gr.Column(scale=1):
                    video_player = gr.Video(label="Cluster Video", autoplay=True)

                with gr.Column(scale=1):
                    gr.Markdown("### Recommendations")
                    recommendations_md = gr.Markdown()

                    current_label = gr.Textbox(label="Current Label", interactive=False)

                    with gr.Row():
                        label_dropdown = gr.Dropdown(
                            label="Select Recommended Label",
                            choices=[],
                            interactive=True,
                        )
                        custom_label = gr.Textbox(
                            label="Or Enter Custom",
                            placeholder="e.g., rear walk",
                        )

                    gr.Markdown("### Turn Direction")
                    with gr.Row():
                        turn_dropdown = gr.Dropdown(
                            label="Turn Direction",
                            choices=["", "left", "right", "straight", "non-deterministic"],
                            value="",
                            interactive=True,
                        )
                        current_turn = gr.Textbox(label="Current Turn", interactive=False)

            gr.Markdown("### Labels Progress")
            summary_table = gr.Dataframe(
                headers=["Cluster", "Assigned", "Turn", "Top Rec", "Distance"],
                datatype=["number", "str", "str", "str", "str"],
            )

            with gr.Row():
                progress_md = gr.Markdown()
                save_btn = gr.Button("Save All Labels", variant="secondary")
                status_md = gr.Markdown()

            nav_outputs = [
                cluster_title, video_player, recommendations_md, current_label,
                label_dropdown, custom_label, turn_dropdown, current_turn,
                progress_md, summary_table,
            ]

            label_apply_outputs = [current_label, custom_label, current_turn, progress_md, summary_table]
            turn_apply_outputs = [current_label, current_turn, progress_md, summary_table]

            def _get_status(idx):
                cid, _, _, cur_label, cur_turn = self.get_cluster_info(int(idx))
                labeled = sum(1 for c in self.cluster_ids if c in self.labels)
                progress = f"**Labeled:** {labeled}/{len(self.cluster_ids)} clusters"
                summary = self.get_summary_df()
                return cur_label, cur_turn, progress, summary

            def update_display(idx):
                idx = int(idx)
                cid, title, recs, cur_label, cur_turn = self.get_cluster_info(idx)
                video = self._get_video_path(cid)

                rec_md = ""
                choices = []
                for i, (label, dist) in enumerate(recs):
                    marker = " **(recommended)**" if i == 0 else ""
                    rec_md += f"{i+1}. {label} (dist: {dist:.4f}){marker}\n"
                    choices.append(label)

                labeled = sum(1 for c in self.cluster_ids if c in self.labels)
                progress = f"**Labeled:** {labeled}/{len(self.cluster_ids)} clusters"

                summary = self.get_summary_df()

                if cur_label:
                    if cur_label not in choices:
                        choices.insert(0, cur_label)
                    dropdown_value = cur_label
                elif choices:
                    dropdown_value = choices[0]
                else:
                    dropdown_value = None

                return (
                    f"### {title}",
                    video,
                    rec_md,
                    cur_label,
                    gr.Dropdown(choices=choices, value=dropdown_value),
                    "",
                    cur_turn or "",
                    cur_turn,
                    progress,
                    summary,
                )

            def go_prev(idx):
                new_idx = max(0, int(idx) - 1)
                return (new_idx,) + update_display(new_idx)

            def go_next(idx):
                new_idx = min(len(self.cluster_ids) - 1, int(idx) + 1)
                return (new_idx,) + update_display(new_idx)

            def on_dropdown_change(idx, dropdown_val):
                if dropdown_val:
                    cid, _, _, _, _ = self.get_cluster_info(int(idx))
                    self.assign_label(cid, dropdown_val)
                cur_label, cur_turn, progress, summary = _get_status(int(idx))
                return cur_label, "", cur_turn, progress, summary

            def on_custom_submit(idx, custom_val):
                if custom_val.strip():
                    cid, _, _, _, _ = self.get_cluster_info(int(idx))
                    self.assign_label(cid, custom_val.strip())
                cur_label, cur_turn, progress, summary = _get_status(int(idx))
                return cur_label, "", cur_turn, progress, summary

            def on_turn_change(idx, turn_val):
                cid, _, _, _, _ = self.get_cluster_info(int(idx))
                self.assign_turn_direction(cid, turn_val or "")
                return _get_status(int(idx))

            def save_labels_fn():
                msg = self.save_labels()
                return msg

            cluster_nav.change(
                update_display,
                inputs=[cluster_nav],
                outputs=nav_outputs,
            )

            prev_btn.click(go_prev, inputs=[cluster_nav], outputs=[cluster_nav] + nav_outputs)
            next_btn.click(go_next, inputs=[cluster_nav], outputs=[cluster_nav] + nav_outputs)

            label_dropdown.input(
                on_dropdown_change,
                inputs=[cluster_nav, label_dropdown],
                outputs=label_apply_outputs,
            )

            custom_label.submit(
                on_custom_submit,
                inputs=[cluster_nav, custom_label],
                outputs=label_apply_outputs,
            )

            turn_dropdown.input(
                on_turn_change,
                inputs=[cluster_nav, turn_dropdown],
                outputs=turn_apply_outputs,
            )

            save_btn.click(save_labels_fn, outputs=[status_md])

            demo.load(
                update_display,
                inputs=[cluster_nav],
                outputs=nav_outputs,
            )

        return demo


def launch_gui(
    config: Any,
    results: dict,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the labeling GUI."""
    if not HAS_GRADIO:
        raise ImportError("Gradio not installed. Run: pip install gradio")

    method_dir = os.path.join(config.experiment_dir, config.method)

    gui = LabelingGUI(
        method_dir=method_dir,
        seed=config.seed,
        recommendations=results["recommendations"],
        cluster_ids=results["cluster_ids"],
    )

    demo = gui.build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


def launch_gui_standalone(
    experiment_dir: str,
    method: str,
    seed: int = 0,
    recommendations_csv: str | None = None,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch GUI from saved recommendations."""
    if not HAS_GRADIO:
        raise ImportError("Gradio not installed. Run: pip install gradio")

    import csv
    from topo_mimic.analysis.feature_analysis.data_loading import load_cluster_dict
    from topo_mimic.labeling.core import LabelRecommendation

    method_dir = os.path.join(experiment_dir, method)
    cluster_dict = load_cluster_dict(method_dir, seed)
    cluster_ids = sorted(cluster_dict.keys())

    recommendations: dict[int, LabelRecommendation] = {}

    if recommendations_csv is None:
        recommendations_csv = os.path.join(method_dir, "labeling", f"recommendations_seed_{seed}.csv")

    if os.path.exists(recommendations_csv):
        with open(recommendations_csv, "r") as f:
            reader = csv.DictReader(f)
            current_cid = None
            current_labels = []
            for row in reader:
                cid = int(row["cluster_id"])
                if current_cid is not None and cid != current_cid:
                    recommendations[current_cid] = LabelRecommendation(current_cid, current_labels)
                    current_labels = []
                current_cid = cid
                current_labels.append((row["label"], float(row["distance"])))
            if current_cid is not None:
                recommendations[current_cid] = LabelRecommendation(current_cid, current_labels)
    else:
        for cid in cluster_ids:
            recommendations[cid] = LabelRecommendation(cid, [])

    gui = LabelingGUI(
        method_dir=method_dir,
        seed=seed,
        recommendations=recommendations,
        cluster_ids=cluster_ids,
    )

    demo = gui.build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
