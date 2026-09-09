#!/usr/bin/env python3
"""Fish Tail Boundary Analysis Agent.

Autonomous analysis of multi-channel Zeiss .czi microscopy files.
Outputs per-image JSON metrics, Fiji/ImageJ .roi boundaries, QA PNGs,
a Prism-ready batch CSV, and a reproducible processing log.

Built from the user's Fish Tail Boundary Analysis Agent specification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


BF_KEYWORDS = ("brightfield", "bright field", "transmitted", "transmission", "tl", "bf")


@dataclass
class AgentConfig:
    pixel_size_um: Optional[float] = None
    channel_index: Optional[int] = None
    threshold_value: Optional[int] = None
    min_area_px2: float = 100.0
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    morphology_kernel_size: int = 5
    rdp_epsilon_px: float = 1.5
    save_png: bool = True
    save_roi: bool = True
    verbose: bool = True


class ProcessingLogger:
    def __init__(self, log_path: Path, verbose: bool = True):
        self.log_path = Path(log_path)
        self.verbose = verbose
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.verbose:
            print(line)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class FishTailAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.results: List[Dict[str, Any]] = []
        self.logger: Optional[ProcessingLogger] = None

    # ------------------------------ utilities ------------------------------
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)
        elif self.config.verbose:
            print(f"[AGENT] {message}")

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D image after channel/z selection; got shape {arr.shape}")
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            raise ValueError("Image contains no finite pixel values")
        lo, hi = np.percentile(arr[finite], [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(np.min(arr[finite])), float(np.max(arr[finite]))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        return np.round(arr * 255).astype(np.uint8)

    @staticmethod
    def _contour_solidity(contour: np.ndarray) -> float:
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        return area / hull_area if hull_area > 0 else 0.0

    @staticmethod
    def _contour_aspect_ratio(contour: np.ndarray) -> float:
        _, _, w, h = cv2.boundingRect(contour)
        return float(w) / float(h) if h else 0.0

    # ------------------------------ CZI reading -----------------------------
    def read_czi(self, filepath: Path) -> Dict[str, Any]:
        """Read a .czi using AICSImageIO when available, then czifile.

        The user's original package names `aicspython`; in practice, readers vary by
        environment. This implementation accepts AICSImageIO if present and keeps
        czifile as the lightweight fallback, while failing clearly if neither exists.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)
        if filepath.suffix.lower() != ".czi":
            raise ValueError(f"Expected .czi input, got: {filepath.name}")

        errors: List[str] = []

        # AICSImageIO gives dimension labels and channel names, which makes robust
        # channel/z handling possible.
        try:
            from aicsimageio import AICSImage  # type: ignore

            img = AICSImage(str(filepath))
            data = img.get_image_data("CZYX", T=0, S=0)
            channel_names = list(img.channel_names or [])
            pps = getattr(img, "physical_pixel_sizes", None)
            pixel_size = None
            if pps is not None and getattr(pps, "X", None):
                pixel_size = float(pps.X)
            return {
                "array": np.asarray(data),
                "reader": "aicsimageio",
                "channel_names": channel_names,
                "pixel_size_um_metadata": pixel_size,
                "raw_shape": list(data.shape),
            }
        except Exception as exc:
            errors.append(f"aicsimageio: {exc}")

        try:
            import czifile  # type: ignore

            with czifile.CziFile(str(filepath)) as czi:
                data = np.asarray(czi.asarray())
                axes = getattr(czi, "axes", "")
            return {
                "array": data,
                "reader": "czifile",
                "axes": axes,
                "channel_names": [],
                "pixel_size_um_metadata": None,
                "raw_shape": list(data.shape),
            }
        except Exception as exc:
            errors.append(f"czifile: {exc}")

        raise ImportError(
            "Could not read .czi. Install a CZI reader, e.g. `pip install aicsimageio aicspylibczi` "
            "or `pip install czifile`. Reader errors: " + " | ".join(errors)
        )

    def _coerce_czi_to_czyx(self, payload: Dict[str, Any]) -> np.ndarray:
        arr = np.asarray(payload["array"])
        if payload.get("reader") == "aicsimageio":
            if arr.ndim != 4:
                raise ValueError(f"AICS reader returned unexpected CZYX shape {arr.shape}")
            return arr

        # czifile can include singleton dimensions and explicit axes labels.
        axes = str(payload.get("axes", ""))
        if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
            # Pick index 0 for scene/time/other dimensions, retain C/Z/Y/X.
            slicer: List[Any] = []
            kept_axes: List[str] = []
            for ax, size in zip(axes, arr.shape):
                if ax in "CZYX":
                    slicer.append(slice(None))
                    kept_axes.append(ax)
                else:
                    slicer.append(0)
            arr2 = np.asarray(arr[tuple(slicer)])
            # Add missing C/Z axes.
            current_axes = kept_axes.copy()
            for needed in "CZ":
                if needed not in current_axes:
                    arr2 = np.expand_dims(arr2, axis=0)
                    current_axes.insert(0, needed)
            perm = [current_axes.index(a) for a in "CZYX"]
            return np.transpose(arr2, perm)

        # Heuristic fallback only when axis metadata is unavailable.
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            return squeezed[None, None, :, :]
        if squeezed.ndim == 3:
            # Small first dimension is more likely channels; otherwise z stack.
            if 1 <= squeezed.shape[0] <= 8:
                return squeezed[:, None, :, :]
            return squeezed[None, :, :, :]
        if squeezed.ndim == 4:
            # Assume C,Z,Y,X if first two axes are plausibly small.
            if squeezed.shape[0] <= 8:
                return squeezed
        raise ValueError(
            f"Could not infer channel/z axes from shape {arr.shape}. Use a CZI reader with axis metadata."
        )

    def select_brightfield(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        czyx = self._coerce_czi_to_czyx(payload)
        n_channels, n_z, height, width = czyx.shape
        channel_names = payload.get("channel_names") or []

        channel_stats = []
        for i in range(n_channels):
            sample = czyx[i].astype(np.float32)
            mean = float(np.mean(sample))
            std = float(np.std(sample))
            flat = sample.ravel()
            if flat.size > 200_000:
                flat = flat[:: max(1, flat.size // 200_000)]
            sk = float(skew(flat, bias=False, nan_policy="omit")) if flat.size > 2 else 0.0
            channel_stats.append({"index": i, "mean": mean, "std": std, "skewness": sk})

        reason = "statistical heuristic"
        confidence = 0.6
        if self.config.channel_index is not None:
            idx = self.config.channel_index
            if not 0 <= idx < n_channels:
                raise ValueError(f"--channel {idx} is invalid; image has {n_channels} channels")
            reason = "user override"
            confidence = 1.0
        else:
            idx = -1
            # Metadata first.
            for i, name in enumerate(channel_names):
                low = str(name).strip().lower()
                if any(k == low or k in low for k in BF_KEYWORDS):
                    idx = i
                    reason = f"channel metadata: {name}"
                    confidence = 0.98
                    break
            if idx < 0:
                means = np.array([s["mean"] for s in channel_stats], dtype=float)
                stds = np.array([s["std"] for s in channel_stats], dtype=float)
                valid = stds > np.maximum(1e-9, 0.10 * np.maximum(means, 1e-9))
                candidates = np.where(valid)[0]
                if candidates.size == 0:
                    candidates = np.arange(n_channels)
                idx = int(candidates[np.argmax(means[candidates])])
                if n_channels > 1:
                    sorted_means = np.sort(means)[::-1]
                    sep = (sorted_means[0] - sorted_means[1]) / max(abs(sorted_means[0]), 1e-9)
                    confidence = float(np.clip(0.60 + 0.35 * sep, 0.60, 0.95))
                else:
                    confidence = 0.95

        selected = czyx[idx]
        if n_z > 1:
            projected = np.max(selected, axis=0)
            projection = "maximum intensity projection"
        else:
            projected = selected[0]
            projection = "single plane"

        info = {
            "channels_detected": n_channels,
            "z_planes": n_z,
            "height": height,
            "width": width,
            "brightfield_channel_index": idx,
            "brightfield_channel_name": channel_names[idx] if idx < len(channel_names) else None,
            "channel_selection_reason": reason,
            "channel_selection_confidence": confidence,
            "channel_statistics": channel_stats,
            "projection_method": projection,
        }
        self._log(
            f"Brightfield channel {idx} selected ({reason}, confidence={confidence:.2f}); "
            f"z={n_z}, projection={projection}"
        )
        return np.asarray(projected), info

    # ------------------------------ processing -----------------------------
    def enhance_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        norm = self._normalize_to_uint8(image)
        focus_before = float(cv2.Laplacian(norm, cv2.CV_64F).var())
        denoised = norm
        denoising_applied = False
        # The specification requests automatic denoising on low-quality/noisy images.
        # Laplacian variance is primarily a focus metric, so use conservative denoising.
        if focus_before < 20 and np.std(norm) > 8:
            denoised = cv2.fastNlMeansDenoising(norm, None, h=7, templateWindowSize=7, searchWindowSize=21)
            denoising_applied = True

        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clahe_clip_limit),
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        )
        enhanced = clahe.apply(denoised)
        return enhanced, {
            "focus_laplacian_variance_preprocessing": focus_before,
            "denoising_applied": denoising_applied,
            "clahe_clip_limit": self.config.clahe_clip_limit,
            "clahe_tile_size": self.config.clahe_tile_size,
        }

    def _threshold_candidates(self, enhanced: np.ndarray) -> Iterable[Tuple[str, float, np.ndarray]]:
        if self.config.threshold_value is not None:
            t = int(self.config.threshold_value)
            _, b = cv2.threshold(enhanced, t, 255, cv2.THRESH_BINARY)
            yield "manual", float(t), b
            return

        t, b = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield "otsu", float(t), b
        t2, b2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        yield "triangle", float(t2), b2
        # OpenCV adaptive Gaussian provides a robust local fallback where Niblack is
        # unavailable in base opencv-python.
        b3 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
        yield "adaptive_gaussian", float("nan"), b3

    def _clean_binary(self, binary: np.ndarray) -> np.ndarray:
        k = int(self.config.morphology_kernel_size)
        if k < 1:
            return binary
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _find_viable_contours(self, binary: np.ndarray, min_area: float) -> Tuple[List[np.ndarray], int]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape
        image_area = float(h * w)
        viable = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > 0.90 * image_area:
                continue
            ar = self._contour_aspect_ratio(c)
            if not 0.1 <= ar <= 10.0:
                continue
            viable.append(c)
        return viable, len(contours)

    def segment_tail(self, enhanced: np.ndarray) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        for method, threshold, binary0 in self._threshold_candidates(enhanced):
            for polarity, raw_binary in (("normal", binary0), ("inverted", cv2.bitwise_not(binary0))):
                cleaned = self._clean_binary(raw_binary)
                viable, total = self._find_viable_contours(cleaned, self.config.min_area_px2)
                if not viable and self.config.min_area_px2 > 50:
                    viable, total = self._find_viable_contours(cleaned, 50.0)
                    min_area_used = 50.0
                else:
                    min_area_used = self.config.min_area_px2

                attempt = {
                    "method": method,
                    "threshold": None if math.isnan(threshold) else threshold,
                    "polarity": polarity,
                    "contours_total": total,
                    "contours_viable": len(viable),
                    "min_area_used": min_area_used,
                }
                attempts.append(attempt)
                if not viable:
                    continue

                # Primary target is largest. If several are comparable, select the
                # most solid among candidates within 70% of the largest area.
                viable_sorted = sorted(viable, key=cv2.contourArea, reverse=True)
                largest_area = float(cv2.contourArea(viable_sorted[0]))
                comparable = [c for c in viable_sorted if cv2.contourArea(c) >= 0.70 * largest_area]
                ambiguity = len(comparable) > 1
                if ambiguity:
                    contour = max(comparable, key=self._contour_solidity)
                else:
                    contour = viable_sorted[0]

                area = float(cv2.contourArea(contour))
                solidity = self._contour_solidity(contour)
                h, w = enhanced.shape
                area_fraction = area / float(h * w)
                # Candidate score favors substantial, solid contours while penalizing
                # suspiciously image-filling regions and ambiguity.
                score = (
                    0.45 * np.clip(solidity, 0, 1)
                    + 0.35 * np.clip(area_fraction / 0.35, 0, 1)
                    + 0.20 * (1.0 if len(viable) <= 5 else max(0.0, 1 - (len(viable) - 5) / 30))
                )
                if area_fraction > 0.70:
                    score *= 0.35
                if ambiguity:
                    score *= 0.85

                candidate = {
                    "contour": contour,
                    "binary": cleaned,
                    "method": method,
                    "threshold": None if math.isnan(threshold) else threshold,
                    "polarity": polarity,
                    "contours_total": total,
                    "contours_viable": len(viable),
                    "ambiguity": ambiguity,
                    "candidate_score": float(score),
                    "area_fraction": area_fraction,
                }
                if best is None or candidate["candidate_score"] > best["candidate_score"]:
                    best = candidate

            # Prefer a good first-method result rather than trying every fallback.
            if best is not None and method in ("manual", "otsu") and best["candidate_score"] >= 0.72:
                break

        if best is None:
            raise ValueError("No viable tail contour detected after automatic recovery attempts")

        original_contour = best["contour"]
        epsilon = max(0.0, float(self.config.rdp_epsilon_px))
        if epsilon > 0:
            smoothed = cv2.approxPolyDP(original_contour, epsilon, True)
            if len(smoothed) >= 3:
                best["contour"] = smoothed
        best["original_contour_points"] = int(len(original_contour))
        best["smoothed_contour_points"] = int(len(best["contour"]))
        best["attempts"] = attempts
        self._log(
            f"Segmentation: {best['method']} ({best['polarity']}), viable contours={best['contours_viable']}, "
            f"candidate score={best['candidate_score']:.2f}"
        )
        return best

    def quantify(self, contour: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w / h) if h else 0.0
        circularity = float(4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        points = contour.reshape(-1, 2).astype(np.float64)
        eccentricity = 0.0
        orientation = 0.0
        if len(points) >= 3:
            centered = points - np.mean(points, axis=0)
            cov = np.cov(centered, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = np.maximum(vals[order], 0)
            vecs = vecs[:, order]
            if vals[0] > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - vals[1] / vals[0])))
            orientation = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
            if orientation > 90:
                orientation -= 180
            elif orientation < -90:
                orientation += 180

        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx, cy = float(x + w / 2), float(y + h / 2)

        mask = np.zeros(enhanced.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        pixels = enhanced[mask > 0]
        if pixels.size:
            intensity = {
                "mean_intensity": float(np.mean(pixels)),
                "intensity_std": float(np.std(pixels)),
                "intensity_min": int(np.min(pixels)),
                "intensity_max": int(np.max(pixels)),
            }
        else:
            intensity = {"mean_intensity": 0.0, "intensity_std": 0.0, "intensity_min": 0, "intensity_max": 0}

        metrics: Dict[str, Any] = {
            "area_pixels_sq": area,
            "perimeter_pixels": perimeter,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "eccentricity": eccentricity,
            **intensity,
            "centroid_x": cx,
            "centroid_y": cy,
            "orientation_angle_degrees": orientation,
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        }
        if self.config.pixel_size_um is not None:
            px = float(self.config.pixel_size_um)
            metrics["area_micrometers_sq"] = area * px * px
            metrics["perimeter_micrometers"] = perimeter * px
        return metrics

    def assess_quality(
        self, enhanced: np.ndarray, metrics: Dict[str, Any], segmentation: Dict[str, Any]
    ) -> Dict[str, Any]:
        lap_var = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
        # Focus score: continuous 0..100, with >50 variance reaching strong quality.
        focus_score = float(np.clip(100 * lap_var / 50.0, 0, 100))

        hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256]).ravel().astype(float)
        hist /= max(hist.sum(), 1.0)
        # Between-class variance proxy from selected threshold where available.
        threshold = segmentation.get("threshold")
        if threshold is not None:
            t = int(np.clip(round(threshold), 1, 254))
            w0, w1 = hist[: t + 1].sum(), hist[t + 1 :].sum()
            if w0 > 0 and w1 > 0:
                mu0 = np.dot(np.arange(t + 1), hist[: t + 1]) / w0
                mu1 = np.dot(np.arange(t + 1, 256), hist[t + 1 :]) / w1
                separation = abs(mu1 - mu0) / 255.0
            else:
                separation = 0.0
        else:
            separation = float(np.std(enhanced) / 128.0)
        separation = float(np.clip(separation, 0, 1))

        contour_cleanliness = float(
            np.clip(1.0 - max(0, segmentation["contours_viable"] - 1) / 20.0, 0.25, 1.0)
        )
        solidity_component = float(np.clip(metrics["solidity"], 0, 1))
        segmentation_confidence = float(
            np.clip(
                0.40 * separation + 0.25 * contour_cleanliness + 0.25 * solidity_component
                + 0.10 * segmentation["candidate_score"],
                0,
                1,
            )
        )
        if segmentation.get("ambiguity"):
            segmentation_confidence *= 0.85
        if segmentation.get("area_fraction", 0) > 0.70:
            segmentation_confidence *= 0.5

        plausibility_failures = []
        if not 0.5 <= metrics["aspect_ratio"] <= 5.0:
            plausibility_failures.append("aspect_ratio")
        if not 0.5 <= metrics["circularity"] <= 0.95:
            plausibility_failures.append("circularity")
        if not 0.70 <= metrics["solidity"] <= 0.99:
            plausibility_failures.append("solidity")
        if not plausibility_failures:
            plausibility = "pass"
        elif len(plausibility_failures) == 1:
            plausibility = "warning"
        else:
            plausibility = "fail"

        quality_score = int(round(np.clip(0.65 * focus_score + 35 * segmentation_confidence, 0, 100)))
        if quality_score > 75 and segmentation_confidence > 0.85 and plausibility == "pass":
            flag = "good"
            recommendation = "Proceed"
        elif quality_score < 50 or segmentation_confidence < 0.70 or plausibility == "fail":
            flag = "poor"
            recommendation = "Consider re-capturing image or manual trace"
        else:
            flag = "review"
            recommendation = "Check PNG/ROI for boundary accuracy"

        return {
            "laplacian_variance": lap_var,
            "quality_score": quality_score,
            "segmentation_confidence": segmentation_confidence,
            "anatomical_plausibility": plausibility,
            "plausibility_failures": plausibility_failures,
            "quality_flag": flag,
            "recommendation": recommendation,
        }

    # ------------------------------- output --------------------------------
    def _save_roi(self, contour: np.ndarray, out_path: Path) -> Optional[str]:
        try:
            import roifile  # type: ignore
        except Exception as exc:
            self._log(f"ROI export skipped: roifile unavailable ({exc})")
            return None
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            raise ValueError("Contour has fewer than 3 points; cannot create polygon ROI")
        # ImagejRoi is the stable API used by current roifile releases.
        if hasattr(roifile, "ImagejRoi"):
            roi = roifile.ImagejRoi.frompoints(points.astype(np.int16), name=out_path.stem)
            roi.tofile(str(out_path))
        elif hasattr(roifile, "Roi"):
            roi = roifile.Roi(x=points[:, 0], y=points[:, 1])
            roifile.roiwrite(str(out_path), roi)
        else:
            raise RuntimeError("Unsupported roifile API")
        return str(out_path)

    def _save_png(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        binary: np.ndarray,
        contour: np.ndarray,
        metrics: Dict[str, Any],
        quality: Dict[str, Any],
        out_path: Path,
        filename: str,
    ) -> str:
        original_u8 = self._normalize_to_uint8(original)
        overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        panels = [
            (original_u8, "Original Brightfield", "gray"),
            (enhanced, "CLAHE Enhanced", "gray"),
            (binary, "Binary Segmentation", "gray"),
            (overlay, "Detected Tail Boundary", None),
        ]
        for ax, (img, title, cmap) in zip(axes.ravel(), panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
        axes[1, 1].text(
            0.02,
            0.98,
            f"Area: {metrics['area_pixels_sq']:.1f} px²\n"
            f"Perimeter: {metrics['perimeter_pixels']:.1f} px\n"
            f"Quality: {quality['quality_score']}/100\n"
            f"Flag: {quality['quality_flag']}",
            transform=axes[1, 1].transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )
        fig.suptitle(f"{filename} - Boundary Trace Analysis\n{self._utc_timestamp()}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): FishTailAgent._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [FishTailAgent._json_safe(v) for v in obj]
        return obj

    def process_array(
        self,
        image_2d: np.ndarray,
        source_name: str = "array",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a 2D image. Exposed for validation/testing without CZI I/O."""
        enhanced, prep = self.enhance_image(image_2d)
        segmentation = self.segment_tail(enhanced)
        metrics = self.quantify(segmentation["contour"], enhanced)
        quality = self.assess_quality(enhanced, metrics, segmentation)
        return {
            "source_name": source_name,
            "enhanced": enhanced,
            "binary": segmentation["binary"],
            "contour": segmentation["contour"],
            "preprocessing": prep,
            "segmentation": segmentation,
            "metrics": metrics,
            "quality": quality,
            "metadata": metadata or {},
        }

    def process_file(self, filepath: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        filepath = Path(filepath)
        output_dir = Path(output_dir) if output_dir else filepath.parent / "analyzed"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.logger is None:
            self.logger = ProcessingLogger(output_dir / "processing_log.txt", self.config.verbose)

        self._log(f"Processing: {filepath.name}")
        payload = self.read_czi(filepath)
        bf, selection = self.select_brightfield(payload)
        processed = self.process_array(bf, filepath.name, metadata=selection)

        pixel_size = self.config.pixel_size_um
        if pixel_size is None and payload.get("pixel_size_um_metadata"):
            pixel_size = float(payload["pixel_size_um_metadata"])
            # Quantification already ran; add calibrated metrics here.
            processed["metrics"]["area_micrometers_sq"] = processed["metrics"]["area_pixels_sq"] * pixel_size**2
            processed["metrics"]["perimeter_micrometers"] = processed["metrics"]["perimeter_pixels"] * pixel_size

        stem = filepath.stem
        json_path = output_dir / f"{stem}_metrics.json"
        roi_path = output_dir / f"{stem}_boundary.roi"
        png_path = output_dir / f"{stem}_boundary_trace.png"

        seg = processed["segmentation"]
        record = {
            "file_metadata": {
                "filename": filepath.name,
                "processing_timestamp": self._utc_timestamp(),
                "reader_library": payload.get("reader"),
                "original_shape": payload.get("raw_shape"),
                "channels_detected": selection["channels_detected"],
                "brightfield_channel_index": selection["brightfield_channel_index"],
                "brightfield_channel_name": selection["brightfield_channel_name"],
                "channel_selection_confidence": selection["channel_selection_confidence"],
                "z_planes": selection["z_planes"],
                "projection_method": selection["projection_method"],
                "pixel_size_um": pixel_size,
            },
            "image_quality": processed["quality"],
            "morphometric_measurements": processed["metrics"],
            "processing_parameters": {
                **asdict(self.config),
                "threshold_method_used": seg["method"],
                "threshold_value_used": seg["threshold"],
                "threshold_polarity": seg["polarity"],
                "original_contour_points": seg["original_contour_points"],
                "smoothed_contour_points": seg["smoothed_contour_points"],
            },
            "processing_notes": {
                "channel_selection_reason": selection["channel_selection_reason"],
                "preprocessing": processed["preprocessing"],
                "segmentation_attempts": seg["attempts"],
                "multiple_large_contours": seg["ambiguity"],
            },
        }
        json_path.write_text(json.dumps(self._json_safe(record), indent=2), encoding="utf-8")

        roi_saved = None
        if self.config.save_roi:
            try:
                roi_saved = self._save_roi(processed["contour"], roi_path)
            except Exception as exc:
                self._log(f"ROI generation failed for {filepath.name}: {exc}")

        png_saved = None
        if self.config.save_png:
            try:
                png_saved = self._save_png(
                    bf,
                    processed["enhanced"],
                    processed["binary"],
                    processed["contour"],
                    processed["metrics"],
                    processed["quality"],
                    png_path,
                    filepath.name,
                )
            except Exception as exc:
                self._log(f"PNG generation failed for {filepath.name}: {exc}")

        summary = self._summary_row(filepath.name, record)
        self.results.append(summary)
        self._log(
            f"Completed {filepath.name}: area={summary['Area_px2']:.1f} px², "
            f"quality={summary['QualityScore']}, flag={summary['Quality_Flag']}"
        )
        return {"record": record, "json": str(json_path), "roi": roi_saved, "png": png_saved, "summary": summary}

    def _summary_row(self, filename: str, record: Dict[str, Any]) -> Dict[str, Any]:
        m = record["morphometric_measurements"]
        q = record["image_quality"]
        row = {
            "Filename": filename,
            "Area_px2": m["area_pixels_sq"],
            "Perimeter_px": m["perimeter_pixels"],
            "Aspect_Ratio": m["aspect_ratio"],
            "Circularity": m["circularity"],
            "Solidity": m["solidity"],
            "Eccentricity": m["eccentricity"],
            "Mean_Intensity": m["mean_intensity"],
            "Intensity_Std": m["intensity_std"],
            "Centroid_X": m["centroid_x"],
            "Centroid_Y": m["centroid_y"],
            "Orientation_Deg": m["orientation_angle_degrees"],
            "QualityScore": q["quality_score"],
            "SegmentationConfidence": q["segmentation_confidence"],
            "Quality_Flag": q["quality_flag"],
        }
        if "area_micrometers_sq" in m:
            row["Area_um2"] = m["area_micrometers_sq"]
            row["Perimeter_um"] = m["perimeter_micrometers"]
        return row

    def export_summary_csv(self, output_dir: Path) -> Optional[Path]:
        if not self.results:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tail_metrics_summary.csv"
        fields = list(self.results[0].keys())
        # Include calibrated columns if any record has them.
        for extra in ("Area_um2", "Perimeter_um"):
            if any(extra in r for r in self.results) and extra not in fields:
                fields.append(extra)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        self._log(f"Summary CSV saved: {path}")
        return path

    def process_directory(self, directory: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        output_dir = Path(output_dir) if output_dir else directory / "analyzed"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ProcessingLogger(output_dir / "processing_log.txt", self.config.verbose)
        files = sorted(directory.glob("*.czi"))
        self._log("========== Fish Tail Boundary Analysis ==========")
        self._log(f"Directory: {directory}")
        self._log(f"Found {len(files)} .czi file(s)")
        if not files:
            return {"total": 0, "successful": 0, "failed": 0, "errors": []}

        errors = []
        successful = 0
        for idx, filepath in enumerate(files, 1):
            self._log(f"[{idx}/{len(files)}] {filepath.name}")
            try:
                self.process_file(filepath, output_dir)
                successful += 1
            except Exception as exc:
                errors.append({"filename": filepath.name, "error": str(exc)})
                self._log(f"ERROR {filepath.name}: {exc}")
                if self.config.verbose:
                    self._log(traceback.format_exc().rstrip())

        csv_path = self.export_summary_csv(output_dir)
        counts: Dict[str, int] = {"good": 0, "review": 0, "poor": 0}
        for r in self.results:
            counts[r["Quality_Flag"]] = counts.get(r["Quality_Flag"], 0) + 1
        self._log("========== BATCH SUMMARY ==========")
        self._log(f"Total={len(files)} Successful={successful} Failed={len(errors)}")
        self._log(f"Quality: {counts}")
        if csv_path:
            self._log(f"Prism-ready CSV: {csv_path}")
        return {"total": len(files), "successful": successful, "failed": len(errors), "errors": errors}


# ----------------------------------- CLI -----------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous Fish Tail Boundary Analysis Agent for Zeiss .czi microscopy images."
    )
    p.add_argument("input", help="Path to a .czi file or directory containing .czi files")
    p.add_argument("--batch", action="store_true", help="Process all .czi files in a directory")
    p.add_argument("--pixel_size", type=float, default=None, help="Pixel size in micrometers")
    p.add_argument("--channel", type=int, default=None, help="Force brightfield channel index")
    p.add_argument("--threshold", type=int, default=None, help="Manual threshold value (0-255)")
    p.add_argument("--min_area", type=float, default=100.0, help="Minimum contour area in px²")
    p.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit (default 2.0)")
    p.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    p.add_argument("--roi-only", action="store_true", help="Save ROI, skip PNG")
    p.add_argument("--png-only", action="store_true", help="Save PNG, skip ROI")
    p.add_argument("--no-png", action="store_true", help="Do not save PNG")
    p.add_argument("--no-roi", action="store_true", help="Do not save ROI")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="Detailed console logging")
    group.add_argument("--quiet", action="store_true", help="Minimal console output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threshold is not None and not 0 <= args.threshold <= 255:
        raise SystemExit("--threshold must be between 0 and 255")
    if args.pixel_size is not None and args.pixel_size <= 0:
        raise SystemExit("--pixel_size must be > 0")
    if args.min_area <= 0:
        raise SystemExit("--min_area must be > 0")

    save_png, save_roi = True, True
    if args.roi_only:
        save_png, save_roi = False, True
    elif args.png_only:
        save_png, save_roi = True, False
    else:
        if args.no_png:
            save_png = False
        if args.no_roi:
            save_roi = False

    config = AgentConfig(
        pixel_size_um=args.pixel_size,
        channel_index=args.channel,
        threshold_value=args.threshold,
        min_area_px2=args.min_area,
        clahe_clip_limit=args.clahe_clip,
        save_png=save_png,
        save_roi=save_roi,
        verbose=not args.quiet,
    )
    agent = FishTailAgent(config)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        if args.batch or input_path.is_dir():
            summary = agent.process_directory(input_path, output_dir)
            return 0 if summary["failed"] == 0 else 2
        result = agent.process_file(input_path, output_dir)
        out_dir = Path(result["json"]).parent
        agent.export_summary_csv(out_dir)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if agent.logger is not None:
            agent.logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
