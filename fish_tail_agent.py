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

_MOBILE_SAM_MODEL = None
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
                "pixel_size_source": "aicsimageio_physical_pixel_sizes" if pixel_size else None,
                "raw_shape": list(data.shape),
            }
        except Exception as exc:
            errors.append(f"aicsimageio: {exc}")

        try:
            import czifile  # type: ignore
            import xml.etree.ElementTree as ET

            pixel_size = None
            pixel_size_source = None
            with czifile.CziFile(str(filepath)) as czi:
                data = np.asarray(czi.asarray())
                axes = getattr(czi, "axes", "")

                # Zeiss stores spatial scaling in CZI XML metadata. Values are
                # normally in meters; convert X scaling to micrometers/pixel.
                try:
                    meta = czi.metadata()
                    if isinstance(meta, bytes):
                        meta = meta.decode("utf-8", errors="ignore")
                    if meta:
                        root = ET.fromstring(meta)
                        for elem in root.iter():
                            tag = elem.tag.split("}")[-1]
                            if tag == "Distance" and str(elem.attrib.get("Id", "")).upper() == "X":
                                value = None
                                for child in elem.iter():
                                    if child.tag.split("}")[-1] == "Value" and child.text:
                                        value = float(child.text)
                                        break
                                if value and value > 0:
                                    # CZI physical distance values are typically meters.
                                    pixel_size = value * 1e6 if value < 1e-3 else value
                                    pixel_size_source = "czi_metadata_scaling_x"
                                    break
                except Exception:
                    pixel_size = None
                    pixel_size_source = None

            return {
                "array": data,
                "reader": "czifile",
                "axes": axes,
                "channel_names": [],
                "pixel_size_um_metadata": pixel_size,
                "pixel_size_source": pixel_size_source,
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
        """Test 1 preprocessing: flatten illumination and suppress fine internal texture.

        The first real brightfield images showed that CLAHE strongly emphasized fin rays
        and internal body structures while the translucent outer fin margin remained
        comparatively weak. This test uses large-scale background correction followed by
        edge-preserving smoothing and mild CLAHE. The goal is not simply "more contrast";
        it is to favor broad anatomical boundaries over fine internal texture.
        """
        norm = self._normalize_to_uint8(image)
        focus_before = float(cv2.Laplacian(norm, cv2.CV_64F).var())

        h, w = norm.shape
        # Large Gaussian estimates illumination/background. Scale with image size so the
        # method behaves similarly on different acquisitions.
        sigma_bg = max(18.0, min(h, w) * 0.035)
        background = cv2.GaussianBlur(norm, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)

        # Signed background correction, recentered at mid-gray to retain both bright and
        # dark fin-margin deviations.
        flat = norm.astype(np.float32) - background.astype(np.float32)
        lo, hi = np.percentile(flat, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        flat_u8 = np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        # Bilateral filtering suppresses small internal texture while preserving larger
        # edges such as the outside fin boundary.
        smooth = cv2.bilateralFilter(flat_u8, d=9, sigmaColor=32, sigmaSpace=11)

        # Mild CLAHE after background correction; deliberately weaker than the old
        # texture-amplifying pipeline.
        clahe = cv2.createCLAHE(
            clipLimit=min(float(self.config.clahe_clip_limit), 1.6),
            tileGridSize=(12, 12),
        )
        enhanced = clahe.apply(smooth)

        # A small unsharp component restores broad boundary definition after smoothing.
        broad = cv2.GaussianBlur(enhanced, (0, 0), 2.2)
        enhanced = cv2.addWeighted(enhanced, 1.30, broad, -0.30, 0)

        return enhanced, {
            "preprocessing_test": "v3_test1_background_corrected_edge_preserving",
            "focus_laplacian_variance_preprocessing": focus_before,
            "background_sigma_px": float(sigma_bg),
            "background_correction": "large_gaussian_subtraction",
            "texture_suppression": "bilateral_filter",
            "bilateral_d": 9,
            "bilateral_sigma_color": 32,
            "bilateral_sigma_space": 11,
            "clahe_clip_limit_effective": min(float(self.config.clahe_clip_limit), 1.6),
            "clahe_tile_size_effective": 12,
            "unsharp_sigma_px": 2.2,
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

    def _fill_external_contours(self, binary: np.ndarray) -> np.ndarray:
        """Fill external contours to convert edge fragments into solid candidate regions."""
        out = np.zeros_like(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
        return out

    def _edge_envelope_candidate(self, enhanced: np.ndarray) -> Optional[Dict[str, Any]]:
        """Build a coarse anatomical envelope from edge/texture density.

        Brightfield caudal fins are often translucent, so direct intensity thresholding
        can fragment the fin into rays and internal structures. This method detects the
        textured/edged region, joins those structures, fills the envelope, and then
        scores large left-connected candidates (the fish enters the field from the left
        in the intended acquisition layout).
        """
        h, w = enhanced.shape
        image_area = float(h * w)

        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        finite = mag[np.isfinite(mag)]
        if finite.size == 0:
            return None
        thresh = float(np.percentile(finite, 88.0))
        edges = np.where(mag >= thresh, 255, 0).astype(np.uint8)

        # Suppress isolated speckles and bridge fin rays/outline into a region.
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        join = max(9, int(round(min(h, w) * 0.010)))
        if join % 2 == 0:
            join += 1
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join, join)),
            iterations=1,
        )
        close_k = max(21, int(round(min(h, w) * 0.025)))
        if close_k % 2 == 0:
            close_k += 1
        envelope = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=2,
        )
        envelope = self._fill_external_contours(envelope)

        # Gentle cleanup after filling.
        cleanup = max(7, int(round(min(h, w) * 0.006)))
        if cleanup % 2 == 0:
            cleanup += 1
        envelope = cv2.morphologyEx(
            envelope,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cleanup, cleanup)),
            iterations=1,
        )

        contours, _ = cv2.findContours(envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area <= 0:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            area_fraction = area / image_area
            width_fraction = bw / float(w)
            height_fraction = bh / float(h)
            left_gap_fraction = x / float(w)
            touches_left_zone = x <= 0.12 * w

            # Reject the exact failure seen in the first real run: tiny border fragments.
            if area_fraction < 0.008:
                continue
            if width_fraction < 0.15 or height_fraction < 0.10:
                continue
            if area_fraction > 0.55:
                continue

            solidity = self._contour_solidity(c)
            size_score = float(np.exp(-((area_fraction - 0.12) / 0.14) ** 2))
            span_score = float(np.clip((width_fraction / 0.45 + height_fraction / 0.35) / 2, 0, 1))
            left_score = 1.0 if touches_left_zone else float(np.clip(1.0 - left_gap_fraction / 0.35, 0, 1))
            solidity_score = float(np.clip((solidity - 0.35) / 0.55, 0, 1))
            score = 0.35 * size_score + 0.30 * span_score + 0.25 * left_score + 0.10 * solidity_score

            candidates.append((score, c, area_fraction, width_fraction, height_fraction, x, y, bw, bh))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, contour, af, wf, hf, x, y, bw, bh = candidates[0]
        return {
            "contour": contour,
            "binary": envelope,
            "method": "edge_envelope",
            "threshold": None,
            "polarity": "structure",
            "contours_total": len(contours),
            "contours_viable": len(candidates),
            "ambiguity": len(candidates) > 1 and candidates[1][0] >= 0.90 * score,
            "candidate_score": float(score),
            "area_fraction": float(af),
            "bbox_width_fraction": float(wf),
            "bbox_height_fraction": float(hf),
            "bbox_x_fraction": float(x / w),
        }

    def _candidate_anatomy_score(self, contour: np.ndarray, enhanced: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Score a contour by anatomical footprint, not only solidity."""
        h, w = enhanced.shape
        image_area = float(h * w)
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        af = area / image_area
        wf = bw / float(w)
        hf = bh / float(h)
        xfrac = x / float(w)
        solidity = self._contour_solidity(contour)

        # These broad bounds are deliberately permissive, but exclude specks and
        # tiny edge fragments that can never represent the caudal region.
        if af < 0.008 or wf < 0.15 or hf < 0.10 or af > 0.55:
            return 0.0, {
                "area_fraction": af, "bbox_width_fraction": wf,
                "bbox_height_fraction": hf, "bbox_x_fraction": xfrac,
            }

        size_score = float(np.exp(-((af - 0.12) / 0.16) ** 2))
        span_score = float(np.clip((wf / 0.45 + hf / 0.35) / 2, 0, 1))
        left_score = float(np.clip(1.0 - xfrac / 0.40, 0, 1))
        solidity_score = float(np.clip((solidity - 0.30) / 0.60, 0, 1))
        score = 0.35 * size_score + 0.30 * span_score + 0.25 * left_score + 0.10 * solidity_score
        return float(score), {
            "area_fraction": af, "bbox_width_fraction": wf,
            "bbox_height_fraction": hf, "bbox_x_fraction": xfrac,
        }

    def _trace_smooth_boundary_path(
        self,
        edge_strength: np.ndarray,
        center_y: float,
        x0: int,
        x1: int,
        side: str,
        expected_half_height: float,
    ) -> Optional[np.ndarray]:
        """Trace a weak-but-smooth outer boundary using dynamic programming.

        The path score combines local edge evidence with an anatomical prior that
        favors the outer fin margin at a plausible distance from the centerline.
        Smoothness strongly penalizes sudden inward jumps onto dark internal rays.
        """
        h, w = edge_strength.shape
        ds = 4
        small = cv2.resize(edge_strength, (max(2, w // ds), max(2, h // ds)), interpolation=cv2.INTER_AREA)
        hs, ws = small.shape
        cx0 = max(0, min(ws - 2, int(round(x0 / ds))))
        cx1 = max(cx0 + 2, min(ws - 1, int(round(x1 / ds))))
        cy = center_y / ds
        eh = expected_half_height / ds

        if side == "upper":
            y_min = max(1, int(round(cy - 1.9 * eh)))
            y_max = max(y_min + 3, int(round(cy - 0.45 * eh)))
        else:
            y_min = min(hs - 4, int(round(cy + 0.45 * eh)))
            y_max = min(hs - 2, int(round(cy + 1.9 * eh)))
        if y_max <= y_min + 2:
            return None

        ys = np.arange(y_min, y_max + 1, dtype=np.int32)
        n_y = len(ys)
        n_x = cx1 - cx0 + 1
        local = np.empty((n_x, n_y), dtype=np.float32)

        # Robustly normalize edge evidence within the search band.
        band = small[y_min:y_max + 1, cx0:cx1 + 1]
        lo, hi = np.percentile(band, [15.0, 99.0])
        norm = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)

        # Anatomical radial prior: weak enough to follow real edge evidence, but
        # strong enough to keep the path outside the darker inner rays.
        for i, x in enumerate(range(cx0, cx1 + 1)):
            dist = np.abs(ys.astype(np.float32) - cy)
            radial = np.exp(-0.5 * ((dist - eh) / max(0.45 * eh, 2.0)) ** 2)
            outward = np.clip(dist / max(1.5 * eh, 1.0), 0, 1)
            edge = norm[ys, x]
            local[i] = -(1.00 * edge + 0.42 * radial + 0.18 * outward)

        inf = np.float32(1e9)
        dp = np.full((n_x, n_y), inf, dtype=np.float32)
        prev = np.full((n_x, n_y), -1, dtype=np.int16)
        dp[0] = local[0]
        max_jump = max(3, int(round(0.055 * eh + 4)))
        smooth_lambda = 0.055

        for i in range(1, n_x):
            for j in range(n_y):
                a = max(0, j - max_jump)
                b = min(n_y, j + max_jump + 1)
                idx = np.arange(a, b)
                dy = (ys[idx] - ys[j]).astype(np.float32)
                transition = dp[i - 1, a:b] + smooth_lambda * (dy * dy)
                k_rel = int(np.argmin(transition))
                k = a + k_rel
                dp[i, j] = local[i, j] + transition[k_rel]
                prev[i, j] = k

        j = int(np.argmin(dp[-1]))
        path_y = np.empty(n_x, dtype=np.int32)
        path_y[-1] = ys[j]
        for i in range(n_x - 1, 0, -1):
            j = int(prev[i, j])
            if j < 0:
                return None
            path_y[i - 1] = ys[j]

        xs_full = np.arange(cx0, cx1 + 1, dtype=np.float32) * ds
        ys_full = path_y.astype(np.float32) * ds
        # Smooth only after path optimization so weak gaps bridge naturally.
        k = max(9, int(round(len(ys_full) * 0.035)))
        if k % 2 == 0:
            k += 1
        if k < len(ys_full):
            ys_full = cv2.GaussianBlur(ys_full.reshape(1, -1), (k, 1), 0).ravel()
        return np.column_stack([xs_full, ys_full]).astype(np.int32)

    def _boundary_continuation_candidate(self, enhanced: np.ndarray) -> Optional[Dict[str, Any]]:
        """Test 2: infer the outer fin envelope from two smooth boundary paths.

        Uses the coarse edge envelope only to localize the tail. Final upper/lower
        paths are optimized independently with weak edge evidence + smoothness +
        an outer-distance prior, specifically to avoid snapping to dark inner rays.
        """
        h, w = enhanced.shape
        seed = self._edge_envelope_candidate(enhanced)
        if seed is None:
            return None
        sx, sy, sbw, sbh = cv2.boundingRect(seed["contour"])
        center_y = sy + 0.5 * sbh
        expected_half = max(0.155 * h, 1.20 * sbh)
        x0 = max(0, int(sx - 0.03 * w))
        x1 = min(w - 1, int(sx + max(sbw * 1.18, 0.38 * w)))

        # Edge evidence intentionally keeps weak boundaries. We use vertical gradient
        # more heavily because the upper/lower fin margins are predominantly horizontal
        # or gently curved in this acquisition geometry.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        edge = gy + 0.30 * gx

        upper = self._trace_smooth_boundary_path(edge, center_y, x0, x1, "upper", expected_half)
        lower = self._trace_smooth_boundary_path(edge, center_y, x0, x1, "lower", expected_half)
        if upper is None or lower is None or len(upper) < 10 or len(lower) < 10:
            return None

        # Enforce upper/lower ordering and reject implausibly narrow envelopes.
        n = min(len(upper), len(lower))
        upper = upper[:n]
        lower = lower[:n]
        sep = lower[:, 1] - upper[:, 1]
        if np.median(sep) < 0.12 * h or np.mean(sep > 0.08 * h) < 0.85:
            return None

        polygon = np.vstack([upper, lower[::-1]])
        contour = polygon.reshape(-1, 1, 2).astype(np.int32)
        area = float(cv2.contourArea(contour))
        image_area = float(h * w)
        if area < 0.02 * image_area or area > 0.45 * image_area:
            return None

        binary = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary, [contour], 255)
        x, y, bw, bh = cv2.boundingRect(contour)
        af = area / image_area
        wf = bw / float(w)
        hf = bh / float(h)

        # Boundary evidence sampled along paths. Because weak visible segments are
        # allowed, confidence also incorporates smoothness and anatomical separation.
        edge_norm = edge / max(float(np.percentile(edge, 99.5)), 1e-6)
        edge_norm = np.clip(edge_norm, 0, 1)
        up_e = edge_norm[np.clip(upper[:,1],0,h-1), np.clip(upper[:,0],0,w-1)]
        lo_e = edge_norm[np.clip(lower[:,1],0,h-1), np.clip(lower[:,0],0,w-1)]
        evidence = float(np.mean(np.r_[up_e, lo_e]))
        roughness = float(np.mean(np.abs(np.diff(upper[:,1], n=2)))) + float(np.mean(np.abs(np.diff(lower[:,1], n=2))))
        smooth_score = float(np.exp(-roughness / 7.0))
        sep_score = float(np.clip(np.median(sep) / (0.34 * h), 0, 1))
        score = 0.45 * evidence + 0.30 * smooth_score + 0.25 * sep_score

        return {
            "contour": contour,
            "binary": binary,
            "method": "boundary_continuation_test2",
            "threshold": None,
            "polarity": "outer_paths",
            "contours_total": 1,
            "contours_viable": 1,
            "ambiguity": False,
            "candidate_score": float(score),
            "area_fraction": float(af),
            "bbox_width_fraction": float(wf),
            "bbox_height_fraction": float(hf),
            "bbox_x_fraction": float(x / w),
            "upper_path": upper,
            "lower_path": lower,
            "boundary_edge_evidence": evidence,
            "boundary_smoothness_score": smooth_score,
            "boundary_separation_score": sep_score,
            "center_y_estimate": float(center_y),
            "expected_half_height": float(expected_half),
        }

    def segment_tail(self, enhanced: np.ndarray) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        # Test 2 preferred method: smooth outer-boundary continuation.
        continuation = self._boundary_continuation_candidate(enhanced)
        if continuation is not None:
            attempts.append({
                "method": "boundary_continuation_test2",
                "threshold": None,
                "polarity": "outer_paths",
                "contours_total": 1,
                "contours_viable": 1,
                "min_area_used": "smooth anatomical envelope",
            })
            best = continuation

        # 1) Anatomy-aware edge envelope: fallback for translucent brightfield fins.
        edge_candidate = self._edge_envelope_candidate(enhanced)
        if edge_candidate is not None:
            attempts.append({
                "method": "edge_envelope",
                "threshold": None,
                "polarity": "structure",
                "contours_total": edge_candidate["contours_total"],
                "contours_viable": edge_candidate["contours_viable"],
                "min_area_used": "relative anatomical filter",
            })
            if best is None:
                best = edge_candidate

        # 2) Classic intensity thresholds remain as fallback/competitors, but every
        # contour must now pass the anatomical footprint test.
        for method, threshold, binary0 in self._threshold_candidates(enhanced):
            for polarity, raw_binary in (("normal", binary0), ("inverted", cv2.bitwise_not(binary0))):
                cleaned = self._clean_binary(raw_binary)
                contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                scored = []
                for c in contours:
                    score, anatomy = self._candidate_anatomy_score(c, enhanced)
                    if score > 0:
                        scored.append((score, c, anatomy))

                attempts.append({
                    "method": method,
                    "threshold": None if math.isnan(threshold) else threshold,
                    "polarity": polarity,
                    "contours_total": len(contours),
                    "contours_viable": len(scored),
                    "min_area_used": "relative anatomical filter",
                })
                if not scored:
                    continue
                scored.sort(key=lambda x: x[0], reverse=True)
                score, contour, anatomy = scored[0]
                candidate = {
                    "contour": contour,
                    "binary": cleaned,
                    "method": method,
                    "threshold": None if math.isnan(threshold) else threshold,
                    "polarity": polarity,
                    "contours_total": len(contours),
                    "contours_viable": len(scored),
                    "ambiguity": len(scored) > 1 and scored[1][0] >= 0.90 * score,
                    "candidate_score": float(score),
                    **anatomy,
                }
                if best is None:
                    best = candidate
                elif best.get("method") != "boundary_continuation_test2" and candidate["candidate_score"] > best["candidate_score"]:
                    best = candidate

        if best is None:
            raise ValueError(
                "Segmentation failed: no anatomically plausible tail region was found. "
                "The agent rejected small/fragmented contours rather than exporting a misleading ROI."
            )

        # Hard safety gate. Never export the tiny-artifact failure observed in the
        # first real CZI run even if it happens to score well on local contrast.
        if (
            best.get("area_fraction", 0.0) < 0.008
            or best.get("bbox_width_fraction", 0.0) < 0.15
            or best.get("bbox_height_fraction", 0.0) < 0.10
            or best.get("candidate_score", 0.0) < 0.35
        ):
            raise ValueError(
                "Segmentation failed anatomical safety checks; no ROI was exported. "
                "Please review the image or use a manual trace."
            )

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
            f"Segmentation: {best['method']} ({best['polarity']}), "
            f"area_fraction={best['area_fraction']:.3f}, "
            f"bbox=({best.get('bbox_width_fraction', 0):.2f}W x {best.get('bbox_height_fraction', 0):.2f}H), "
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
            metrics["width_micrometers"] = float(w) * px
            metrics["height_micrometers"] = float(h) * px
            metrics["centroid_x_micrometers"] = cx * px
            metrics["centroid_y_micrometers"] = cy * px
        return metrics


    @staticmethod
    def _edge_clipping_status(contour: np.ndarray, shape: Tuple[int, int], margin_px: int = 3) -> Dict[str, Any]:
        """Report whether the selected ROI intersects the image boundary.

        A contour touching an image edge may still be a correct segmentation, but
        morphometrics then describe only the visible portion of the anatomy.
        """
        h, w = shape
        pts = contour.reshape(-1, 2)
        left = bool(np.any(pts[:, 0] <= margin_px))
        right = bool(np.any(pts[:, 0] >= (w - 1 - margin_px)))
        top = bool(np.any(pts[:, 1] <= margin_px))
        bottom = bool(np.any(pts[:, 1] >= (h - 1 - margin_px)))
        edges = [name for name, hit in (
            ("left", left), ("right", right), ("top", top), ("bottom", bottom)
        ) if hit]
        return {
            "edge_clipped": bool(edges),
            "clipped_edges": edges,
            "edge_margin_px": int(margin_px),
            "measurement_scope": "visible_portion_only" if edges else "complete_within_field_of_view",
        }

    @staticmethod
    def _boundary_evidence_score(original: np.ndarray, contour: np.ndarray) -> float:
        """Estimate independent image support along the model boundary.

        This is deliberately a modest QA term, not the segmentation engine: the
        translucent fin edge can be faint, so weak local contrast must not force the
        contour inward toward dark internal structures.
        """
        img = original.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        boundary = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.drawContours(boundary, [contour.astype(np.int32)], -1, 255, thickness=5)
        vals = mag[boundary > 0]
        if vals.size == 0:
            return 0.0

        global_ref = float(np.percentile(mag, 70))
        if global_ref <= 1e-6:
            return 0.5

        # A score near 1 means the proposed boundary has at least as much edge
        # evidence as a typical strong-ish image location. Cap to avoid overrewarding
        # dark internal structures.
        ratio = float(np.median(vals) / global_ref)
        return float(np.clip(0.25 + 0.60 * ratio, 0.0, 1.0))


    def _estimate_notochord_endpoint(
        self,
        original: np.ndarray,
        outer_mask: np.ndarray,
        outer_contour: np.ndarray,
    ) -> Dict[str, Any]:
        """Reference-informed notochord tip detector.

        The 9 researcher ROIs establish a strong geometric prior for where the
        notochord tip sits relative to the distal tail edge. We then search around
        that expected location for two image cues:
        1. the termination of the parallel notochord structure;
        2. the BEGINNING/proximal edge of the dark distal stain.

        This intentionally avoids the earlier failure mode of following the stain
        all the way to its distal end.
        """
        img = self._normalize_to_uint8(original)
        h, w = img.shape[:2]
        mask = outer_mask.astype(bool)
        x, y, bw, bh = cv2.boundingRect(outer_contour.astype(np.int32))

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        bw = int(max(2, min(bw, w - x)))
        bh = int(max(2, min(bh, h - y)))

        y0 = int(np.clip(y + 0.18 * bh, 0, h - 1))
        y1 = int(np.clip(y + 0.82 * bh, y0 + 1, h))

        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 12)).apply(img)
        blur = cv2.GaussianBlur(clahe, (0, 0), 1.3)

        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        horizontal_evidence = np.maximum(np.abs(gy) - 0.65 * np.abs(gx), 0.0)

        roi_evidence = horizontal_evidence[y0:y1, x:x + bw]
        profile = np.percentile(roi_evidence, 88, axis=0).astype(np.float32)
        profile = cv2.GaussianBlur(
            profile.reshape(1, -1), (0, 0), sigmaX=max(1.5, 0.012 * bw)
        ).reshape(-1)

        side_n = max(8, int(round(0.28 * bw)))
        side_n = min(side_n, max(1, bw // 2))
        left_score = float(np.median(profile[:side_n]))
        right_score = float(np.median(profile[-side_n:]))

        edge_margin = max(3, int(round(0.005 * w)))
        touches_left = x <= edge_margin
        touches_right = (x + bw - 1) >= (w - 1 - edge_margin)

        if touches_left and not touches_right:
            proximal_side = "left"
            orientation_conf = 0.92
        elif touches_right and not touches_left:
            proximal_side = "right"
            orientation_conf = 0.92
        else:
            proximal_side = "left" if left_score >= right_score else "right"
            orientation_conf = float(
                np.clip(abs(left_score - right_score) / max(left_score, right_score, 1e-6), 0, 1)
            )

        # ---- researcher-derived geometric prior ----
        tip_distal_ratio = 0.3431952662721893
        if proximal_side == "left":
            distal_x = x + bw - 1
            prior_tip_x = int(round(distal_x - tip_distal_ratio * bh))
        else:
            distal_x = x
            prior_tip_x = int(round(distal_x + tip_distal_ratio * bh))
        prior_tip_x = int(np.clip(prior_tip_x, x + 1, x + bw - 2))

        # ---- fit approximate notochord centerline ----
        sample_lo = x + max(2, int(0.05 * bw))
        sample_hi = x + bw - max(2, int(0.10 * bw))
        fit_x, fit_y = [], []
        yy = np.arange(y0, y1, dtype=np.float32)

        for xx in range(sample_lo, sample_hi, max(1, bw // 140)):
            vals = horizontal_evidence[y0:y1, xx].astype(np.float32)
            if vals.size == 0 or float(vals.max()) <= 1e-6:
                continue
            thr = float(np.percentile(vals, 87))
            weights = np.where(vals >= thr, vals, 0.0)
            sw = float(weights.sum())
            if sw <= 1e-6:
                continue
            yc = float(np.sum(yy * weights) / sw)
            fit_x.append(float(xx))
            fit_y.append(yc)

        if len(fit_x) >= 8:
            coef = np.polyfit(np.asarray(fit_x), np.asarray(fit_y), 1)
            pred = np.polyval(coef, np.asarray(fit_x))
            resid = np.asarray(fit_y) - pred
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med))) + 1e-6
            keep = np.abs(resid - med) <= max(4.0, 3.5 * mad)
            if int(np.sum(keep)) >= 6:
                coef = np.polyfit(np.asarray(fit_x)[keep], np.asarray(fit_y)[keep], 1)
        else:
            coef = np.array([0.0, y + 0.53 * bh], dtype=float)

        # ---- dark-stain onset cue ----
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(15, 2 * (bw // 30) + 1), max(9, 2 * (bh // 60) + 1))
        )
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)
        dark = 0.55 * (255.0 - blur.astype(np.float32)) + 0.45 * blackhat.astype(np.float32)

        band_half = max(8, int(round(0.055 * bh)))
        stain_profile = np.zeros(bw, dtype=np.float32)
        for i in range(bw):
            xx = x + i
            yc = float(np.polyval(coef, xx))
            ys0 = int(max(y0, round(yc - band_half)))
            ys1 = int(min(y1, round(yc + band_half)))
            if ys1 <= ys0:
                continue
            stain_profile[i] = float(np.percentile(dark[ys0:ys1, xx], 88))

        stain_profile = cv2.GaussianBlur(
            stain_profile.reshape(1, -1), (0, 0), sigmaX=max(1.5, 0.009 * bw)
        ).reshape(-1)

        if proximal_side == "left":
            directional_stain = stain_profile
            directional_line = profile
            prior_i = prior_tip_x - x
        else:
            directional_stain = stain_profile[::-1]
            directional_line = profile[::-1]
            prior_i = (x + bw - 1) - prior_tip_x

        prior_i = int(np.clip(prior_i, 2, bw - 3))
        search_radius = max(18, int(round(0.12 * bh)))
        lo = max(2, prior_i - search_radius)
        hi = min(bw - 2, prior_i + search_radius)

        d_stain = np.gradient(directional_stain)
        d_line = np.gradient(directional_line)

        stain_scale = max(
            float(np.percentile(np.abs(d_stain[lo:hi]), 90)) if hi > lo else 0.0,
            1e-6,
        )
        line_scale = max(
            float(np.percentile(np.abs(d_line[lo:hi]), 90)) if hi > lo else 0.0,
            1e-6,
        )

        best_i = prior_i
        best_score = -1.0
        best_stain = 0.0
        best_line = 0.0
        best_prior = 0.0

        for i in range(lo, hi):
            # Positive dark-stain derivative = proximal edge / stain beginning.
            stain_onset = float(np.clip(d_stain[i] / stain_scale, 0.0, 1.0))
            # Negative horizontal-evidence derivative = notochord termination.
            line_term = float(np.clip((-d_line[i]) / line_scale, 0.0, 1.0))
            prior_score = float(
                np.exp(-0.5 * ((i - prior_i) / max(1.0, 0.055 * bh)) ** 2)
            )
            score = 0.44 * stain_onset + 0.26 * line_term + 0.30 * prior_score
            if score > best_score:
                best_score = score
                best_i = i
                best_stain = stain_onset
                best_line = line_term
                best_prior = prior_score

        if proximal_side == "left":
            endpoint_x = int(x + best_i)
        else:
            endpoint_x = int(x + (bw - 1 - best_i))

        endpoint_y = float(np.polyval(coef, endpoint_x))
        endpoint_y = float(np.clip(endpoint_y, y0, y1 - 1))

        # Vertical intersection span from the actual outer fin mask.
        ys_here = np.flatnonzero(mask[:, endpoint_x])
        if len(ys_here):
            cutoff_top = int(ys_here.min())
            cutoff_bottom = int(ys_here.max())
        else:
            cutoff_top = y
            cutoff_bottom = y + bh - 1

        geometry_error = abs(endpoint_x - prior_tip_x) / max(float(bh), 1.0)
        geometry_agreement = float(np.clip(1.0 - geometry_error / 0.13, 0.0, 1.0))

        confidence = float(np.clip(
            0.34 * best_stain
            + 0.24 * best_line
            + 0.24 * geometry_agreement
            + 0.10 * best_prior
            + 0.08 * orientation_conf,
            0.0, 1.0
        ))

        proximal_x = float(x if proximal_side == "left" else x + bw - 1)
        proximal_y = float(np.polyval(coef, proximal_x))

        return {
            "endpoint_x": int(endpoint_x),
            "endpoint_y": float(endpoint_y),
            "proximal_side": proximal_side,
            "cutoff_orientation": "researcher_style_notch",
            "cutoff_x": int(endpoint_x),
            "cutoff_y_top": int(cutoff_top),
            "cutoff_y_bottom": int(cutoff_bottom),
            "confidence": confidence,
            "method": "reference_prior_plus_stain_onset",
            "axis_slope_pixels_per_pixel": float(coef[0]),
            "trace_points": [
                [float(proximal_x), float(proximal_y)],
                [float(endpoint_x), float(endpoint_y)],
            ],
            "reference_prior_tip_x": int(prior_tip_x),
            "stain_onset_component": float(best_stain),
            "line_termination_component": float(best_line),
            "geometry_agreement": float(geometry_agreement),
            "orientation_confidence": float(orientation_conf),
            "reference_tip_to_distal_ratio": tip_distal_ratio,
            "review_required": bool(confidence < 0.62),
        }

    def _apply_notochord_cutoff(
        self,
        outer_mask: np.ndarray,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct the researcher-style ROI while preserving the true tail edge.

        The manual ROIs consistently contain:
        - the actual distal/upper/lower tail boundary;
        - a proximal vertical boundary;
        - an inward central notch around the body/notochord;
        - convergence of that notch at the notochord tip.

        MobileSAM supplies the outer tail boundary. The researcher annotations are
        used only to constrain the proximal/notch geometry.
        """
        mask = outer_mask.astype(bool)
        h, w = mask.shape[:2]
        ys_all, xs_all = np.where(mask)
        if len(xs_all) == 0:
            raise RuntimeError("Outer fin mask is empty")

        outer_xmin, outer_xmax = int(xs_all.min()), int(xs_all.max())
        outer_ymin, outer_ymax = int(ys_all.min()), int(ys_all.max())
        roi_h = max(1, outer_ymax - outer_ymin + 1)

        tip_x = int(np.clip(detection["endpoint_x"], 0, w - 1))
        tip_y = float(np.clip(detection["endpoint_y"], 0, h - 1))
        proximal_side = detection.get("proximal_side", "left")

        notch_opening = 0.23339960238568588 * roi_h
        tip_to_distal = 0.3431952662721893 * roi_h
        roi_aspect = 0.8475499092558983
        expected_roi_width = roi_aspect * roi_h

        if proximal_side == "left":
            distal_x = outer_xmax
            prox_by_width = int(round(distal_x - expected_roi_width))
            prox_by_tip = int(round(tip_x - (expected_roi_width - tip_to_distal)))
            proximal_x = int(round(0.45 * prox_by_width + 0.55 * prox_by_tip))
            proximal_x = int(np.clip(proximal_x, outer_xmin, tip_x - 2))
        else:
            distal_x = outer_xmin
            prox_by_width = int(round(distal_x + expected_roi_width))
            prox_by_tip = int(round(tip_x + (expected_roi_width - tip_to_distal)))
            proximal_x = int(round(0.45 * prox_by_width + 0.55 * prox_by_tip))
            proximal_x = int(np.clip(proximal_x, tip_x + 2, outer_xmax))

        # Retain only the broad distal tail region.
        keep = np.zeros_like(mask, dtype=bool)
        if proximal_side == "left":
            keep[:, proximal_x:] = True
        else:
            keep[:, :proximal_x + 1] = True
        final_mask = mask & keep

        # Create the central inward notch seen in the researcher ROIs.
        corridor = np.zeros_like(mask, dtype=np.uint8)
        half_open = 0.5 * notch_opening

        if proximal_side == "left":
            xs = np.arange(proximal_x, tip_x + 1)
            denom = max(tip_x - proximal_x, 1)
            for xx in xs:
                t = (xx - proximal_x) / denom
                # Manual ROIs taper smoothly toward the notochord tip.
                taper = (1.0 - t) ** 0.72
                half = half_open * taper
                y0 = int(max(0, round(tip_y - half)))
                y1 = int(min(h - 1, round(tip_y + half)))
                corridor[y0:y1 + 1, xx] = 1
        else:
            xs = np.arange(tip_x, proximal_x + 1)
            denom = max(proximal_x - tip_x, 1)
            for xx in xs:
                t = (proximal_x - xx) / denom
                taper = (1.0 - t) ** 0.72
                half = half_open * taper
                y0 = int(max(0, round(tip_y - half)))
                y1 = int(min(h - 1, round(tip_y + half)))
                corridor[y0:y1 + 1, xx] = 1

        final_mask = final_mask & (~corridor.astype(bool))
        final_u8 = final_mask.astype(np.uint8) * 255

        # Keep morphology deliberately mild so the MobileSAM outer tail boundary is
        # not rounded away or replaced by a synthetic shape.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_u8 = cv2.morphologyEx(final_u8, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(final_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise RuntimeError("Reference-informed ROI construction produced no contour")

        contour = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(final_u8)
        cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

        outer_area = max(float(np.count_nonzero(outer_mask)), 1.0)
        measured_area = float(np.count_nonzero(clean))

        _, _, fw, fh = cv2.boundingRect(contour)
        final_aspect = float(fw / max(fh, 1))
        shape_prior_score = float(np.clip(
            1.0 - abs(final_aspect - 0.84755) / 0.30,
            0.0, 1.0
        ))

        detection["researcher_proximal_boundary_x"] = int(proximal_x)
        detection["reference_notch_opening_px"] = float(notch_opening)
        detection["reference_expected_roi_width_px"] = float(expected_roi_width)
        detection["reference_shape_prior_score"] = shape_prior_score

        return {
            "binary": clean,
            "contour": contour,
            "retained_fraction_of_outer_mask": measured_area / outer_area,
            "researcher_proximal_boundary_x": int(proximal_x),
            "shape_prior_score": shape_prior_score,
        }

    def _save_notochord_overlay(
        self,
        original: np.ndarray,
        outer_contour: np.ndarray,
        final_contour: np.ndarray,
        detection: Dict[str, Any],
        out_path: Path,
    ) -> str:
        """Research-facing overlay: outer fin, traced notochord, endpoint and cutoff."""
        img = self._normalize_to_uint8(original)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Outer SAM envelope in a thin blue line.
        cv2.drawContours(rgb, [outer_contour.astype(np.int32)], -1, (80, 150, 255), 2)

        # Final measured ROI in green.
        cv2.drawContours(rgb, [final_contour.astype(np.int32)], -1, (0, 255, 0), 4)

        pts = detection.get("trace_points") or []
        if len(pts) >= 2:
            p0 = tuple(int(round(v)) for v in pts[0])
            p1 = tuple(int(round(v)) for v in pts[-1])
            cv2.line(rgb, p0, p1, (0, 220, 255), 3)

        cx = int(detection.get("cutoff_x", detection.get("endpoint_x", 0)))
        yt = int(detection.get("cutoff_y_top", 0))
        yb = int(detection.get("cutoff_y_bottom", img.shape[0] - 1))
        cv2.line(rgb, (cx, yt), (cx, yb), (255, 220, 0), 4)

        ey = int(round(float(detection.get("endpoint_y", (yt + yb) / 2.0))))
        cv2.circle(rgb, (cx, ey), 8, (255, 80, 80), thickness=-1)

        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return str(out_path)

    def _save_notochord_diagnostic(
        self,
        original: np.ndarray,
        outer_mask: np.ndarray,
        final_mask: np.ndarray,
        outer_contour: np.ndarray,
        final_contour: np.ndarray,
        detection: Dict[str, Any],
        out_path: Path,
        filename: str,
    ) -> str:
        img = self._normalize_to_uint8(original)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        left = rgb.copy()
        cv2.drawContours(left, [outer_contour.astype(np.int32)], -1, (80, 150, 255), 2)
        pts = detection.get("trace_points") or []
        if len(pts) >= 2:
            p0 = tuple(int(round(v)) for v in pts[0])
            p1 = tuple(int(round(v)) for v in pts[-1])
            cv2.line(left, p0, p1, (0, 220, 255), 3)
        cx = int(detection.get("cutoff_x", 0))
        yt = int(detection.get("cutoff_y_top", 0))
        yb = int(detection.get("cutoff_y_bottom", img.shape[0] - 1))
        ey = int(round(float(detection.get("endpoint_y", (yt + yb) / 2.0))))
        cv2.line(left, (cx, yt), (cx, yb), (255, 220, 0), 4)
        cv2.circle(left, (cx, ey), 8, (255, 80, 80), -1)

        right = rgb.copy()
        final_bool = final_mask.astype(bool)
        tint = right.copy()
        tint[final_bool] = np.array([0, 255, 0], dtype=np.uint8)
        right = cv2.addWeighted(right, 0.72, tint, 0.28, 0)
        cv2.drawContours(right, [final_contour.astype(np.int32)], -1, (0, 255, 0), 4)
        cv2.line(right, (cx, yt), (cx, yb), (255, 220, 0), 4)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(left)
        axes[0].set_title("Notochord endpoint + vertical cutoff")
        axes[1].imshow(right)
        axes[1].set_title("Final distal-fin measurement region")
        for ax in axes:
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

        conf = float(detection.get("confidence", 0.0))
        method = detection.get("method", "unknown")
        status = "REVIEW" if detection.get("review_required") else "PASS"
        fig.suptitle(
            f"{filename} - Notochord-defined Fin ROI\n"
            f"{status} | confidence={100*conf:.0f}% | method={method}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def assess_quality(
        self, enhanced: np.ndarray, metrics: Dict[str, Any], segmentation: Dict[str, Any]
    ) -> Dict[str, Any]:
        lap_var = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
        focus_score = float(np.clip(100 * lap_var / 50.0, 0, 100))

        candidate_score = float(np.clip(segmentation.get("candidate_score", 0.0), 0, 1))

        # AI-segmentation confidence should be judged on the OUTER fin envelope,
        # not penalized simply because the scientifically defined ROI is cropped
        # at the notochord endpoint.
        seg_af = float(segmentation.get("outer_area_fraction", segmentation.get("area_fraction", 0.0)))
        seg_wf = float(segmentation.get("outer_bbox_width_fraction", segmentation.get("bbox_width_fraction", 0.0)))
        seg_hf = float(segmentation.get("outer_bbox_height_fraction", segmentation.get("bbox_height_fraction", 0.0)))
        footprint_score = float(np.clip(min(seg_af / 0.04, seg_wf / 0.30, seg_hf / 0.22), 0, 1))

        af = float(segmentation.get("area_fraction", 0.0))
        wf = float(segmentation.get("bbox_width_fraction", 0.0))
        hf = float(segmentation.get("bbox_height_fraction", 0.0))

        solidity_component = float(np.clip(metrics["solidity"], 0, 1))
        boundary_evidence = float(np.clip(segmentation.get("boundary_evidence_score", 0.5), 0, 1))

        segmentation_confidence = float(np.clip(
            0.35 * candidate_score
            + 0.25 * footprint_score
            + 0.15 * solidity_component
            + 0.25 * boundary_evidence,
            0, 1
        ))
        if segmentation.get("ambiguity"):
            segmentation_confidence *= 0.85

        notochord = segmentation.get("notochord") or {}
        notochord_confidence = float(np.clip(notochord.get("confidence", 0.0), 0, 1))
        notochord_review = bool(notochord.get("review_required", False))
        retained = float(segmentation.get("retained_fraction_of_outer_mask", 1.0))
        shape_prior_score = float(np.clip(segmentation.get("reference_shape_prior_score", 1.0), 0, 1))

        plausibility_failures = []
        if not 0.12 <= metrics["aspect_ratio"] <= 7.0:
            plausibility_failures.append("aspect_ratio")
        if not 0.03 <= metrics["circularity"] <= 0.98:
            plausibility_failures.append("circularity")
        if not 0.30 <= metrics["solidity"] <= 0.999:
            plausibility_failures.append("solidity")
        if af < 0.003:
            plausibility_failures.append("area_fraction_too_small")
        if wf < 0.06:
            plausibility_failures.append("bbox_width_too_small")
        if hf < 0.08:
            plausibility_failures.append("bbox_height_too_small")
        if segmentation.get("model_source") == "mobilesam" and not 0.10 <= retained <= 0.90:
            plausibility_failures.append("notochord_cutoff_retained_fraction")

        if not plausibility_failures:
            plausibility = "pass"
        elif len(plausibility_failures) == 1:
            plausibility = "warning"
        else:
            plausibility = "fail"

        clipping = segmentation.get("edge_clipping", {})
        edge_clipped = bool(clipping.get("edge_clipped", False))

        quality_score = int(round(np.clip(
            (
                0.15 * (focus_score / 100.0)
                + 0.50 * segmentation_confidence
                + 0.22 * notochord_confidence
                + 0.13 * shape_prior_score
            ) * 100,
            0, 100
        )))
        if segmentation_confidence < 0.55:
            quality_score = min(quality_score, 49)

        model_selected = segmentation.get("model_source") == "mobilesam"

        if segmentation_confidence < 0.65 or plausibility == "fail":
            flag = "poor"
            recommendation = "SEGMENTATION FAILED - DO NOT USE ROI or morphometrics"
        elif notochord_review or notochord_confidence < 0.60:
            flag = "review"
            recommendation = (
                "Outer fin segmentation is usable, but the notochord endpoint is uncertain. "
                "Inspect the yellow vertical cutoff before using the measurements."
            )
        elif edge_clipped:
            flag = "review"
            edges = ", ".join(clipping.get("clipped_edges", []))
            recommendation = (
                f"Distal-fin ROI intersects the image edge ({edges}). "
                "Measurements describe only the visible portion; validate in Fiji."
            )
        elif (
            model_selected
            and segmentation_confidence >= 0.78
            and notochord_confidence >= 0.65
            and plausibility == "pass"
            and quality_score >= 75
        ):
            flag = "good"
            recommendation = (
                "Outer fin and notochord cutoff passed automated QA. "
                "Proceed after routine visual validation in Fiji."
            )
        else:
            flag = "review"
            recommendation = (
                "Review the green distal-fin boundary and yellow notochord cutoff "
                "before using measurements."
            )

        return {
            "laplacian_variance": lap_var,
            "focus_score": int(round(focus_score)),
            "quality_score": quality_score,
            "segmentation_confidence": segmentation_confidence,
            "boundary_evidence_score": boundary_evidence,
            "notochord_confidence": notochord_confidence,
            "notochord_review_required": notochord_review,
            "reference_shape_prior_score": shape_prior_score,
            "anatomical_plausibility": plausibility,
            "plausibility_failures": plausibility_failures,
            "edge_clipping": clipping,
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


    def _save_original_preview(self, original: np.ndarray, out_path: Path) -> str:
        original_u8 = self._normalize_to_uint8(original)
        cv2.imwrite(str(out_path), original_u8)
        return str(out_path)

    def _save_boundary_overlay_preview(
        self,
        original: np.ndarray,
        contour: np.ndarray,
        out_path: Path,
        notochord: Optional[Dict[str, Any]] = None,
        outer_contour: Optional[np.ndarray] = None,
    ) -> str:
        original_u8 = self._normalize_to_uint8(original)
        rgb = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2RGB)

        if outer_contour is not None:
            cv2.drawContours(
                rgb, [outer_contour.astype(np.int32)], -1, (80, 150, 255), 2
            )

        cv2.drawContours(rgb, [contour.astype(np.int32)], -1, (0, 255, 0), 4)

        if notochord and notochord.get("cutoff_x") is not None:
            pts = notochord.get("trace_points") or []
            if len(pts) >= 2:
                p0 = tuple(int(round(v)) for v in pts[0])
                p1 = tuple(int(round(v)) for v in pts[-1])
                cv2.line(rgb, p0, p1, (0, 220, 255), 3)

            cx = int(notochord["cutoff_x"])
            yt = int(notochord.get("cutoff_y_top", 0))
            yb = int(notochord.get("cutoff_y_bottom", original_u8.shape[0] - 1))
            ey = int(round(float(notochord.get("endpoint_y", (yt + yb) / 2.0))))
            cv2.line(rgb, (cx, yt), (cx, yb), (255, 220, 0), 4)
            cv2.circle(rgb, (cx, ey), 8, (255, 80, 80), -1)

        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return str(out_path)

    @staticmethod
    def _save_raw_mask(mask: np.ndarray, out_path: Path) -> str:
        """Save the exact selected binary mask without contour smoothing."""
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.max() <= 1:
            mask_u8 = mask_u8 * 255
        cv2.imwrite(str(out_path), mask_u8)
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
            (enhanced, "Enhanced Brightfield", "gray"),
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
            f"Focus: {quality.get('focus_score', 'NA')}/100\n"
            f"Segmentation: {100*quality['segmentation_confidence']:.0f}%\n"
            f"Overall QA: {quality['quality_score']}/100\n"
            f"Boundary evidence: {100*quality.get('boundary_evidence_score', 0):.0f}%\n"
            f"Clipped: {quality.get('edge_clipping', {}).get('edge_clipped', False)}\n"
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


    def _expanded_prompt_box(self, contour: np.ndarray, shape: Tuple[int, int]) -> List[int]:
        """Create a generous SAM prompt box from the old rough localization.

        The classical contour is used only as a locator. The vertical expansion is
        deliberately large because the previous methods captured internal rays but
        under-estimated the translucent fin height.
        """
        h, w = shape
        x, y, bw, bh = cv2.boundingRect(contour)
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        target_w = max(bw * 1.25, w * 0.30)
        target_h = max(bh * 3.25, h * 0.32)

        x0 = int(max(0, cx - target_w * 0.58))
        x1 = int(min(w - 1, cx + target_w * 0.58))
        y0 = int(max(0, cy - target_h * 0.52))
        y1 = int(min(h - 1, cy + target_h * 0.52))

        # If the rough localization is near the left edge, preserve that connection.
        if x < 0.08 * w:
            x0 = 0

        return [x0, y0, x1, y1]

    @staticmethod
    def _mask_to_main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        mask_u8 = (mask.astype(np.uint8) * 255)
        # Close small gaps and fill the main object.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _score_model_contour(self, contour: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, Dict[str, float]]:
        h, w = shape
        image_area = float(h * w)
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = area / hull_area
        area_fraction = area / image_area
        wf = bw / float(w)
        hf = bh / float(h)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        # Broad fan-like masks should have meaningful width and height, but need not
        # be convex because fins are irregular.
        score = 0.0
        score += 0.28 * float(np.clip((area_fraction - 0.015) / 0.18, 0, 1))
        score += 0.25 * float(np.clip((wf - 0.18) / 0.40, 0, 1))
        score += 0.25 * float(np.clip((hf - 0.16) / 0.38, 0, 1))
        score += 0.12 * float(np.clip((solidity - 0.35) / 0.45, 0, 1))
        # Prefer masks that touch/approach the left side for this acquisition geometry.
        left_bonus = float(np.clip((0.18 - x / float(w)) / 0.18, 0, 1))
        score += 0.10 * left_bonus

        return float(score), {
            "area_fraction": float(area_fraction),
            "bbox_width_fraction": float(wf),
            "bbox_height_fraction": float(hf),
            "bbox_x_fraction": float(x / float(w)),
            "solidity": float(solidity),
            "circularity": float(circularity),
        }

    def _mobile_sam_candidate(
        self,
        original: np.ndarray,
        rough_segmentation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run MobileSAM using the classical result only as a generous box prompt."""
        global _MOBILE_SAM_MODEL

        try:
            from ultralytics import SAM
        except Exception as exc:
            return {
                "success": False,
                "error": f"Ultralytics/MobileSAM import failed: {exc}",
                "prompt_box": None,
                "candidates": [],
            }

        original_u8 = self._normalize_to_uint8(original)
        rgb = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2RGB)
        prompt_box = self._expanded_prompt_box(rough_segmentation["contour"], original_u8.shape)

        try:
            if _MOBILE_SAM_MODEL is None:
                # Ultralytics downloads/caches this lightweight checkpoint on first use.
                _MOBILE_SAM_MODEL = SAM("mobile_sam.pt")

            results = _MOBILE_SAM_MODEL.predict(
                source=rgb,
                bboxes=prompt_box,
                verbose=False,
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"MobileSAM inference failed: {exc}",
                "prompt_box": prompt_box,
                "candidates": [],
            }

        candidates = []
        try:
            result = results[0]
            if result.masks is None or result.masks.data is None:
                raise RuntimeError("MobileSAM returned no masks")

            masks = result.masks.data.detach().cpu().numpy()
            for i, mask in enumerate(masks):
                mask_bool = mask > 0.5
                contour = self._mask_to_main_contour(mask_bool)
                if contour is None or cv2.contourArea(contour) <= 0:
                    continue

                score, anatomy = self._score_model_contour(contour, original_u8.shape)
                candidate = {
                    "index": int(i),
                    "score": float(score),
                    "contour": contour,
                    "mask": mask_bool,
                    **anatomy,
                }
                candidates.append(candidate)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Could not parse MobileSAM masks: {exc}",
                "prompt_box": prompt_box,
                "candidates": [],
            }

        if not candidates:
            return {
                "success": False,
                "error": "MobileSAM produced no usable mask contours",
                "prompt_box": prompt_box,
                "candidates": [],
            }

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Conservative gate for this first model test: don't silently substitute a
        # model mask unless it is broad enough to plausibly represent the outer fin.
        success = (
            best["score"] >= 0.42
            and best["area_fraction"] >= 0.012
            and best["bbox_width_fraction"] >= 0.18
            and best["bbox_height_fraction"] >= 0.16
        )

        return {
            "success": bool(success),
            "error": None if success else "Best MobileSAM mask failed anatomical safety gate",
            "prompt_box": prompt_box,
            "candidates": candidates,
            "best": best,
        }

    def _save_model_test3_png(
        self,
        original: np.ndarray,
        rough_segmentation: Dict[str, Any],
        model_result: Dict[str, Any],
        selected_segmentation: Dict[str, Any],
        out_path: Path,
        filename: str,
    ) -> str:
        original_u8 = self._normalize_to_uint8(original)
        rgb = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2RGB)
        overlay = rgb.copy()

        # Rough classical contour in orange-ish line.
        rough_pts = rough_segmentation["contour"].reshape(-1, 2)
        cv2.polylines(overlay, [rough_pts.astype(np.int32)], True, (255, 170, 0), 3)

        prompt_box = model_result.get("prompt_box")
        if prompt_box is not None:
            x0, y0, x1, y1 = map(int, prompt_box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 150, 255), 3)

        best = model_result.get("best")
        model_mask_rgb = rgb.copy()
        if best is not None:
            mask = best["mask"]
            tint = model_mask_rgb.copy()
            tint[mask] = np.array([0, 255, 0], dtype=np.uint8)
            model_mask_rgb = cv2.addWeighted(model_mask_rgb, 0.72, tint, 0.28, 0)
            pts = best["contour"].reshape(-1, 2).astype(np.int32)
            cv2.polylines(model_mask_rgb, [pts], True, (0, 255, 0), 3)

        final_rgb = rgb.copy()
        final_pts = selected_segmentation["contour"].reshape(-1, 2).astype(np.int32)
        cv2.polylines(final_rgb, [final_pts], True, (0, 255, 0), 3)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(overlay)
        axes[0].set_title("Rough Locator + MobileSAM Prompt Box")
        axes[1].imshow(model_mask_rgb)
        axes[1].set_title("MobileSAM Best Proposed Mask")
        axes[2].imshow(final_rgb)
        axes[2].set_title("Selected Final Boundary")

        for ax in axes:
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

        if model_result.get("success"):
            status = "FINAL: MobileSAM mask selected"
        else:
            status = "FINAL MODEL ATTEMPT FAILED/REJECTED: classical fallback retained"
        if model_result.get("error"):
            status += f"\n{model_result['error']}"

        fig.suptitle(f"{filename} - Final MobileSAM Segmentation Diagnostic\n{status}")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def process_array(
        self,
        image_2d: np.ndarray,
        source_name: str = "array",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a 2D image. Exposed for validation/testing without CZI I/O."""
        enhanced, prep = self.enhance_image(image_2d)
        segmentation = self.segment_tail(enhanced)
        # Independent QA evidence and image-edge clipping are computed after the
        # final contour is selected. They never change the contour itself.
        segmentation["boundary_evidence_score"] = self._boundary_evidence_score(
            self._normalize_to_uint8(image_2d), segmentation["contour"]
        )
        segmentation["edge_clipping"] = self._edge_clipping_status(
            segmentation["contour"], self._normalize_to_uint8(image_2d).shape
        )

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

    def _save_boundary_test2_png(self, original: np.ndarray, enhanced: np.ndarray, segmentation: Dict[str, Any], out_path: Path, filename: str) -> str:
        """Always emit a Test 2 diagnostic so deployment/version can be verified."""
        original_u8 = self._normalize_to_uint8(original)
        method = segmentation.get("method", "unknown")
        contour = segmentation.get("contour")
        upper = segmentation.get("upper_path")
        lower = segmentation.get("lower_path")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, img, title in [
            (axes[0], original_u8, "Original + Test 2 Result"),
            (axes[1], enhanced, "Enhanced + Selected Contour"),
        ]:
            ax.imshow(img, cmap="gray")
            if upper is not None:
                ax.plot(upper[:,0], upper[:,1], linewidth=1.8, label="upper outer path")
            if lower is not None:
                ax.plot(lower[:,0], lower[:,1], linewidth=1.8, label="lower outer path")
            if contour is not None:
                pts = contour.reshape(-1,2)
                ax.plot(np.r_[pts[:,0], pts[0,0]], np.r_[pts[:,1], pts[0,1]], linewidth=1.2, alpha=0.9, label="selected contour")
            ax.set_title(title)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.legend(loc="upper right", fontsize=8)

        if method == "boundary_continuation_test2":
            status = "TEST 2 ACTIVE: boundary-continuation contour selected"
        else:
            status = f"TEST 2 ATTEMPTED BUT REJECTED: fallback selected = {method}"
        fig.suptitle(f"{filename} - Boundary Continuation Test 2\n{status}")
        fig.tight_layout(rect=(0,0,1,0.92))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def _save_preprocessing_test_png(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        out_path: Path,
        filename: str,
    ) -> str:
        """Save a side-by-side diagnostic specifically for Test 1."""
        original_u8 = self._normalize_to_uint8(original)

        # Reproduce the legacy CLAHE view for direct comparison.
        legacy_clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clahe_clip_limit),
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        ).apply(original_u8)

        # Show gradient magnitude of the new preprocessing to assess whether the
        # outer fin edge becomes more continuous than internal texture.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.8)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        p99 = float(np.percentile(mag, 99.5))
        edge_view = np.clip(mag / max(p99, 1e-6) * 255.0, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        panels = [
            (original_u8, "Original Brightfield"),
            (legacy_clahe, "Previous CLAHE"),
            (enhanced, "Enhanced Brightfield"),
            (edge_view, "Test 1 Edge Response"),
        ]
        for ax, (img, title) in zip(axes.ravel(), panels):
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
        fig.suptitle(
            f"{filename} - Preprocessing Test 1\n"
            "Goal: strengthen outer translucent fin margin while suppressing internal texture"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def process_file(self, filepath: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        filepath = Path(filepath)
        output_dir = Path(output_dir) if output_dir else filepath.parent / "analyzed"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.logger is None:
            self.logger = ProcessingLogger(output_dir / "processing_log.txt", self.config.verbose)

        self._log(f"Processing: {filepath.name} | build=REFERENCE_INFORMED_ROI_V2_2026-09-17")
        payload = self.read_czi(filepath)
        bf, selection = self.select_brightfield(payload)

        # First run the existing pipeline only to obtain a rough anatomical locator.
        enhanced, prep = self.enhance_image(bf)
        rough_segmentation = self.segment_tail(enhanced)

        # Test 3: MobileSAM receives the original brightfield plus a generous prompt
        # box derived from the rough locator. It, not the old contour, proposes the
        # final biological mask when it passes safety gates.
        model_result = self._mobile_sam_candidate(bf, rough_segmentation)

        if model_result.get("success"):
            best_model = model_result["best"]
            outer_contour = best_model["contour"]
            outer_mask = (best_model["mask"].astype(np.uint8) * 255)

            # NEW anatomical landmark step: estimate the distal end of the
            # notochord and close the measurement ROI with a vertical line.
            notochord = self._estimate_notochord_endpoint(
                bf, outer_mask, outer_contour
            )
            cutoff = self._apply_notochord_cutoff(outer_mask, notochord)

            contour = cutoff["contour"]
            epsilon = max(0.0, float(self.config.rdp_epsilon_px))
            smoothed = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
            if smoothed is not None and len(smoothed) >= 3:
                contour = smoothed

            # Geometry of the outer AI envelope and final scientific ROI are
            # tracked separately.
            outer_score, outer_geom = self._score_model_contour(
                outer_contour, self._normalize_to_uint8(bf).shape
            )
            final_score, final_geom = self._score_model_contour(
                contour, self._normalize_to_uint8(bf).shape
            )

            segmentation = {
                "contour": contour,
                "binary": cutoff["binary"],
                "outer_contour": outer_contour,
                "outer_binary": outer_mask,
                "method": "mobilesam_reference_informed_roi",
                "model_source": "mobilesam",
                "threshold": None,
                "polarity": "model_mask_plus_reference_informed_notch",
                "contours_total": len(model_result.get("candidates", [])),
                "contours_viable": len(model_result.get("candidates", [])),
                "ambiguity": False,
                "candidate_score": float(best_model["score"]),
                "outer_area_fraction": float(outer_geom["area_fraction"]),
                "outer_bbox_width_fraction": float(outer_geom["bbox_width_fraction"]),
                "outer_bbox_height_fraction": float(outer_geom["bbox_height_fraction"]),
                "outer_bbox_x_fraction": float(outer_geom["bbox_x_fraction"]),
                "area_fraction": float(final_geom["area_fraction"]),
                "bbox_width_fraction": float(final_geom["bbox_width_fraction"]),
                "bbox_height_fraction": float(final_geom["bbox_height_fraction"]),
                "bbox_x_fraction": float(final_geom["bbox_x_fraction"]),
                "retained_fraction_of_outer_mask": float(
                    cutoff["retained_fraction_of_outer_mask"]
                ),
                "reference_shape_prior_score": float(cutoff.get("shape_prior_score", 1.0)),
                "researcher_proximal_boundary_x": cutoff.get("researcher_proximal_boundary_x"),
                "notochord": notochord,
                "original_contour_points": int(len(cutoff["contour"])),
                "smoothed_contour_points": int(len(contour)),
                "attempts": [{
                    "method": "mobilesam_reference_informed_roi",
                    "prompt_box": model_result.get("prompt_box"),
                    "candidate_count": len(model_result.get("candidates", [])),
                    "best_model_score": float(best_model["score"]),
                    "notochord_confidence": float(notochord["confidence"]),
                    "notochord_method": notochord["method"],
                    "cutoff_x": int(notochord["cutoff_x"]),
                }],
            }
        else:
            # If MobileSAM itself fails, preserve the classical fallback and mark
            # the notochord step unavailable rather than inventing a biological ROI.
            segmentation = rough_segmentation
            segmentation["model_source"] = "classical_fallback"
            segmentation["notochord"] = {
                "confidence": 0.0,
                "method": "not_attempted_model_failed",
                "review_required": True,
                "endpoint_x": None,
                "endpoint_y": None,
                "cutoff_x": None,
                "proximal_side": None,
            }

        # Independent QA support and clipping status are computed after the
        # final contour is selected. They do not alter the segmentation.
        bf_u8 = self._normalize_to_uint8(bf)
        evidence_contour = segmentation.get("outer_contour", segmentation["contour"])
        segmentation["boundary_evidence_score"] = self._boundary_evidence_score(
            bf_u8, evidence_contour
        )
        segmentation["edge_clipping"] = self._edge_clipping_status(
            segmentation["contour"], bf_u8.shape
        )

        metrics = self.quantify(segmentation["contour"], enhanced)
        quality = self.assess_quality(enhanced, metrics, segmentation)
        processed = {
            "source_name": filepath.name,
            "enhanced": enhanced,
            "binary": segmentation["binary"],
            "contour": segmentation["contour"],
            "preprocessing": prep,
            "segmentation": segmentation,
            "metrics": metrics,
            "quality": quality,
            "metadata": selection,
        }

        pixel_size = self.config.pixel_size_um
        calibration_source = "manual_override" if self.config.pixel_size_um is not None else None
        if pixel_size is None and payload.get("pixel_size_um_metadata"):
            pixel_size = float(payload["pixel_size_um_metadata"])
            calibration_source = payload.get("pixel_size_source") or "czi_metadata"
            # Quantification already ran; add calibrated metrics here.
            processed["metrics"]["area_micrometers_sq"] = processed["metrics"]["area_pixels_sq"] * pixel_size**2
            processed["metrics"]["perimeter_micrometers"] = processed["metrics"]["perimeter_pixels"] * pixel_size
            bbox = processed["metrics"]["bounding_box"]
            processed["metrics"]["width_micrometers"] = float(bbox["width"]) * pixel_size
            processed["metrics"]["height_micrometers"] = float(bbox["height"]) * pixel_size
            processed["metrics"]["centroid_x_micrometers"] = processed["metrics"]["centroid_x"] * pixel_size
            processed["metrics"]["centroid_y_micrometers"] = processed["metrics"]["centroid_y"] * pixel_size
        elif pixel_size is not None:
            calibration_source = calibration_source or "manual_override"

        stem = filepath.stem
        json_path = output_dir / f"{stem}_metrics.json"
        roi_path = output_dir / f"{stem}_boundary.roi"
        png_path = output_dir / f"{stem}_boundary_trace.png"
        preprocessing_test_path = output_dir / f"{stem}_preprocessing_test1.png"
        boundary_test2_path = output_dir / f"{stem}_boundary_test2.png"
        model_test3_path = output_dir / f"{stem}_model_diagnostic.png"
        raw_mask_path = output_dir / f"{stem}_raw_model_mask.png"
        original_preview_path = output_dir / f"{stem}_original_brightfield.png"
        overlay_preview_path = output_dir / f"{stem}_boundary_overlay.png"
        notochord_overlay_path = output_dir / f"{stem}_notochord_overlay.png"
        notochord_diagnostic_path = output_dir / f"{stem}_notochord_diagnostic.png"
        measurement_mask_path = output_dir / f"{stem}_measurement_mask.png"

        if self.config.save_png:
            try:
                self._save_original_preview(bf, original_preview_path)
                self._save_boundary_overlay_preview(
                    bf,
                    processed["contour"],
                    overlay_preview_path,
                    notochord=processed["segmentation"].get("notochord"),
                    outer_contour=processed["segmentation"].get("outer_contour"),
                )
            except Exception as exc:
                self._log(f"User preview generation failed for {filepath.name}: {exc}")

            if processed["segmentation"].get("model_source") == "mobilesam":
                try:
                    self._save_raw_mask(
                        processed["segmentation"]["outer_binary"], raw_mask_path
                    )
                    self._save_raw_mask(processed["binary"], measurement_mask_path)
                    self._save_notochord_overlay(
                        bf,
                        processed["segmentation"]["outer_contour"],
                        processed["contour"],
                        processed["segmentation"]["notochord"],
                        notochord_overlay_path,
                    )
                    self._save_notochord_diagnostic(
                        bf,
                        processed["segmentation"]["outer_binary"],
                        processed["binary"],
                        processed["segmentation"]["outer_contour"],
                        processed["contour"],
                        processed["segmentation"]["notochord"],
                        notochord_diagnostic_path,
                        filepath.name,
                    )
                except Exception as exc:
                    self._log(f"Notochord QA output generation failed for {filepath.name}: {exc}")
            try:
                self._save_model_test3_png(
                    bf,
                    rough_segmentation,
                    model_result,
                    processed["segmentation"],
                    model_test3_path,
                    filepath.name,
                )
            except Exception as exc:
                self._log(f"Model Test 3 diagnostic PNG generation failed for {filepath.name}: {exc}")
            try:
                self._save_boundary_test2_png(bf, processed["enhanced"], processed["segmentation"], boundary_test2_path, filepath.name)
            except Exception as exc:
                self._log(f"Boundary Test 2 diagnostic PNG generation failed for {filepath.name}: {exc}")
            try:
                self._save_preprocessing_test_png(
                    bf,
                    processed["enhanced"],
                    preprocessing_test_path,
                    filepath.name,
                )
            except Exception as exc:
                self._log(f"Preprocessing comparison PNG generation failed for {filepath.name}: {exc}")

        seg = processed["segmentation"]
        record = {
            "file_metadata": {
                "filename": filepath.name,
                "agent_build": "REFERENCE_INFORMED_ROI_V2_2026-09-17",
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
                "pixel_size_source": calibration_source,
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
                "test2_status": "legacy_fallback_component",
                "test2_selected_method": rough_segmentation.get("method"),
                "segmentation_model": "MobileSAM mobile_sam.pt via Ultralytics",
                "model_status": "model_selected" if seg.get("model_source") == "mobilesam" else "model_attempted_but_fallback_selected",
                "selected_method": seg.get("method"),
                "model_prompt_box": model_result.get("prompt_box"),
                "model_error": model_result.get("error"),
                "edge_clipping": seg.get("edge_clipping"),
                "boundary_evidence_score": seg.get("boundary_evidence_score"),
                "measurement_scope": seg.get("edge_clipping", {}).get("measurement_scope"),
                "notochord_detection": seg.get("notochord"),
                "distal_fin_retained_fraction": seg.get("retained_fraction_of_outer_mask"),
                "roi_definition": "MobileSAM outer tail boundary plus researcher-informed proximal notch anchored at detected notochord tip",
                "reference_shape_priors": {
                    "tip_to_distal_over_height": 0.3431952662721893,
                    "notch_opening_over_height": 0.23339960238568588,
                    "roi_aspect_ratio": 0.8475499092558983
                },
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
            "Segmentation_Method": record["processing_parameters"]["threshold_method_used"],
            "QualityScore": q["quality_score"],
            "SegmentationConfidence": q["segmentation_confidence"],
            "BoundaryEvidenceScore": q.get("boundary_evidence_score"),
            "NotochordConfidence": q.get("notochord_confidence"),
            "ReferenceShapePriorScore": q.get("reference_shape_prior_score"),
            "NotochordReviewRequired": q.get("notochord_review_required"),
            "Notochord_Method": record.get("processing_notes", {}).get("notochord_detection", {}).get("method"),
            "Notochord_Endpoint_X_px": record.get("processing_notes", {}).get("notochord_detection", {}).get("endpoint_x"),
            "Notochord_Endpoint_Y_px": record.get("processing_notes", {}).get("notochord_detection", {}).get("endpoint_y"),
            "Notochord_Cutoff_X_px": record.get("processing_notes", {}).get("notochord_detection", {}).get("cutoff_x"),
            "Proximal_Side": record.get("processing_notes", {}).get("notochord_detection", {}).get("proximal_side"),
            "Distal_Fin_Retained_Fraction": record.get("processing_notes", {}).get("distal_fin_retained_fraction"),
            "Edge_Clipped": q.get("edge_clipping", {}).get("edge_clipped", False),
            "Clipped_Edges": ",".join(q.get("edge_clipping", {}).get("clipped_edges", [])),
            "Measurement_Scope": q.get("edge_clipping", {}).get("measurement_scope", ""),
            "Quality_Flag": q["quality_flag"],
            "Recommendation": q["recommendation"],
        }
        if "area_micrometers_sq" in m:
            row["Pixel_Size_um"] = record["file_metadata"].get("pixel_size_um")
            row["Calibration_Source"] = record["file_metadata"].get("pixel_size_source")
            row["Area_um2"] = m["area_micrometers_sq"]
            row["Perimeter_um"] = m["perimeter_micrometers"]
            row["Width_um"] = m.get("width_micrometers")
            row["Height_um"] = m.get("height_micrometers")
            row["Centroid_X_um"] = m.get("centroid_x_micrometers")
            row["Centroid_Y_um"] = m.get("centroid_y_micrometers")
            px_um = record["file_metadata"].get("pixel_size_um")
            cutoff_px = row.get("Notochord_Cutoff_X_px")
            endpoint_y_px = row.get("Notochord_Endpoint_Y_px")
            if px_um is not None and cutoff_px is not None:
                row["Notochord_Cutoff_X_um"] = float(cutoff_px) * float(px_um)
            if px_um is not None and endpoint_y_px is not None:
                row["Notochord_Endpoint_Y_um"] = float(endpoint_y_px) * float(px_um)
        return row

    def export_summary_csv(self, output_dir: Path) -> Optional[Path]:
        if not self.results:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tail_metrics_summary.csv"
        fields = list(self.results[0].keys())
        # Include calibrated columns if any record has them.
        for extra in ("Pixel_Size_um", "Calibration_Source", "Area_um2", "Perimeter_um", "Width_um", "Height_um", "Centroid_X_um", "Centroid_Y_um", "Notochord_Cutoff_X_um", "Notochord_Endpoint_Y_um"):
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
