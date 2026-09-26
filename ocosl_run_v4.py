#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCOSL runner v4
================
Extends v3 with a stronger learned relation-generator experiment requested by
reviewers. In addition to the original geometry-based relation candidates, v4
can ingest predictions from an official RelTR checkout + pretrained checkpoint
and evaluate the SAME learned relation candidate pool:

  (A) RelTR-only (greedy selection, no ontology-constrained ILP)
  (B) RelTR + OCOSL (same RelTR candidates followed by Algorithm 1)

This directly tests whether the benefit of ontology-constrained optimization
persists when relation proposals come from a substantially stronger learned
scene-graph generator.

New features
------------
* --relation_generator {geometry,reltr,hybrid}
* --reltr_repo / --reltr_ckpt / --reltr_dataset / --reltr_device
* --mode eval_strong_rel
* Paired per-image output for RelTR-only vs RelTR+OCOSL
* Bootstrap 95% confidence intervals for relation F1
* Paired permutation test on per-image relation F1 differences
* CSV + JSON export for direct insertion into the manuscript

Important reproducibility note
------------------------------
The script does NOT download models automatically. Clone the official RelTR
repository and download an appropriate pretrained checkpoint separately. This
keeps the experiment auditable and avoids silently changing model versions.

The existing ocosl_utils_v3.py file is still required.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ocosl_utils_v3 import (
    DetObj, RelCand,
    load_ontology_schema, build_relation_type_constraints, build_synonym_table,
    spatial_relations, build_ontology_score, DEFAULT_TYPE_MAP,
    solve_ilp, greedy_repair, enrich_graph, draw_labeled_image,
    eval_bbox_label_accuracy, normalize_label,
    load_openimages_class_map, load_openimages_boxes_for_subset,
    load_openimages_rel_for_subset, eval_openimages_relations,
)


# -----------------------------------------------------------------------------
# YOLOv8 object detector
# -----------------------------------------------------------------------------
def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Please install ultralytics: python -m pip install ultralytics"
        ) from e
    return YOLO(model_name)


def yolo_detect(model, image_path: str, conf=0.25, max_det=50):
    res = model.predict(source=image_path, conf=conf, max_det=max_det, verbose=False)
    r = res[0]
    boxes = r.boxes
    names = r.names
    dets = []
    if boxes is None or len(boxes) == 0:
        return dets
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = xyxy
        bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        cls_id = int(b.cls[0].item())
        label = names.get(cls_id, str(cls_id))
        conf_score = float(b.conf[0].item())
        dets.append((bbox, label, conf_score))
    return dets


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def dedup_detections(dets, iou_thr=0.6):
    """Keep the highest-confidence same-label box within an IoU cluster."""
    dets = sorted(dets, key=lambda x: x[2], reverse=True)
    kept = []
    for bbox, label, conf in dets:
        dup = False
        for kb, kl, _ in kept:
            if label == kl and iou_xywh(bbox, kb) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append((bbox, label, conf))
    return kept


def build_candidates(det_label: str, det_conf: float, topk: int, synonym_map: dict):
    """Construct a small candidate-label set from YOLO output and ontology aliases."""
    det_label_n = normalize_label(det_label)
    cands = [(det_label_n, det_conf)]
    if det_label_n in synonym_map:
        canon = synonym_map[det_label_n]
        if canon != det_label_n:
            cands.append((canon, max(0.01, det_conf - 0.05)))
    if len(cands) < topk:
        cands.append(("physical object", max(0.01, det_conf * 0.20)))

    uniq = []
    seen = set()
    for lbl, score in sorted(cands, key=lambda z: z[1], reverse=True):
        if lbl not in seen:
            uniq.append((lbl, float(score)))
            seen.add(lbl)
        if len(uniq) >= topk:
            break
    return uniq


def build_objects_for_image(image_path: str, model, args, synonym_map, type_map):
    dets = yolo_detect(model, image_path, conf=args.det_conf, max_det=args.max_det)
    dets = dedup_detections(dets, iou_thr=args.iou_dup)

    objects = []
    for i, (bbox, det_label, det_conf) in enumerate(dets):
        candidates = build_candidates(det_label, det_conf, args.topk, synonym_map)
        onto_scores = {}
        for cand_label, sdet in candidates:
            onto_scores[cand_label] = build_ontology_score(
                det_score=sdet,
                det_label=det_label,
                cand_label=cand_label,
                synonym_map=synonym_map,
                type_map=type_map,
                rel_support=0.0,
            )
        objects.append(
            DetObj(
                idx=i,
                bbox=bbox,
                det_label=det_label,
                det_conf=det_conf,
                candidates=candidates,
                onto_scores=onto_scores,
            )
        )
    return objects


# -----------------------------------------------------------------------------
# Original geometry relation generator
# -----------------------------------------------------------------------------
def geometry_relation_candidates(objects: List[DetObj], args) -> List[RelCand]:
    rel_cands: List[RelCand] = []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            for rel, score, feasible in spatial_relations(
                objects[i].bbox, objects[j].bbox, near_px=args.near_px
            ):
                score = float(score)
                feasible = bool(feasible)
                if (not feasible) or score < args.tau_rel:
                    continue
                rel_cands.append(
                    RelCand(i=i, j=j, rel=rel, score=score, feasible=True)
                )
    return rel_cands


# -----------------------------------------------------------------------------
# RelTR stronger learned relation generator
# -----------------------------------------------------------------------------
# Visual Genome predicate vocabulary used by the official RelTR inference demo.
RELTR_VG_REL_CLASSES = [
    "__background__", "above", "across", "against", "along", "and", "at",
    "attached to", "behind", "belonging to", "between", "carrying",
    "covered in", "covering", "eating", "flying in", "for", "from",
    "growing on", "hanging from", "has", "holding", "in", "in front of",
    "laying on", "looking at", "lying on", "made of", "mounted on", "near",
    "of", "on", "on back of", "over", "painted on", "parked on", "part of",
    "playing", "riding", "says", "sitting on", "standing on", "to", "under",
    "using", "walking in", "walking on", "watching", "wearing", "wears", "with",
]

# Map learned predicates to the canonical OCOSL relation names used by the ontology
# and by eval_openimages_relations(). Unknown predicates can be ignored to avoid
# falsely claiming semantic equivalence.
RELTR_TO_OCOSL = {
    "above": "isAbove",
    "behind": "behind",
    "carrying": "isHolding",
    "holding": "isHolding",
    "in": "insideOf",
    "in front of": "inFrontOf",
    "near": "near",
    "on": "isOn",
    "over": "isAbove",
    "parked on": "parkedOn",
    "part of": "partOf",
    "riding": "riding",
    "sitting on": "sittingOn",
    "standing on": "standingOn",
    "under": "isBelow",
    "using": "usesTool",
    "wearing": "wearing",
    "wears": "wearing",
    "with": "interactsWith",
    "attached to": "attachedTo",
    "at": "near",
}


def _load_relation_label_list(path: str) -> List[str]:
    """Load relation labels from JSON list/dict or one-label-per-line text file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Relation label file not found: {path}")
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            # Accept {"0": "rel", ...} or {"rel": 0, ...}
            if all(str(k).isdigit() for k in obj.keys()):
                return [str(obj[k]) for k in sorted(obj, key=lambda z: int(z))]
            inv = sorted(((int(v), str(k)) for k, v in obj.items()), key=lambda z: z[0])
            return [name for _, name in inv]
        raise ValueError("Unsupported JSON relation-label structure")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


class RelTRAdapter:
    """Thin inference adapter around the official RelTR repository.

    The adapter intentionally keeps RelTR separate from OCOSL. RelTR first generates
    subject-predicate-object triplets. Those learned triplets are then matched to the
    YOLO candidate objects by bounding-box IoU. This produces RelCand objects that can
    be used either directly (learned-only baseline) or as input to the OCOSL ILP.
    """

    def __init__(
        self,
        repo: str,
        checkpoint: str,
        dataset: str = "vg",
        device: str = "cpu",
        relation_labels_path: str = "",
        query_threshold: float = 0.30,
        max_triplets: int = 50,
    ):
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.dataset = dataset
        self.device_name = device
        self.query_threshold = float(query_threshold)
        self.max_triplets = int(max_triplets)

        if not self.repo.exists():
            raise FileNotFoundError(f"RelTR repository not found: {self.repo}")
        if not (self.repo / "inference.py").exists():
            raise FileNotFoundError(
                f"Expected official RelTR inference.py under: {self.repo}"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"RelTR checkpoint not found: {self.checkpoint}")

        try:
            import torch
            import torchvision.transforms as T
        except Exception as e:
            raise RuntimeError(
                "RelTR experiment requires torch and torchvision in the active environment."
            ) from e

        self.torch = torch
        self.T = T

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[RelTR] CUDA requested but unavailable; using CPU.")
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)

        # Import the official inference module so that its parser remains the source
        # of model-architecture defaults. This avoids silently diverging from RelTR.
        repo_str = str(self.repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        spec = importlib.util.spec_from_file_location(
            "_ocosl_reltr_inference", self.repo / "inference.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import RelTR inference.py")
        reltr_inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reltr_inference)
        reltr_args = reltr_inference.get_args_parser().parse_args([])
        reltr_args.dataset = dataset
        reltr_args.device = self.device_name
        reltr_args.resume = str(self.checkpoint)

        # Build through the official RelTR model factory imported by inference.py.
        build_model = reltr_inference.build_model
        model, _, _ = build_model(reltr_args)
        try:
            ckpt = torch.load(
                str(self.checkpoint), map_location=self.device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(str(self.checkpoint), map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

        if relation_labels_path:
            self.rel_labels = _load_relation_label_list(relation_labels_path)
        elif dataset == "vg":
            self.rel_labels = RELTR_VG_REL_CLASSES
        else:
            raise ValueError(
                "For --reltr_dataset oi, provide --reltr_relation_labels with the "
                "predicate vocabulary matching the Open Images RelTR checkpoint."
            )

        self.transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        return RelTRAdapter._stack(
            [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
        )

    @staticmethod
    def _stack(parts):
        import torch
        return torch.stack(parts, dim=1)

    def _rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self._box_cxcywh_to_xyxy(out_bbox)
        scale = self.torch.tensor(
            [img_w, img_h, img_w, img_h], dtype=self.torch.float32, device=b.device
        )
        return b * scale

    def predict(self, image_path: str) -> List[dict]:
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        tensor = self.transform(im).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(tensor)

        rel_p = outputs["rel_logits"].softmax(-1)[0, :, :-1]
        sub_p = outputs["sub_logits"].softmax(-1)[0, :, :-1]
        obj_p = outputs["obj_logits"].softmax(-1)[0, :, :-1]

        rel_score, rel_idx = rel_p.max(-1)
        sub_score, _ = sub_p.max(-1)
        obj_score, _ = obj_p.max(-1)
        joint = rel_score * sub_score * obj_score

        keep = (
            (rel_score >= self.query_threshold)
            & (sub_score >= self.query_threshold)
            & (obj_score >= self.query_threshold)
        )
        qidx = self.torch.nonzero(keep, as_tuple=True)[0]
        if qidx.numel() == 0:
            return []

        order = self.torch.argsort(joint[qidx], descending=True)
        qidx = qidx[order[: self.max_triplets]]

        sub_boxes = self._rescale_bboxes(outputs["sub_boxes"][0, qidx], im.size)
        obj_boxes = self._rescale_bboxes(outputs["obj_boxes"][0, qidx], im.size)

        preds = []
        for q, sb, ob in zip(qidx.tolist(), sub_boxes.tolist(), obj_boxes.tolist()):
            ridx = int(rel_idx[q].item())
            if ridx < 0 or ridx >= len(self.rel_labels):
                continue
            learned_rel = normalize_label(self.rel_labels[ridx])
            if learned_rel == "__background__":
                continue
            canonical = RELTR_TO_OCOSL.get(learned_rel)
            if canonical is None:
                # Skip unmapped predicates so evaluation does not equate unrelated labels.
                continue
            preds.append(
                {
                    "sub_box": xyxy_to_xywh(sb),
                    "obj_box": xyxy_to_xywh(ob),
                    "learned_rel": learned_rel,
                    "rel": canonical,
                    "score": float(joint[q].item()),
                }
            )
        return preds


class PrecomputedRelTRAdapter:
    """Read RelTR predictions exported from a separate compatible environment.

    JSONL format (one image per line):
      {"image_id": "abc", "relations": [
        {"sub_box": [x,y,w,h], "obj_box": [x,y,w,h],
         "relation": "on", "score": 0.83}
      ]}

    This path is useful because the official RelTR release targets an older
    PyTorch stack; OCOSL can still consume its predictions reproducibly.
    """

    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"RelTR prediction JSONL not found: {self.path}")
        self.by_id = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = str(row.get("image_id") or row.get("image") or "")
                if iid:
                    self.by_id[iid] = row.get("relations", [])

    def predict(self, image_path: str) -> List[dict]:
        iid = Path(image_path).stem
        out = []
        for r in self.by_id.get(iid, []):
            learned_rel = normalize_label(str(r.get("relation", r.get("rel", ""))))
            canonical = RELTR_TO_OCOSL.get(learned_rel)
            if canonical is None:
                continue
            out.append({
                "sub_box": tuple(float(x) for x in r["sub_box"]),
                "obj_box": tuple(float(x) for x in r["obj_box"]),
                "learned_rel": learned_rel,
                "rel": canonical,
                "score": float(r.get("score", 1.0)),
            })
        return out


def _best_object_match(box, objects: List[DetObj], min_iou: float) -> Optional[int]:
    best_idx = None
    best_iou = 0.0
    for o in objects:
        val = iou_xywh(box, o.bbox)
        if val > best_iou:
            best_iou = val
            best_idx = o.idx
    return best_idx if best_idx is not None and best_iou >= min_iou else None


def reltr_relation_candidates(
    image_path: str,
    objects: List[DetObj],
    adapter: RelTRAdapter,
    args,
) -> List[RelCand]:
    preds = adapter.predict(image_path)
    best: Dict[Tuple[int, int, str], float] = {}
    for pred in preds:
        i = _best_object_match(pred["sub_box"], objects, args.reltr_match_iou)
        j = _best_object_match(pred["obj_box"], objects, args.reltr_match_iou)
        if i is None or j is None or i == j:
            continue
        score = float(pred["score"])
        if score < args.tau_rel:
            continue
        key = (i, j, pred["rel"])
        best[key] = max(best.get(key, 0.0), score)

    return [
        RelCand(i=i, j=j, rel=r, score=s, feasible=True)
        for (i, j, r), s in sorted(best.items(), key=lambda z: z[1], reverse=True)
    ]


def merge_relation_candidates(*candidate_lists: Sequence[RelCand]) -> List[RelCand]:
    """Union candidate lists while retaining the highest score per (i,j,r)."""
    best: Dict[Tuple[int, int, str], RelCand] = {}
    for lst in candidate_lists:
        for rc in lst:
            key = (rc.i, rc.j, rc.rel)
            if key not in best or rc.score > best[key].score:
                best[key] = rc
    return sorted(best.values(), key=lambda z: z.score, reverse=True)


# -----------------------------------------------------------------------------
# Candidate problem construction and optimization
# -----------------------------------------------------------------------------
def build_problem_for_image(
    image_path: str,
    model,
    args,
    synonym_map,
    type_map,
    rel_constraints,
    reltr_adapter: Optional[RelTRAdapter] = None,
):
    objects = build_objects_for_image(image_path, model, args, synonym_map, type_map)

    if args.relation_generator == "geometry":
        rel_cands = geometry_relation_candidates(objects, args)
    elif args.relation_generator == "reltr":
        if reltr_adapter is None:
            raise RuntimeError("RelTR adapter is required for relation_generator=reltr")
        rel_cands = reltr_relation_candidates(image_path, objects, reltr_adapter, args)
    elif args.relation_generator == "hybrid":
        if reltr_adapter is None:
            raise RuntimeError("RelTR adapter is required for relation_generator=hybrid")
        rel_cands = merge_relation_candidates(
            geometry_relation_candidates(objects, args),
            reltr_relation_candidates(image_path, objects, reltr_adapter, args),
        )
    else:
        raise ValueError(f"Unknown relation generator: {args.relation_generator}")

    return objects, rel_cands


def solve_problem(objects, rel_cands, args, rel_constraints, synonym_map, type_map):
    budget = args.rel_budget if args.rel_budget > 0 else max(1, 2 * len(objects))

    t0 = time.time()
    try:
        x_star, y_star, status, used_fallback = solve_ilp(
            objects,
            rel_cands,
            rel_constraints,
            synonym_map,
            type_map,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            time_limit_s=args.time_limit,
            rel_budget=budget,
        )
    except Exception:
        x_star, y_star = greedy_repair(
            objects, rel_cands, rel_constraints, synonym_map, type_map
        )
        status, used_fallback = "FallbackGreedy", True
    dt = time.time() - t0

    sel_label, enriched_rels = enrich_graph(objects, x_star, y_star)
    return x_star, y_star, sel_label, enriched_rels, status, used_fallback, dt


def select_learned_relations_without_ocosl(
    rel_cands: List[RelCand],
    n_objects: int,
    rel_budget: int,
) -> List[Tuple[int, int, str]]:
    """Greedy learned-generator baseline with no ontology-constrained ILP.

    To keep output density comparable, this baseline uses the same global relation
    budget as OCOSL and retains at most one highest-scoring predicate per ordered
    object pair. No ontology domain/range constraint is used in this selection.
    """
    budget = rel_budget if rel_budget > 0 else max(1, 2 * n_objects)
    pair_best: Dict[Tuple[int, int], RelCand] = {}
    for rc in sorted(rel_cands, key=lambda z: z.score, reverse=True):
        pair_best.setdefault((rc.i, rc.j), rc)
    selected = sorted(pair_best.values(), key=lambda z: z.score, reverse=True)[:budget]
    return [(rc.i, rc.j, rc.rel) for rc in selected]


# -----------------------------------------------------------------------------
# Statistics for the stronger-generator experiment
# -----------------------------------------------------------------------------
def bootstrap_mean_ci(values, n_boot=5000, seed=42, alpha=0.05):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = rng.choice(x, size=len(x), replace=True).mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(x.mean()), lo, hi


def paired_permutation_pvalue(a, b, n_perm=20000, seed=42):
    """Two-sided paired randomization test on per-image metric differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    d = b - a
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(d))
        if abs(float((d * signs).mean())) >= obs - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def summarize_relation_metrics(rows: List[dict], prefix: str):
    keys = ["rel_precision", "rel_recall", "rel_f1"]
    out = {}
    for key in keys:
        vals = [r[f"{prefix}_{key}"] for r in rows]
        mean, lo, hi = bootstrap_mean_ci(vals)
        out[key] = mean
        out[f"{key}_ci95_low"] = lo
        out[f"{key}_ci95_high"] = hi
    out["avg_pred_rels"] = float(np.mean([r[f"{prefix}_pred_rels"] for r in rows]))
    out["avg_time_s"] = float(np.mean([r[f"{prefix}_time_s"] for r in rows]))
    return out


# -----------------------------------------------------------------------------
# Single-image mode
# -----------------------------------------------------------------------------
def _make_reltr_adapter_if_needed(args):
    if args.relation_generator not in ("reltr", "hybrid") and args.mode != "eval_strong_rel":
        return None
    if args.reltr_predictions_jsonl:
        return PrecomputedRelTRAdapter(args.reltr_predictions_jsonl)
    if not args.reltr_repo or not args.reltr_ckpt:
        raise SystemExit(
            "RelTR mode requires either --reltr_predictions_jsonl OR both "
            "--reltr_repo and --reltr_ckpt. See --help."
        )
    return RelTRAdapter(
        repo=args.reltr_repo,
        checkpoint=args.reltr_ckpt,
        dataset=args.reltr_dataset,
        device=args.reltr_device,
        relation_labels_path=args.reltr_relation_labels,
        query_threshold=args.reltr_query_thr,
        max_triplets=args.reltr_topk,
    )


def run_single(args):
    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    model = load_yolo(args.yolo_model)
    reltr_adapter = _make_reltr_adapter_if_needed(args)

    objects, rel_cands = build_problem_for_image(
        args.image,
        model,
        args,
        synonym_map,
        type_map,
        rel_constraints,
        reltr_adapter=reltr_adapter,
    )
    if not objects:
        print("No detections.")
        return

    _, _, sel_label, enriched_rels, status, used_fallback, dt = solve_problem(
        objects, rel_cands, args, rel_constraints, synonym_map, type_map
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path(args.image).with_suffix("").with_name(Path(args.image).stem + "_ocosl.png")
    )
    draw_labeled_image(args.image, str(out_path), objects, sel_label)

    print(f"Relation generator: {args.relation_generator}")
    print(f"Candidate relations: {len(rel_cands)}")
    print(f"Status: {status} | fallback={used_fallback} | time={dt:.3f}s")
    print("Saved:", out_path)
    print("Selected labels:", sel_label)
    print("Enriched relations (first 20):", enriched_rels[:20])


# -----------------------------------------------------------------------------
# COCO evaluation (unchanged object-label evaluation)
# -----------------------------------------------------------------------------
def load_coco_gt(ann_json: str, image_id: int):
    with open(ann_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cat = {c["id"]: c["name"] for c in data["categories"]}
    anns = [a for a in data["annotations"] if a["image_id"] == image_id]
    boxes = [tuple(a["bbox"]) for a in anns]
    labels = [cat[a["category_id"]] for a in anns]
    return boxes, labels


def run_eval_coco(args):
    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    model = load_yolo(args.yolo_model)
    reltr_adapter = _make_reltr_adapter_if_needed(args)

    with open(args.coco_ann, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = coco["images"][: args.max_images]

    metrics = []
    for img in images:
        image_id = int(img["id"])
        file_name = img["file_name"]
        img_path = Path(args.coco_root) / file_name
        if not img_path.exists():
            continue

        objects, rel_cands = build_problem_for_image(
            str(img_path), model, args, synonym_map, type_map, rel_constraints, reltr_adapter
        )
        _, _, sel_label, _, _, _, _ = solve_problem(
            objects, rel_cands, args, rel_constraints, synonym_map, type_map
        )

        gt_boxes, gt_labels = load_coco_gt(args.coco_ann, image_id)
        metrics.append(
            eval_bbox_label_accuracy(
                objects, sel_label, gt_boxes, gt_labels, iou_thr=args.iou_thr
            )
        )

    if not metrics:
        print("No evaluated images. Check coco_root/coco_ann paths.")
        return
    avg = {
        k: float(np.mean([m[k] for m in metrics]))
        for k in ["precision", "recall", "f1", "mean_iou"]
    }
    print("COCO evaluation over", len(metrics), "images")
    print(avg)


# -----------------------------------------------------------------------------
# Open Images loaders/evaluation
# -----------------------------------------------------------------------------
def load_subset_ids(path: str, max_n: int = 0):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s)
            if max_n and len(ids) >= max_n:
                break
    return set(ids)


def prepare_openimages_subset(args):
    from PIL import Image

    subset_ids = load_subset_ids(args.oi_subset, max_n=args.max_images)
    if not subset_ids:
        raise SystemExit("Open Images subset is empty. Provide --oi_subset with image IDs.")

    image_sizes = {}
    image_paths = {}
    for iid in subset_ids:
        p = Path(args.oi_root) / f"{iid}.jpg"
        if not p.exists():
            p = Path(args.oi_root) / f"{iid}.png"
        if not p.exists():
            continue
        image_paths[iid] = str(p)
        with Image.open(p) as im:
            image_sizes[iid] = im.size

    subset_ids = set(image_sizes.keys())
    if not subset_ids:
        raise SystemExit("No images found for the given IDs. Check --oi_root.")

    mid_to_name = load_openimages_class_map(args.oi_classes)
    gt_boxes = load_openimages_boxes_for_subset(
        args.oi_bbox_csv,
        subset_ids,
        mid_to_name,
        image_sizes,
        keep_groupof=args.oi_keep_groupof,
    )
    gt_rels = None
    if args.oi_rel_csv:
        gt_rels = load_openimages_rel_for_subset(
            args.oi_rel_csv, subset_ids, mid_to_name, image_sizes
        )
    return subset_ids, image_paths, gt_boxes, gt_rels


def run_eval_openimages(args):
    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    model = load_yolo(args.yolo_model)
    reltr_adapter = _make_reltr_adapter_if_needed(args)

    subset_ids, image_paths, gt_boxes, gt_rels = prepare_openimages_subset(args)

    label_metrics = []
    rel_metrics = []
    for iid in sorted(subset_ids):
        img_path = image_paths[iid]
        objects, rel_cands = build_problem_for_image(
            img_path, model, args, synonym_map, type_map, rel_constraints, reltr_adapter
        )
        _, _, sel_label, enriched_rels, _, _, _ = solve_problem(
            objects, rel_cands, args, rel_constraints, synonym_map, type_map
        )

        gt_list = gt_boxes.get(iid, [])
        gt_b = [b for b, _ in gt_list]
        gt_l = [l for _, l in gt_list]
        label_metrics.append(
            eval_bbox_label_accuracy(objects, sel_label, gt_b, gt_l, iou_thr=args.iou_thr)
        )

        if gt_rels is not None:
            rel_metrics.append(
                eval_openimages_relations(
                    objects, enriched_rels, gt_rels.get(iid, []), iou_thr=args.iou_thr
                )
            )

        if args.save_outputs:
            Path(args.save_outputs).mkdir(parents=True, exist_ok=True)
            out_img = Path(args.save_outputs) / f"{iid}_ocosl.png"
            draw_labeled_image(img_path, str(out_img), objects, sel_label)

    avg_lbl = {
        k: float(np.mean([m[k] for m in label_metrics]))
        for k in ["precision", "recall", "f1", "mean_iou"]
    }
    print("Open Images subset label evaluation over", len(label_metrics), "images")
    print(avg_lbl)

    if rel_metrics:
        avg_rel = {
            k: float(np.mean([m[k] for m in rel_metrics]))
            for k in ["rel_precision", "rel_recall", "rel_f1"]
        }
        avg_rel["avg_gt_rels"] = float(np.mean([m["gt_rels"] for m in rel_metrics]))
        avg_rel["avg_pred_rels"] = float(np.mean([m["pred_rels"] for m in rel_metrics]))
        print("Open Images relation evaluation (object-object only; RelationLabel='is' excluded):")
        print(avg_rel)


# -----------------------------------------------------------------------------
# Reviewer-requested stronger relation-generator experiment
# -----------------------------------------------------------------------------
def run_eval_strong_relation(args):
    """Compare learned RelTR proposals with and without OCOSL on the OI subset.

    The relation candidate pool is identical for both methods. The only difference is:
      - RelTR-only: score-ranked greedy relation selection, no ontology-constrained ILP.
      - RelTR + OCOSL: Algorithm 1 with ontology/type/spatial/cardinality constraints.

    This is the cleanest test of whether optimization benefit persists under a stronger
    learned relation proposal model.
    """
    if not args.oi_rel_csv:
        raise SystemExit("eval_strong_rel requires --oi_rel_csv relation ground truth.")

    # Force learned relation candidates for this experiment.
    args.relation_generator = "reltr"

    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    yolo = load_yolo(args.yolo_model)
    reltr = _make_reltr_adapter_if_needed(args)

    subset_ids, image_paths, _, gt_rels = prepare_openimages_subset(args)
    rows = []

    for idx, iid in enumerate(sorted(subset_ids), 1):
        img_path = image_paths[iid]

        # Shared detector objects and shared learned candidate relations.
        t_build = time.time()
        objects = build_objects_for_image(img_path, yolo, args, synonym_map, type_map)
        rel_cands = reltr_relation_candidates(img_path, objects, reltr, args)
        shared_generator_time = time.time() - t_build

        # A) Learned generator without OCOSL.
        t0 = time.time()
        raw_rels = select_learned_relations_without_ocosl(
            rel_cands, len(objects), args.rel_budget
        )
        raw_select_time = time.time() - t0
        raw_m = eval_openimages_relations(
            objects, raw_rels, gt_rels.get(iid, []), iou_thr=args.iou_thr
        )

        # B) Same learned generator + OCOSL optimization.
        t1 = time.time()
        _, _, _, opt_rels, status, used_fallback, solve_dt = solve_problem(
            objects, rel_cands, args, rel_constraints, synonym_map, type_map
        )
        opt_total_postgen = time.time() - t1
        opt_m = eval_openimages_relations(
            objects, opt_rels, gt_rels.get(iid, []), iou_thr=args.iou_thr
        )

        row = {
            "image_id": iid,
            "n_objects": len(objects),
            "n_reltr_candidates": len(rel_cands),
            "shared_generator_time_s": shared_generator_time,
            "reltr_rel_precision": raw_m["rel_precision"],
            "reltr_rel_recall": raw_m["rel_recall"],
            "reltr_rel_f1": raw_m["rel_f1"],
            "reltr_pred_rels": raw_m["pred_rels"],
            "reltr_time_s": shared_generator_time + raw_select_time,
            "ocosl_rel_precision": opt_m["rel_precision"],
            "ocosl_rel_recall": opt_m["rel_recall"],
            "ocosl_rel_f1": opt_m["rel_f1"],
            "ocosl_pred_rels": opt_m["pred_rels"],
            "ocosl_time_s": shared_generator_time + opt_total_postgen,
            "solver_time_s": solve_dt,
            "solver_status": status,
            "fallback": int(bool(used_fallback)),
        }
        rows.append(row)
        print(
            f"[{idx}/{len(subset_ids)}] {iid}: candidates={len(rel_cands)} "
            f"RelTR F1={raw_m['rel_f1']:.3f} -> RelTR+OCOSL F1={opt_m['rel_f1']:.3f}"
        )

    if not rows:
        raise SystemExit("No images were evaluated.")

    raw = summarize_relation_metrics(rows, "reltr")
    opt = summarize_relation_metrics(rows, "ocosl")
    pval = paired_permutation_pvalue(
        [r["reltr_rel_f1"] for r in rows],
        [r["ocosl_rel_f1"] for r in rows],
        n_perm=args.permutation_samples,
        seed=args.stats_seed,
    )

    summary = {
        "experiment": "stronger_relation_generator",
        "generator": "RelTR",
        "reltr_dataset": args.reltr_dataset,
        "n_images": len(rows),
        "tau_rel": args.tau_rel,
        "rel_budget": args.rel_budget if args.rel_budget > 0 else "2n",
        "reltr_only": raw,
        "reltr_plus_ocosl": opt,
        "paired_rel_f1_gain": opt["rel_f1"] - raw["rel_f1"],
        "paired_permutation_pvalue_rel_f1": pval,
        "fallback_count": int(sum(r["fallback"] for r in rows)),
    }

    print("\n=== Stronger relation-generator experiment ===")
    print(json.dumps(summary, indent=2))

    if args.experiment_csv:
        out = Path(args.experiment_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Per-image CSV:", out)

    if args.experiment_json:
        out = Path(args.experiment_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Summary JSON:", out)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="OCOSL v4: ontology-constrained inference with optional RelTR relation proposals"
    )
    p.add_argument(
        "--ontology",
        type=str,
        default="JOCO_verified_ontology_schema_fixed.owl",
    )
    p.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--det_conf", type=float, default=0.25)
    p.add_argument("--max_det", type=int, default=30)
    p.add_argument("--near_px", type=float, default=80.0)

    p.add_argument(
        "--tau_rel",
        type=float,
        default=0.7,
        help="Relation score threshold for geometry/learned candidate pruning",
    )
    p.add_argument(
        "--rel_budget",
        type=int,
        default=0,
        help="Global relation budget C; 0 uses the manuscript default C=2n",
    )
    p.add_argument(
        "--iou_dup",
        type=float,
        default=0.6,
        help="IoU threshold for same-label duplicate suppression",
    )

    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.20)
    # ocosl_utils_v3 currently exposes alpha/beta/gamma but not lambda in solve_ilp.
    p.add_argument(
        "--lambda_conflict",
        type=float,
        default=0.10,
        help="Reserved manuscript conflict-penalty weight; hard constraints are enforced in v3 utils",
    )
    p.add_argument("--time_limit", type=int, default=10)

    p.add_argument(
        "--relation_generator",
        choices=["geometry", "reltr", "hybrid"],
        default="geometry",
        help="Relation proposal source",
    )

    # RelTR integration. The official repository and checkpoint are user-supplied.
    p.add_argument(
        "--reltr_predictions_jsonl", type=str, default="",
        help="Optional precomputed RelTR predictions; avoids importing the legacy RelTR environment",
    )
    p.add_argument("--reltr_repo", type=str, default="", help="Path to official RelTR repository")
    p.add_argument("--reltr_ckpt", type=str, default="", help="Path to pretrained RelTR checkpoint")
    p.add_argument(
        "--reltr_dataset",
        choices=["vg", "oi"],
        default="vg",
        help="RelTR checkpoint label space",
    )
    p.add_argument(
        "--reltr_relation_labels",
        type=str,
        default="",
        help="Predicate vocabulary file required for OI checkpoints; JSON or one label/line",
    )
    p.add_argument("--reltr_device", type=str, default="cpu")
    p.add_argument(
        "--reltr_query_thr",
        type=float,
        default=0.30,
        help="Minimum RelTR subject/object/predicate query confidence",
    )
    p.add_argument("--reltr_topk", type=int, default=50)
    p.add_argument(
        "--reltr_match_iou",
        type=float,
        default=0.30,
        help="Minimum IoU for mapping a RelTR subject/object box to a YOLO object",
    )

    p.add_argument(
        "--mode",
        choices=["single", "eval_coco", "eval_openimages", "eval_strong_rel"],
        default="single",
    )
    p.add_argument("--image", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--max_images", type=int, default=50)
    p.add_argument("--iou_thr", type=float, default=0.5)

    # COCO
    p.add_argument("--coco_root", type=str, default="")
    p.add_argument("--coco_ann", type=str, default="")

    # Open Images subset
    p.add_argument("--oi_root", type=str, default="", help="Folder containing <ImageID>.jpg/.png")
    p.add_argument("--oi_subset", type=str, default="", help="Text file with ImageIDs, one per line")
    p.add_argument("--oi_bbox_csv", type=str, default="", help="Open Images bbox CSV")
    p.add_argument("--oi_classes", type=str, default="", help="class-descriptions-boxable.csv")
    p.add_argument("--oi_rel_csv", type=str, default="", help="Open Images visual relationships CSV")
    p.add_argument("--oi_keep_groupof", action="store_true")
    p.add_argument("--save_outputs", type=str, default="")

    # Stronger-generator result export and uncertainty analysis
    p.add_argument(
        "--experiment_csv",
        type=str,
        default="strong_relation_generator_per_image.csv",
    )
    p.add_argument(
        "--experiment_json",
        type=str,
        default="strong_relation_generator_summary.json",
    )
    p.add_argument("--stats_seed", type=int, default=42)
    p.add_argument("--permutation_samples", type=int, default=20000)
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "single":
        if not args.image:
            raise SystemExit("--image is required in single mode")
        run_single(args)
    elif args.mode == "eval_coco":
        if not args.coco_root or not args.coco_ann:
            raise SystemExit("--coco_root and --coco_ann are required in eval_coco mode")
        run_eval_coco(args)
    elif args.mode in ("eval_openimages", "eval_strong_rel"):
        needed = [args.oi_root, args.oi_subset, args.oi_bbox_csv, args.oi_classes]
        if any(not x for x in needed):
            raise SystemExit(
                "--oi_root --oi_subset --oi_bbox_csv --oi_classes are required"
            )
        if args.mode == "eval_openimages":
            run_eval_openimages(args)
        else:
            run_eval_strong_relation(args)


if __name__ == "__main__":
    main()
