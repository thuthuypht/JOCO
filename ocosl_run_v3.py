#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCOSL runner v3:
- Implements Algorithm 1 (ILP + fallback) and Algorithm 2 (enrichment)
- Adds:
  (1) --tau_rel relation candidate pruning
  (2) --rel_budget global ILP sparsity constraint sum(y) <= C (0 => auto 2*n)
  (3) --iou_dup IoU-based duplicate box filtering (same label)
- Modes:
  single, eval_coco, eval_openimages
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np

from ocosl_utils_v3 import (
    DetObj, RelCand,
    load_ontology_schema, build_relation_type_constraints, build_synonym_table,
    spatial_relations, build_ontology_score, DEFAULT_TYPE_MAP,
    solve_ilp, greedy_repair, enrich_graph, draw_labeled_image,
    eval_bbox_label_accuracy, normalize_label,
    load_openimages_class_map, load_openimages_boxes_for_subset, load_openimages_rel_for_subset,
    eval_openimages_relations
)

# -------- YOLOv8 --------
def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Please install ultralytics: python -m pip install ultralytics") from e
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

def dedup_detections(dets, iou_thr=0.6):
    """
    Keep higher-confidence detection when same label overlaps with IoU > iou_thr.
    dets: list of (bbox_xywh, label, conf)
    """
    dets = sorted(dets, key=lambda x: x[2], reverse=True)
    kept = []
    for bbox, label, conf in dets:
        dup = False
        for kb, kl, kc in kept:
            if label == kl and iou_xywh(bbox, kb) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append((bbox, label, conf))
    return kept

def build_candidates(det_label: str, det_conf: float, topk: int, synonym_map: dict):
    """
    Practical top-k for YOLO-only output:
    - include detector label
    - include canonical synonym (if exists)
    - include generic fallback
    """
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
    for lbl, s in sorted(cands, key=lambda z: z[1], reverse=True):
        if lbl not in seen:
            uniq.append((lbl, float(s)))
            seen.add(lbl)
        if len(uniq) >= topk:
            break
    return uniq

def build_problem_for_image(image_path: str, model, args, synonym_map, type_map, rel_constraints):
    dets = yolo_detect(model, image_path, conf=args.det_conf, max_det=args.max_det)
    # (3) IoU-based duplicate-box filter for same label
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
                rel_support=0.0
            )
        objects.append(DetObj(idx=i, bbox=bbox, det_label=det_label, det_conf=det_conf,
                              candidates=candidates, onto_scores=onto_scores))

    # Relation candidates (4.3) with (1) pruning by tau_rel
    rel_cands = []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            for rel, score, feasible in spatial_relations(objects[i].bbox, objects[j].bbox, near_px=args.near_px):
                score = float(score)
                feasible = bool(feasible)
                if (not feasible) or (score < args.tau_rel):
                    continue
                rel_cands.append(RelCand(i=i, j=j, rel=rel, score=score, feasible=True))

    return objects, rel_cands

def solve_problem(objects, rel_cands, args, rel_constraints, synonym_map, type_map):
    # (2) Global relation budget sum(y) <= C
    budget = args.rel_budget if args.rel_budget > 0 else max(1, 2 * len(objects))

    t0 = time.time()
    try:
        x_star, y_star, status, used_fallback = solve_ilp(
            objects, rel_cands, rel_constraints, synonym_map, type_map,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            time_limit_s=args.time_limit,
            rel_budget=budget
        )
    except Exception:
        x_star, y_star = greedy_repair(objects, rel_cands, rel_constraints, synonym_map, type_map)
        status, used_fallback = "FallbackGreedy", True
    dt = time.time() - t0

    sel_label, enriched_rels = enrich_graph(objects, x_star, y_star)
    return x_star, y_star, sel_label, enriched_rels, status, used_fallback, dt

def run_single(args):
    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    model = load_yolo(args.yolo_model)

    objects, rel_cands = build_problem_for_image(args.image, model, args, synonym_map, type_map, rel_constraints)
    if not objects:
        print("No detections.")
        return

    _, _, sel_label, enriched_rels, status, used_fallback, dt = solve_problem(objects, rel_cands, args, rel_constraints, synonym_map, type_map)

    out_path = Path(args.output) if args.output else Path(args.image).with_suffix("").with_name(Path(args.image).stem + "_ocosl.png")
    draw_labeled_image(args.image, str(out_path), objects, sel_label)

    print(f"Status: {status} | fallback={used_fallback} | time={dt:.3f}s")
    print("Saved:", out_path)
    print("Selected labels:", sel_label)
    print("Enriched relations (first 20):", enriched_rels[:20])

# -------- COCO evaluation (lightweight) --------
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

    with open(args.coco_ann, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = coco["images"][:args.max_images]

    metrics = []
    for img in images:
        image_id = int(img["id"])
        file_name = img["file_name"]
        img_path = Path(args.coco_root) / file_name
        if not img_path.exists():
            continue

        objects, rel_cands = build_problem_for_image(str(img_path), model, args, synonym_map, type_map, rel_constraints)
        _, _, sel_label, enriched_rels, _, _, _ = solve_problem(objects, rel_cands, args, rel_constraints, synonym_map, type_map)

        gt_boxes, gt_labels = load_coco_gt(args.coco_ann, image_id)
        m = eval_bbox_label_accuracy(objects, sel_label, gt_boxes, gt_labels, iou_thr=args.iou_thr)
        metrics.append(m)

    if not metrics:
        print("No evaluated images. Check coco_root/coco_ann paths.")
        return
    avg = {k: float(np.mean([m[k] for m in metrics])) for k in ["precision", "recall", "f1", "mean_iou"]}
    print("COCO evaluation over", len(metrics), "images")
    print(avg)

# -------- Open Images subset evaluation --------
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

def run_eval_openimages(args):
    from PIL import Image

    g, base = load_ontology_schema(args.ontology)
    rel_constraints = build_relation_type_constraints(g, base)
    synonym_map = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    model = load_yolo(args.yolo_model)

    subset_ids = load_subset_ids(args.oi_subset, max_n=args.max_images)
    if not subset_ids:
        raise SystemExit("Open Images subset is empty. Provide --oi_subset with image IDs (one per line).")

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
            w, h = im.size
        image_sizes[iid] = (w, h)

    subset_ids = set(image_sizes.keys())
    if not subset_ids:
        raise SystemExit("No images found for the given IDs. Check --oi_root and file extensions.")

    mid_to_name = load_openimages_class_map(args.oi_classes)
    gt_boxes = load_openimages_boxes_for_subset(args.oi_bbox_csv, subset_ids, mid_to_name, image_sizes, keep_groupof=args.oi_keep_groupof)

    gt_rels = None
    if args.oi_rel_csv:
        gt_rels = load_openimages_rel_for_subset(args.oi_rel_csv, subset_ids, mid_to_name, image_sizes)

    label_metrics = []
    rel_metrics = []

    for iid in sorted(subset_ids):
        img_path = image_paths[iid]
        objects, rel_cands = build_problem_for_image(img_path, model, args, synonym_map, type_map, rel_constraints)
        _, _, sel_label, enriched_rels, _, _, _ = solve_problem(objects, rel_cands, args, rel_constraints, synonym_map, type_map)

        gt_list = gt_boxes.get(iid, [])
        gt_b = [b for b, _ in gt_list]
        gt_l = [l for _, l in gt_list]

        m = eval_bbox_label_accuracy(objects, sel_label, gt_b, gt_l, iou_thr=args.iou_thr)
        label_metrics.append(m)

        if gt_rels is not None:
            rm = eval_openimages_relations(objects, enriched_rels, gt_rels.get(iid, []), iou_thr=args.iou_thr)
            rel_metrics.append(rm)

        if args.save_outputs:
            out_img = Path(args.save_outputs) / f"{iid}_ocosl.png"
            draw_labeled_image(img_path, str(out_img), objects, sel_label)

    avg_lbl = {k: float(np.mean([m[k] for m in label_metrics])) for k in ["precision", "recall", "f1", "mean_iou"]}
    print("Open Images subset label evaluation over", len(label_metrics), "images")
    print(avg_lbl)

    if rel_metrics:
        avg_rel = {k: float(np.mean([m[k] for m in rel_metrics])) for k in ["rel_precision", "rel_recall", "rel_f1"]}
        avg_rel["avg_gt_rels"] = float(np.mean([m["gt_rels"] for m in rel_metrics]))
        avg_rel["avg_pred_rels"] = float(np.mean([m["pred_rels"] for m in rel_metrics]))
        print("Open Images subset relation evaluation (object-object only; skips RelationLabel='is'):")
        print(avg_rel)
    elif args.oi_rel_csv:
        print("Relation CSV provided but no object-object relations were evaluated (possibly only 'is' attributes).")

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ontology", type=str, default="v6_ocosl_ontology_schema_openimages.owl")
    p.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--det_conf", type=float, default=0.25)
    p.add_argument("--max_det", type=int, default=30)
    p.add_argument("--near_px", type=float, default=80.0)

    # NEW: relation pruning + sparsity + duplicate filtering
    p.add_argument("--tau_rel", type=float, default=0.6, help="Relation score threshold for candidate pruning")
    p.add_argument("--rel_budget", type=int, default=0,
                   help="Global relation budget C for ILP: sum(y) <= C (0 disables; auto uses 2*n_objects)")
    p.add_argument("--iou_dup", type=float, default=0.6,
                   help="IoU threshold for removing duplicate detections of the same label")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--time_limit", type=int, default=10)

    p.add_argument("--mode", choices=["single", "eval_coco", "eval_openimages"], default="single")
    p.add_argument("--image", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--max_images", type=int, default=50)
    p.add_argument("--iou_thr", type=float, default=0.5)

    # COCO
    p.add_argument("--coco_root", type=str, default="")
    p.add_argument("--coco_ann", type=str, default="")

    # Open Images subset
    p.add_argument("--oi_root", type=str, default="", help="Folder containing <ImageID>.jpg/.png")
    p.add_argument("--oi_subset", type=str, default="", help="Text file with ImageIDs (one per line)")
    p.add_argument("--oi_bbox_csv", type=str, default="", help="Open Images bbox CSV")
    p.add_argument("--oi_classes", type=str, default="", help="class-descriptions-boxable.csv (MID->name)")
    p.add_argument("--oi_rel_csv", type=str, default="", help="Optional visual relationships CSV")
    p.add_argument("--oi_keep_groupof", action="store_true", help="Keep IsGroupOf boxes")
    p.add_argument("--save_outputs", type=str, default="", help="Optional folder to save labeled images")

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
    elif args.mode == "eval_openimages":
        needed = [args.oi_root, args.oi_subset, args.oi_bbox_csv, args.oi_classes]
        if any(not x for x in needed):
            raise SystemExit("--oi_root --oi_subset --oi_bbox_csv --oi_classes are required for eval_openimages")
        run_eval_openimages(args)

if __name__ == "__main__":
    main()
