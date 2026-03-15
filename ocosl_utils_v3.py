#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for OCOSL (Ontology-Constrained Combinatorial Optimization for Semantic Image Labeling).
v3:
- Pillow>=10 compatibility (textbbox)
- ILP solver with optional global relation budget: sum(y) <= C
- Domain/range constraint handling: treat 'DetectedObject' as unconstrained
- Open Images subset helpers (bbox + optional VRD relations)
"""

from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
except Exception as e:
    raise RuntimeError("Please install rdflib (pip install rdflib)") from e

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


# ---------------- Data structures ----------------
@dataclass
class DetObj:
    idx: int
    bbox: Tuple[float, float, float, float]  # x,y,w,h in pixels
    det_label: str
    det_conf: float
    candidates: List[Tuple[str, float]]  # (label, s_det)
    onto_scores: Dict[str, float]        # label -> s_onto


@dataclass
class RelCand:
    i: int
    j: int
    rel: str
    score: float
    feasible: bool


# ---------------- Ontology helpers ----------------
def load_ontology_schema(path: str):
    g = Graph()
    g.parse(path, format="xml")

    ont_iri = None
    for s in g.subjects(RDF.type, OWL.Ontology):
        ont_iri = str(s)
        break

    base = Namespace((ont_iri + "#") if ont_iri and not ont_iri.endswith("#") else (ont_iri or "http://example.org/coco-ontology#"))
    g.bind("base", base)
    return g, base


def _localname(uri: URIRef) -> str:
    s = str(uri)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]


def build_relation_type_constraints(g: Graph, base: Namespace):
    """
    Read rdfs:domain/rdfs:range constraints from ontology schema.
    Returns dict: rel_localname -> (domain_localname, range_localname)
    """
    rel_constraints: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for rel in g.subjects(RDF.type, OWL.ObjectProperty):
        if not str(rel).startswith(str(base)):
            continue
        dn = next(g.objects(rel, RDFS.domain), None)
        rn = next(g.objects(rel, RDFS.range), None)
        if dn is None and rn is None:
            continue
        rel_constraints[_localname(rel)] = (_localname(dn) if dn else None, _localname(rn) if rn else None)
    return rel_constraints


def build_synonym_table(g: Graph):
    """
    Build synonym table from skos:altLabel annotations.
    Returns dict: alt_label_lower -> canonical_label_lower
    """
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    alt_to_canon: Dict[str, str] = {}
    for s, alt in g.subject_objects(SKOS.altLabel):
        canon = next(g.objects(s, RDFS.label), None)
        if canon is None:
            continue
        alt_to_canon[str(alt).lower()] = str(canon).lower()
    return alt_to_canon


# ---------------- Label mapping / scoring ----------------
def normalize_label(lbl: str) -> str:
    return lbl.strip().lower().replace("_", " ")


def lexical_score(a: str, b: str) -> float:
    a = normalize_label(a)
    b = normalize_label(b)
    if a == b:
        return 1.0
    if fuzz is not None:
        return fuzz.token_set_ratio(a, b) / 100.0
    # fallback
    return 0.6 if (a in b or b in a) else 0.0


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


def bbox_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def _horiz_overlap_ratio(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    a1, a2 = x1, x1 + w1
    b1_, b2_ = x2, x2 + w2
    inter = max(0.0, min(a2, b2_) - max(a1, b1_))
    denom = min(w1, w2) if min(w1, w2) > 0 else 1.0
    return inter / denom


def spatial_relations(i_bbox, j_bbox, near_px=80.0):
    """
    Generate spatial relation candidates from bbox geometry.
    Returns list of (relation, score, feasible_flag).
    """
    ix, iy, iw, ih = i_bbox
    jx, jy, jw, jh = j_bbox
    icx, icy = bbox_center(i_bbox)
    jcx, jcy = bbox_center(j_bbox)

    rels = []

    # above/below
    if icy + ih * 0.10 < jcy - jh * 0.10:
        rels.append(("isAbove", 0.8, True))
    elif icy - ih * 0.10 > jcy + jh * 0.10:
        rels.append(("isBelow", 0.8, True))

    # left/right
    if icx < jcx - jw * 0.1:
        rels.append(("leftOf", 0.7, True))
    elif icx > jcx + jw * 0.1:
        rels.append(("rightOf", 0.7, True))

    # overlap
    ov = iou_xywh(i_bbox, j_bbox)
    if ov > 0.2:
        rels.append(("overlaps", min(1.0, ov + 0.3), True))

    # containment (rough)
    if ix <= jx and iy <= jy and (ix + iw) >= (jx + jw) and (iy + ih) >= (jy + jh):
        rels.append(("contains", 0.7, True))
    if jx <= ix and jy <= iy and (jx + jw) >= (ix + iw) and (jy + jh) >= (iy + ih):
        rels.append(("insideOf", 0.7, True))

    # isOn / supports (approx): i bottom near j top + horizontal overlap
    i_bottom = iy + ih
    j_top = jy
    horiz = _horiz_overlap_ratio(i_bbox, j_bbox)
    if horiz > 0.2 and abs(i_bottom - j_top) < max(10.0, 0.06 * (ih + jh)):
        rels.append(("isOn", 0.75 + 0.2 * horiz, True))
        rels.append(("supports", 0.40 + 0.2 * horiz, True))

    # near
    dist = math.hypot(icx - jcx, icy - jcy)
    if dist < near_px:
        rels.append(("near", 1.0 - dist / near_px, True))

    return rels


# A small, extensible type map for domain-range constraints.
DEFAULT_TYPE_MAP = {
    "person": "Person",
    "man": "Person",
    "woman": "Person",
    "boy": "Person",
    "girl": "Person",
    "dog": "Animal",
    "cat": "Animal",
    "horse": "Animal",
    "car": "Vehicle",
    "bus": "Vehicle",
    "truck": "Vehicle",
    "bicycle": "Vehicle",
    "motorcycle": "Vehicle",
    "tie": "Clothing",
    "shirt": "Clothing",
    "jacket": "Clothing",
    "pants": "Clothing",
    "dress": "Clothing",
    "shoe": "Clothing",
    "sofa": "Furniture",
    "couch": "Furniture",
    "chair": "Furniture",
    "bed": "Furniture",
    "table": "Furniture",
    "tv": "ElectronicDevice",
    "laptop": "ElectronicDevice",
    "cell phone": "ElectronicDevice",
    "phone": "ElectronicDevice",
    "bowl": "Container",
    "cup": "Container",
    "bottle": "Container",
}


def label_to_type(label: str, synonym_map: Dict[str, str], type_map: Dict[str, str]):
    l = normalize_label(label)
    if l in synonym_map:
        l = synonym_map[l]
    return type_map.get(l, "PhysicalObject")


def build_ontology_score(det_score: float,
                         det_label: str,
                         cand_label: str,
                         synonym_map: Dict[str, str],
                         type_map: Dict[str, str],
                         rel_support: float = 0.0,
                         w=(0.55, 0.25, 0.15, 0.05)):
    """
    Practical s_onto:
    - detector evidence
    - lexical mapping to canonical concept
    - type/hierarchy preference (coarse proxy)
    - relation support
    """
    lex = lexical_score(det_label, cand_label)
    t_det = label_to_type(det_label, synonym_map, type_map)
    t_cand = label_to_type(cand_label, synonym_map, type_map)
    hier = 1.0 if t_det == t_cand else 0.4
    w1, w2, w3, w4 = w
    return w1 * det_score + w2 * lex + w3 * hier + w4 * rel_support


# ---------------- Optimization (ILP + fallback) ----------------
def solve_ilp(objects: List[DetObj],
              rel_cands: List[RelCand],
              rel_type_constraints: Dict[str, Tuple[Optional[str], Optional[str]]],
              synonym_map: Dict[str, str],
              type_map: Dict[str, str],
              alpha=1.0, beta=1.0, gamma=0.5,
              time_limit_s: int = 10,
              rel_budget: int = 0):
    """
    Exact ILP for Algorithm 1.
    rel_budget: if >0, enforce sum(y) <= rel_budget.
    """
    try:
        import pulp
    except Exception as e:
        raise RuntimeError("PuLP is required (pip install pulp)") from e

    prob = pulp.LpProblem("OCOSL_Labeling", pulp.LpMaximize)

    # x_{i,l}
    x = {}
    for o in objects:
        for lbl, _ in o.candidates:
            x[(o.idx, lbl)] = pulp.LpVariable(f"x_{o.idx}_{lbl.replace(' ', '_')}", 0, 1, cat=pulp.LpBinary)

    # One-label-per-object
    for o in objects:
        prob += pulp.lpSum([x[(o.idx, lbl)] for lbl, _ in o.candidates]) == 1, f"one_label_{o.idx}"

    # y_{ijr}
    y = {}
    for rc in rel_cands:
        if not rc.feasible:
            continue
        key = (rc.i, rc.j, rc.rel)
        y[key] = pulp.LpVariable(f"y_{rc.i}_{rc.j}_{rc.rel}", 0, 1, cat=pulp.LpBinary)

    # Global relation budget sum(y) <= C
    if rel_budget and len(y) > 0:
        prob += pulp.lpSum([v for v in y.values()]) <= int(rel_budget), "global_relation_budget"

    # At most one relation per pair
    pairs = {}
    for (i, j, r) in y.keys():
        pairs.setdefault((i, j), []).append((i, j, r))
    for (i, j), keys in pairs.items():
        prob += pulp.lpSum([y[k] for k in keys]) <= 1, f"one_rel_{i}_{j}"

    # Domain-range constraints
    # NOTE: spatial relations often have generic domain/range like 'DetectedObject'.
    # We treat 'DetectedObject' as unconstrained because our type map uses Person/Vehicle/.../PhysicalObject.
    for (i, j, r), var in y.items():
        dom, ran = rel_type_constraints.get(r, (None, None))

        if dom in (None, "DetectedObject"):
            dom = None
        if ran in (None, "DetectedObject"):
            ran = None

        if dom:
            prob += var <= pulp.lpSum([x[(i, lbl)] for lbl, _ in objects[i].candidates
                                       if label_to_type(lbl, synonym_map, type_map) == dom]), f"dom_{i}_{j}_{r}"
        if ran:
            prob += var <= pulp.lpSum([x[(j, lbl)] for lbl, _ in objects[j].candidates
                                       if (ran == "PhysicalObject") or (label_to_type(lbl, synonym_map, type_map) == ran)]), f"ran_{i}_{j}_{r}"

    # Objective
    obj_terms = []
    for o in objects:
        for lbl, sdet in o.candidates:
            sonto = o.onto_scores.get(lbl, sdet)
            obj_terms.append((alpha * sdet + beta * sonto) * x[(o.idx, lbl)])

    rel_score_map = {(rc.i, rc.j, rc.rel): rc.score for rc in rel_cands if rc.feasible}
    for (i, j, r), var in y.items():
        obj_terms.append(gamma * rel_score_map.get((i, j, r), 0.0) * var)

    prob += pulp.lpSum(obj_terms)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)
    status_str = pulp.LpStatus.get(prob.status, str(prob.status))

    x_star = {(i, lbl): int(pulp.value(var) > 0.5) for (i, lbl), var in x.items()}
    y_star = {(i, j, r): int(pulp.value(var) > 0.5) for (i, j, r), var in y.items()}
    used_fallback = status_str != "Optimal"
    return x_star, y_star, status_str, used_fallback


def greedy_repair(objects: List[DetObj],
                  rel_cands: List[RelCand],
                  rel_type_constraints: Dict[str, Tuple[Optional[str], Optional[str]]],
                  synonym_map: Dict[str, str],
                  type_map: Dict[str, str]):
    """
    Fallback solution:
    - pick best label per object
    - add best feasible relation per pair (optional)
    """
    x_star = {}
    chosen = {}
    for o in objects:
        best_lbl = None
        best_val = -1e9
        for lbl, sdet in o.candidates:
            val = sdet + o.onto_scores.get(lbl, 0.0)
            if val > best_val:
                best_val = val
                best_lbl = lbl
        chosen[o.idx] = best_lbl
        for lbl, _ in o.candidates:
            x_star[(o.idx, lbl)] = 1 if lbl == best_lbl else 0

    y_star = {}
    seen_pairs = set()
    for rc in sorted([r for r in rel_cands if r.feasible], key=lambda z: z.score, reverse=True):
        if (rc.i, rc.j) in seen_pairs:
            continue
        dom, ran = rel_type_constraints.get(rc.rel, (None, None))
        li = chosen.get(rc.i)
        lj = chosen.get(rc.j)

        if dom and dom != "DetectedObject" and label_to_type(li, synonym_map, type_map) != dom:
            continue
        if ran and ran not in ("DetectedObject", "PhysicalObject") and label_to_type(lj, synonym_map, type_map) != ran:
            continue

        y_star[(rc.i, rc.j, rc.rel)] = 1
        seen_pairs.add((rc.i, rc.j))
    return x_star, y_star


# ---------------- Graph enrichment (Algorithm 2) ----------------
def enrich_graph(objects: List[DetObj],
                 x_star: Dict[Tuple[int, str], int],
                 y_star: Dict[Tuple[int, int, str], int]):
    """
    Lightweight enrichment:
    - add inverse spatial relations
    - add interactsWith for isHolding
    """
    sel_label: Dict[int, str] = {}
    for o in objects:
        for lbl, _ in o.candidates:
            if x_star.get((o.idx, lbl), 0) == 1:
                sel_label[o.idx] = lbl
                break

    rels = set((i, j, r) for (i, j, r), v in y_star.items() if v == 1)

    add = set()
    for (i, j, r) in list(rels):
        if r == "isAbove":
            add.add((j, i, "isBelow"))
        if r == "contains":
            add.add((j, i, "insideOf"))
        if r == "leftOf":
            add.add((j, i, "rightOf"))
        if r == "inFrontOf":
            add.add((j, i, "behind"))
        if r == "isOn":
            add.add((j, i, "supports"))
        if r == "isHolding":
            add.add((i, j, "interactsWith"))

    rels |= add
    return sel_label, sorted(rels)


# ---------------- Visualization ----------------
def draw_labeled_image(image_path: str,
                       out_path: str,
                       objects: List[DetObj],
                       sel_label: Dict[int, str]):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for o in objects:
        x, y, w, h = o.bbox
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], width=3)
        lbl = sel_label.get(o.idx, o.det_label)
        txt = f"{lbl} ({o.det_conf:.2f})"

        # Pillow>=10: use textbbox
        try:
            bb = draw.textbbox((0, 0), txt, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = font.getsize(txt)

        draw.rectangle([x, max(0, y - th - 6), x + tw + 6, y], fill=(255, 255, 255))
        draw.text((x + 3, max(0, y - th - 3)), txt, fill=(0, 0, 0), font=font)

    img.save(out_path)


# ---------------- Evaluation ----------------
def eval_bbox_label_accuracy(pred_objects: List[DetObj],
                             pred_labels: Dict[int, str],
                             gt_boxes: List[Tuple[float, float, float, float]],
                             gt_labels: List[str],
                             iou_thr=0.5):
    """
    Simple matching:
    - each predicted box matches the best GT box (greedy) if IoU>=thr
    - count correct label on matched pairs
    """
    used = set()
    tp = 0
    match_ious = []
    for o in pred_objects:
        best_j = -1
        best_iou = 0.0
        for j, gb in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou_xywh(o.bbox, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            match_ious.append(best_iou)
            if normalize_label(pred_labels.get(o.idx, "")) == normalize_label(gt_labels[best_j]):
                tp += 1
    fp = max(0, len(pred_objects) - tp)
    fn = max(0, len(gt_boxes) - tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    miou = float(np.mean(match_ious)) if match_ious else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mean_iou": miou, "tp": tp, "fp": fp, "fn": fn}


# --- Open Images helpers ---
def load_openimages_class_map(class_desc_csv: str) -> Dict[str, str]:
    """
    class descriptions file: LabelName,DisplayName
    """
    mid_to_name = {}
    with open(class_desc_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "LabelName":
                continue
            mid = row[0].strip()
            name = row[1].strip() if len(row) > 1 else mid
            mid_to_name[mid] = name
    return mid_to_name


def _norm01_to_xywh(xmin, xmax, ymin, ymax, w, h):
    x1 = float(xmin) * w
    x2 = float(xmax) * w
    y1 = float(ymin) * h
    y2 = float(ymax) * h
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def load_openimages_boxes_for_subset(bbox_csv: str,
                                     subset_image_ids: set,
                                     mid_to_name: Dict[str, str],
                                     image_sizes: Dict[str, Tuple[int, int]],
                                     keep_groupof: bool = False):
    """
    Return dict image_id -> list of (bbox_xywh_pixels, display_name)
    """
    gt = {iid: [] for iid in subset_image_ids}
    with open(bbox_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get("ImageID") or row.get("ImageId") or row.get("image_id")
            if iid not in subset_image_ids:
                continue
            if not keep_groupof and row.get("IsGroupOf", "0") == "1":
                continue
            mid = row["LabelName"]
            name = mid_to_name.get(mid, mid)
            w, h = image_sizes[iid]
            bbox = _norm01_to_xywh(row["XMin"], row["XMax"], row["YMin"], row["YMax"], w, h)
            gt[iid].append((bbox, name))
    return gt


def load_openimages_rel_for_subset(rel_csv: str,
                                   subset_image_ids: set,
                                   mid_to_name: Dict[str, str],
                                   image_sizes: Dict[str, Tuple[int, int]]):
    """
    Visual relationships CSV (Open Images format):
    ImageID,LabelName1,LabelName2,XMin1,XMax1,YMin1,YMax1,XMin2,XMax2,YMin2,YMax2,RelationLabel

    Returns dict image_id -> list of (bbox1, name1, relation, bbox2, name2)
    """
    rels = {iid: [] for iid in subset_image_ids}
    with open(rel_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["ImageID"]
            if iid not in subset_image_ids:
                continue
            w, h = image_sizes[iid]
            mid1 = row["LabelName1"]
            mid2 = row["LabelName2"]
            name1 = mid_to_name.get(mid1, mid1)
            name2 = mid_to_name.get(mid2, mid2)
            b1 = _norm01_to_xywh(row["XMin1"], row["XMax1"], row["YMin1"], row["YMax1"], w, h)
            b2 = _norm01_to_xywh(row["XMin2"], row["XMax2"], row["YMin2"], row["YMax2"], w, h)
            rel = row["RelationLabel"].strip().lower()
            rels[iid].append((b1, name1, rel, b2, name2))
    return rels


# Relation mapping between Open Images (VRD) and our internal relation names
OPENIMAGES_REL_MAP = {
    "on": "isOn",
    "under": "isBelow",
    "above": "isAbove",
    "below": "isBelow",
    "inside": "insideOf",
    "in": "insideOf",
    "contains": "contains",
    "over": "isAbove",
    "behind": "behind",
    "in front of": "inFrontOf",
    "at": "near",
    "near": "near",
    "next to": "near",
    "intersect": "overlaps",
    "overlaps": "overlaps",
}


def match_predictions_to_gt(pred_objects: List[DetObj],
                            gt_boxes: List[Tuple[float, float, float, float]],
                            iou_thr=0.5):
    """
    One-to-one greedy matching: pred -> gt index, and gt -> pred index.
    """
    pred_to_gt = {}
    gt_used = set()
    for o in pred_objects:
        best_j, best_iou = -1, 0.0
        for j, gb in enumerate(gt_boxes):
            if j in gt_used:
                continue
            v = iou_xywh(o.bbox, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            pred_to_gt[o.idx] = best_j
            gt_used.add(best_j)
    gt_to_pred = {g: p for p, g in pred_to_gt.items()}
    return pred_to_gt, gt_to_pred


def eval_openimages_relations(pred_objects: List[DetObj],
                              pred_rels: List[Tuple[int, int, str]],
                              gt_rel_list: List[Tuple[Tuple[float, float, float, float], str, str, Tuple[float, float, float, float], str]],
                              iou_thr=0.5):
    """
    Evaluate object-object relations only (skip RelationLabel='is' attributes).
    """
    # Build GT object list from relations (deduplicate boxes)
    gt_boxes = []
    box_key_to_idx = {}

    def key(b):
        return tuple(round(float(x), 4) for x in b)

    rel_gt_trip = []
    for b1, n1, rel, b2, n2 in gt_rel_list:
        if rel == "is":
            continue
        k1, k2 = key(b1), key(b2)
        if k1 not in box_key_to_idx:
            box_key_to_idx[k1] = len(gt_boxes)
            gt_boxes.append(b1)
        if k2 not in box_key_to_idx:
            box_key_to_idx[k2] = len(gt_boxes)
            gt_boxes.append(b2)
        rel_gt_trip.append((box_key_to_idx[k1], rel, box_key_to_idx[k2]))

    if not rel_gt_trip:
        return {"rel_precision": 0.0, "rel_recall": 0.0, "rel_f1": 0.0, "gt_rels": 0, "pred_rels": len(pred_rels)}

    pred_to_gt, _ = match_predictions_to_gt(pred_objects, gt_boxes, iou_thr=iou_thr)

    # Predicted set in GT-index space
    pred_set = set()
    for i, j, r in pred_rels:
        if i in pred_to_gt and j in pred_to_gt:
            pred_set.add((pred_to_gt[i], r, pred_to_gt[j]))

    # GT set mapped to internal relation names
    gt_set = set()
    for gi, rel, gj in rel_gt_trip:
        rel_norm = rel.strip().lower()
        internal = OPENIMAGES_REL_MAP.get(rel_norm, None)
        if internal:
            gt_set.add((gi, internal, gj))

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"rel_precision": prec, "rel_recall": rec, "rel_f1": f1, "gt_rels": len(gt_set), "pred_rels": len(pred_set)}
