#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JOCO Open Images 300: OI-trained RelTR vs hierarchy-aware OCOSL V3
===================================================================

Reproducibility runner for the controlled stronger-relation-generator experiment.

Key design choices
------------------
* Official Open-Images-trained RelTR checkpoint.
* RelTR's own subject/object boxes are used to form the scene graph.
* RelTR-only and RelTR+OCOSL use exactly the same learned candidate pool.
* Open Images relation evaluation uses subject IoU >= 0.50 and object IoU >= 0.50.
* Open Images object labels are canonicalized before ontology candidate construction.
* OCOSL V3 evaluates domain/range compatibility using transitive rdfs:subClassOf
  closure rather than flat label_to_type equality.
* No ontology, threshold, weight, relation-budget, or evaluation-IoU tuning is
  performed relative to the final reported V3 experiment.

Expected reference result on the released 300-image subset
-----------------------------------------------------------
RelTR-only shared pool:
  Precision 0.4615384615
  Recall    0.4567474048
  F1        0.4591304348

RelTR + hierarchy-aware OCOSL V3:
  Precision 0.4614485981
  Recall    0.4555940023
  F1        0.4585026117

micro F1 gain = -0.0006278231
paired permutation p = 1.0
per-image improved/equal/worse = 1 / 298 / 1
fallback count = 0

This script expects ocosl_run_v4.py and ocosl_utils_v3.py from the released
implementation. It can auto-locate Kaggle inputs and optionally prepare the
official RelTR repository/checkpoint/metadata.

Example on Kaggle
-----------------
python run_oi300_reltr_ocosl_v3.py \
  --prepare-reltr \
  --output-dir /kaggle/working/JOCO/results_v3 \
  --n-images 300
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from rdflib import RDF, RDFS, OWL, URIRef

# ---------------------------------------------------------------------
# Fixed experiment defaults
# ---------------------------------------------------------------------

OI_CKPT_GDRIVE_ID = "1pcoUnR0XWsvM9lJZ5f93N5TKHkLdjtnb"
OI_META_GDRIVE_ID = "1kWeG3O071Bx17KI7oLbMdgGvE5xmyY8k"

REFERENCE = {
    "n_completed": 300,
    "reltr_f1": 0.4591304347826087,
    "ocosl_f1": 0.45850261172373763,
    "micro_f1_gain": -0.0006278230588710465,
    "paired_permutation_pvalue": 1.0,
    "improved": 1,
    "equal": 298,
    "worse": 1,
}

OI_OBJECT_CANONICAL_MAP = {
    # Person
    "man": "Person",
    "woman": "Person",
    "boy": "Person",
    "girl": "Person",
    "person": "Person",

    # Eyewear
    "glasses": "Glasses",
    "sunglasses": "Glasses",
    "goggles": "Glasses",

    # Helmets
    "football helmet": "Helmet",
    "bicycle helmet": "Helmet",

    # Footwear
    "high heels": "Shoe",
    "sandal": "Shoe",
    "roller skates": "Shoe",
    "boot": "Boots",

    # Headwear
    "cowboy hat": "Hat",
    "fedora": "Hat",
    "sun hat": "Hat",
    "swim cap": "Hat",
    "crown": "Hat",
    "tiara": "Hat",

    # Other wearables
    "necklace": "Necklace",
    "scarf": "Scarf",
    "belt": "Belt",
    "baseball glove": "Glove",

    # Rideable vehicles
    "boat": "Boat",
    "wheelchair": "Wheelchair",
    "canoe": "Boat",
    "cart": "Vehicle",
    "segway": "Vehicle",
}

# Deliberately not forced:
#   Horse -> Vehicle
#   Handbag -> Clothing

OI_TO_OCOSL_REL = {
    "at": ("near", False),
    "holds": ("isHolding", False),
    "holding": ("isHolding", False),
    "on": ("isOn", False),
    "under": ("isBelow", False),
    "wears": ("wearing", False),
    "wearing": ("wearing", False),
    "ride": ("riding", False),
    "riding": ("riding", False),
    "inside_of": ("insideOf", False),
    "insideof": ("insideOf", False),
    "interacts_with": ("interactsWith", False),
    "interactswith": ("interactsWith", False),
    "contain": ("insideOf", True),
    "contains": ("insideOf", True),
}


def norm_label(s: object) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def local_name(uri: object) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    return s.rsplit("/", 1)[-1]


def canonicalize_oi_label(label: str) -> str:
    return OI_OBJECT_CANONICAL_MAP.get(str(label).strip().lower(), label)


def map_relation_to_ontology(raw_rel: str):
    key = norm_label(raw_rel)
    if key == "is":
        return None
    return OI_TO_OCOSL_REL.get(key)


def xyxy_to_xywh(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_one(root: Path, name: str) -> Optional[Path]:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def locate_oi_root(explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not (p / "oi300_ids.txt").exists():
            raise FileNotFoundError(f"oi300_ids.txt not found under {p}")
        return p
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hit = find_one(kaggle, "oi300_ids.txt")
        if hit:
            return hit.parent
    raise FileNotFoundError(
        "Could not auto-locate OI-300. Pass --oi-root pointing to the directory "
        "containing oi300_ids.txt, images/, and openimages_relationships.csv."
    )


def locate_code_dir(explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not (p / "ocosl_run_v4.py").exists():
            raise FileNotFoundError(f"ocosl_run_v4.py not found under {p}")
        return p
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hit = find_one(kaggle, "ocosl_run_v4.py")
        if hit:
            return hit.parent
    here = Path(__file__).resolve().parent
    if (here / "ocosl_run_v4.py").exists():
        return here
    raise FileNotFoundError(
        "Could not auto-locate ocosl_run_v4.py. Pass --code-dir."
    )


def locate_ontology(code_dir: Path, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        code_dir / "JOCO_verified_ontology_schema_fixed.owl",
        code_dir / "JOCO_verified_ontology_schema.owl",
    ]
    for p in candidates:
        if p.exists():
            return p
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hit = find_one(kaggle, "JOCO_verified_ontology_schema_fixed.owl")
        if hit:
            return hit
    raise FileNotFoundError("Could not locate the JOCO ontology. Pass --ontology.")


def ensure_package(package: str):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
        return importlib.import_module(package)


def prepare_reltr_assets(
    repo: Path,
    ckpt: Path,
    metadata_dir: Path,
    do_prepare: bool,
):
    if not (repo / "inference.py").exists():
        if not do_prepare:
            raise FileNotFoundError(
                f"RelTR repo missing at {repo}. Re-run with --prepare-reltr or pass --reltr-repo."
            )
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yrcong/RelTR.git", str(repo)],
            check=True,
        )

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists() or ckpt.stat().st_size < 1_000_000:
        if not do_prepare:
            raise FileNotFoundError(
                f"OI RelTR checkpoint missing at {ckpt}. Use --prepare-reltr."
            )
        gdown = ensure_package("gdown")
        gdown.download(id=OI_CKPT_GDRIVE_ID, output=str(ckpt), quiet=False)

    metadata_dir.mkdir(parents=True, exist_ok=True)

    def have_metadata() -> bool:
        return bool(list(metadata_dir.rglob("categories_dict.json"))) or (
            bool(list(metadata_dir.rglob("rel.json")))
            and bool(list(metadata_dir.rglob("train.json")))
        )

    if not have_metadata():
        if not do_prepare:
            raise FileNotFoundError(
                f"RelTR OI metadata missing under {metadata_dir}. Use --prepare-reltr."
            )
        gdown = ensure_package("gdown")
        archive = metadata_dir / "oi_annotations.zip"
        gdown.download(id=OI_META_GDRIVE_ID, output=str(archive), quiet=False)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, "r") as z:
                z.extractall(metadata_dir)
        else:
            shutil.unpack_archive(str(archive), str(metadata_dir))


def load_oi_vocab(metadata_dir: Path) -> Tuple[List[str], List[str]]:
    cats = list(metadata_dir.rglob("categories_dict.json"))
    if cats:
        obj = json.loads(cats[0].read_text(encoding="utf-8"))
        return [str(x) for x in obj["obj"]], [str(x) for x in obj["rel"]]

    reljs = list(metadata_dir.rglob("rel.json"))
    trainjs = list(metadata_dir.rglob("train.json"))
    if reljs and trainjs:
        rel_data = json.loads(reljs[0].read_text(encoding="utf-8"))
        rel_labels = [str(x) for x in rel_data["rel_categories"]]
        train_data = json.loads(trainjs[0].read_text(encoding="utf-8"))
        categories = sorted(train_data["categories"], key=lambda x: int(x["id"]))
        obj_labels = [str(x["name"]) for x in categories]
        return obj_labels, rel_labels

    raise RuntimeError("Could not derive OI object/relation vocabularies from RelTR metadata.")


def fit_vocab(vocab: List[str], required_n: int, name: str) -> List[str]:
    vocab = list(vocab)
    if len(vocab) == required_n:
        return vocab
    if len(vocab) == required_n + 1:
        first = str(vocab[0]).lower()
        if "background" in first or first in {"n/a", "__background__"}:
            return vocab[1:]
    raise RuntimeError(
        f"{name} vocabulary mismatch: metadata={len(vocab)}, model={required_n}"
    )


def reltr_predict_full(adapter, image_path: Path, obj_labels: List[str], rel_labels: List[str],
                       query_thr: float, joint_thr: float, topk: int) -> List[dict]:
    import torch

    im = Image.open(image_path).convert("RGB")
    tensor = adapter.transform(im).unsqueeze(0).to(adapter.device)
    with torch.no_grad():
        outputs = adapter.model(tensor)

    rel_p = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    sub_p = outputs["sub_logits"].softmax(-1)[0, :, :-1]
    obj_p = outputs["obj_logits"].softmax(-1)[0, :, :-1]

    rel_score, rel_idx = rel_p.max(-1)
    sub_score, sub_idx = sub_p.max(-1)
    obj_score, obj_idx = obj_p.max(-1)
    joint = rel_score * sub_score * obj_score

    keep = (
        (rel_score >= query_thr)
        & (sub_score >= query_thr)
        & (obj_score >= query_thr)
        & (joint >= joint_thr)
    )
    qidx = torch.nonzero(keep, as_tuple=True)[0]
    if qidx.numel() == 0:
        return []

    order = torch.argsort(joint[qidx], descending=True)
    qidx = qidx[order[:topk]]

    sub_boxes = adapter._rescale_bboxes(outputs["sub_boxes"][0, qidx], im.size)
    obj_boxes = adapter._rescale_bboxes(outputs["obj_boxes"][0, qidx], im.size)

    preds = []
    for q, sb, ob in zip(qidx.tolist(), sub_boxes.tolist(), obj_boxes.tolist()):
        rid = int(rel_idx[q].item())
        sid = int(sub_idx[q].item())
        oid = int(obj_idx[q].item())
        if rid >= len(rel_labels) or sid >= len(obj_labels) or oid >= len(obj_labels):
            continue
        relation = str(rel_labels[rid])
        if norm_label(relation) == "is":
            continue
        preds.append(
            {
                "sub_box_xyxy": np.asarray(sb, dtype=float),
                "obj_box_xyxy": np.asarray(ob, dtype=float),
                "sub_label": str(obj_labels[sid]),
                "obj_label": str(obj_labels[oid]),
                "sub_conf": float(sub_score[q].item()),
                "obj_conf": float(obj_score[q].item()),
                "relation": relation,
                "score": float(joint[q].item()),
            }
        )
    return preds


def add_or_match_entity(
    entities: List[dict],
    box_xyxy: Sequence[float],
    label: str,
    confidence: float,
    merge_iou: float,
    normalize_fn,
) -> int:
    label_n = normalize_fn(label)
    best_idx = None
    best_iou = 0.0

    for idx, entity in enumerate(entities):
        val = iou_xyxy(box_xyxy, entity["box_xyxy"])
        same_label = normalize_fn(entity["label"]) == label_n
        if val >= merge_iou and (same_label or val >= 0.90) and val > best_iou:
            best_idx = idx
            best_iou = val

    if best_idx is None:
        entities.append(
            {
                "box_xyxy": np.asarray(box_xyxy, dtype=float),
                "label": label,
                "conf": float(confidence),
            }
        )
        return len(entities) - 1

    if confidence > entities[best_idx]["conf"]:
        entities[best_idx]["box_xyxy"] = np.asarray(box_xyxy, dtype=float)
        entities[best_idx]["label"] = label
        entities[best_idx]["conf"] = float(confidence)
    return best_idx


def build_scene_graph(
    oc,
    adapter,
    image_path: Path,
    obj_labels: List[str],
    rel_labels: List[str],
    query_thr: float,
    joint_thr: float,
    reltr_topk: int,
    entity_merge_iou: float,
    args,
    synonym_map: dict,
    type_map: dict,
):
    raw = reltr_predict_full(
        adapter, image_path, obj_labels, rel_labels,
        query_thr=query_thr, joint_thr=joint_thr, topk=reltr_topk,
    )

    entities: List[dict] = []
    candidate_meta: Dict[tuple, dict] = {}
    mapping_usage = Counter()

    for p in raw:
        mapping = map_relation_to_ontology(p["relation"])
        if mapping is None:
            continue
        canonical_rel, reverse = mapping

        original_sub = str(p["sub_label"])
        original_obj = str(p["obj_label"])
        canonical_sub = canonicalize_oi_label(original_sub)
        canonical_obj = canonicalize_oi_label(original_obj)

        if canonical_sub.strip().lower() != original_sub.strip().lower():
            mapping_usage[(original_sub, canonical_sub)] += 1
        if canonical_obj.strip().lower() != original_obj.strip().lower():
            mapping_usage[(original_obj, canonical_obj)] += 1

        s = add_or_match_entity(
            entities, p["sub_box_xyxy"], canonical_sub, p["sub_conf"],
            entity_merge_iou, oc.normalize_label,
        )
        o = add_or_match_entity(
            entities, p["obj_box_xyxy"], canonical_obj, p["obj_conf"],
            entity_merge_iou, oc.normalize_label,
        )
        if s == o:
            continue

        solver_i = o if reverse else s
        solver_j = s if reverse else o
        key = (solver_i, solver_j, canonical_rel)

        record = {
            "i": solver_i,
            "j": solver_j,
            "canonical_rel": canonical_rel,
            "eval_sub_box": p["sub_box_xyxy"],
            "eval_obj_box": p["obj_box_xyxy"],
            "eval_rel": norm_label(p["relation"]),
            "score": float(p["score"]),
            "original_sub_label": original_sub,
            "canonical_sub_label": canonical_sub,
            "original_obj_label": original_obj,
            "canonical_obj_label": canonical_obj,
        }

        if key not in candidate_meta or record["score"] > candidate_meta[key]["score"]:
            candidate_meta[key] = record

    objects = []
    for idx, e in enumerate(entities):
        bbox_xywh = xyxy_to_xywh(e["box_xyxy"])
        candidates = oc.build_candidates(e["label"], e["conf"], args.topk, synonym_map)
        onto_scores = {}
        for cand_label, sdet in candidates:
            onto_scores[cand_label] = oc.build_ontology_score(
                det_score=sdet,
                det_label=e["label"],
                cand_label=cand_label,
                synonym_map=synonym_map,
                type_map=type_map,
                rel_support=0.0,
            )
        objects.append(
            oc.DetObj(
                idx=idx,
                bbox=bbox_xywh,
                det_label=e["label"],
                det_conf=float(e["conf"]),
                candidates=candidates,
                onto_scores=onto_scores,
            )
        )

    rel_cands = [
        oc.RelCand(i=k[0], j=k[1], rel=k[2], score=v["score"], feasible=True)
        for k, v in sorted(candidate_meta.items(), key=lambda z: z[1]["score"], reverse=True)
    ]
    return raw, objects, rel_cands, candidate_meta, mapping_usage


def build_class_index(g) -> Dict[str, URIRef]:
    index: Dict[str, URIRef] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        index[norm_label(local_name(cls))] = cls
        for lbl in g.objects(cls, RDFS.label):
            index[norm_label(lbl)] = cls
    return index


def ontology_is_a(g, class_index: dict, child_label: str, required_type: str) -> bool:
    if required_type in (None, "DetectedObject"):
        return True

    child_uri = class_index.get(norm_label(child_label))
    req_uri = class_index.get(norm_label(required_type))

    # Preserve the released V3 experiment's generic PhysicalObject behavior.
    if required_type == "PhysicalObject":
        if child_uri is None or req_uri is None:
            return True
        if child_uri == req_uri:
            return True
        visited, stack = set(), [child_uri]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for parent in g.objects(cur, RDFS.subClassOf):
                if not isinstance(parent, URIRef):
                    continue
                if parent == req_uri:
                    return True
                stack.append(parent)
        return True

    if child_uri is None or req_uri is None:
        return False
    if child_uri == req_uri:
        return True

    visited, stack = set(), [child_uri]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for parent in g.objects(cur, RDFS.subClassOf):
            if not isinstance(parent, URIRef):
                continue
            if parent == req_uri:
                return True
            stack.append(parent)
    return False


def candidate_names(label: str, synonym_map: dict, type_map: dict) -> List[str]:
    names = [str(label)] if label is not None else []
    if isinstance(synonym_map, dict):
        for key in [label, str(label).lower() if label is not None else None,
                    norm_label(label) if label is not None else None]:
            if key is not None and key in synonym_map and synonym_map[key] is not None:
                names.append(local_name(synonym_map[key]))
    if isinstance(type_map, dict):
        for key in [label, str(label).lower() if label is not None else None,
                    norm_label(label) if label is not None else None]:
            if key is not None and key in type_map and type_map[key] is not None:
                names.append(str(type_map[key]))
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def make_label_compatible(g, class_index, ou, synonym_map, type_map):
    def compatible(label: str, required_type: str) -> bool:
        if required_type in (None, "DetectedObject"):
            return True
        for candidate in candidate_names(label, synonym_map, type_map):
            if ontology_is_a(g, class_index, candidate, required_type):
                return True
        try:
            legacy = ou.label_to_type(label, synonym_map, type_map)
        except Exception:
            legacy = None
        if legacy is None:
            return False
        if legacy == required_type:
            return True
        return ontology_is_a(g, class_index, legacy, required_type)
    return compatible


def solve_ilp_hierarchy(
    oc,
    objects,
    rel_cands,
    rel_type_constraints,
    label_compatible,
    alpha: float,
    beta: float,
    gamma: float,
    time_limit_s: int,
    rel_budget: int,
):
    import pulp

    prob = pulp.LpProblem("OCOSL_Labeling_HierarchyAware", pulp.LpMaximize)

    x = {}
    for obj in objects:
        for lbl, _ in obj.candidates:
            safe_lbl = re.sub(r"[^A-Za-z0-9_]+", "_", str(lbl))
            x[(obj.idx, lbl)] = pulp.LpVariable(
                f"x_{obj.idx}_{safe_lbl}", 0, 1, cat=pulp.LpBinary
            )

    for obj in objects:
        prob += (
            pulp.lpSum([x[(obj.idx, lbl)] for lbl, _ in obj.candidates]) == 1,
            f"one_label_{obj.idx}",
        )

    y = {}
    for rc in rel_cands:
        if not rc.feasible:
            continue
        key = (rc.i, rc.j, rc.rel)
        safe_rel = re.sub(r"[^A-Za-z0-9_]+", "_", str(rc.rel))
        y[key] = pulp.LpVariable(
            f"y_{rc.i}_{rc.j}_{safe_rel}", 0, 1, cat=pulp.LpBinary
        )

    if rel_budget and len(y) > 0:
        prob += pulp.lpSum(list(y.values())) <= int(rel_budget), "global_relation_budget"

    pairs: Dict[Tuple[int, int], List[tuple]] = {}
    for i, j, r in y.keys():
        pairs.setdefault((i, j), []).append((i, j, r))
    for (i, j), keys in pairs.items():
        prob += pulp.lpSum([y[k] for k in keys]) <= 1, f"one_rel_{i}_{j}"

    for (i, j, r), var in y.items():
        dom, ran = rel_type_constraints.get(r, (None, None))
        if dom in (None, "DetectedObject"):
            dom = None
        if ran in (None, "DetectedObject"):
            ran = None

        if dom:
            allowed = [
                x[(i, lbl)]
                for lbl, _ in objects[i].candidates
                if label_compatible(lbl, dom)
            ]
            prob += var <= pulp.lpSum(allowed), f"dom_{i}_{j}_{r}"

        if ran:
            allowed = [
                x[(j, lbl)]
                for lbl, _ in objects[j].candidates
                if label_compatible(lbl, ran)
            ]
            prob += var <= pulp.lpSum(allowed), f"ran_{i}_{j}_{r}"

    terms = []
    for obj in objects:
        for lbl, sdet in obj.candidates:
            sonto = obj.onto_scores.get(lbl, sdet)
            terms.append((alpha * sdet + beta * sonto) * x[(obj.idx, lbl)])

    rel_score_map = {
        (rc.i, rc.j, rc.rel): rc.score for rc in rel_cands if rc.feasible
    }
    for key, var in y.items():
        terms.append(gamma * rel_score_map.get(key, 0.0) * var)

    prob += pulp.lpSum(terms)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)

    status = pulp.LpStatus.get(prob.status, str(prob.status))
    x_star = {
        (i, lbl): int(pulp.value(var) is not None and pulp.value(var) > 0.5)
        for (i, lbl), var in x.items()
    }
    y_star = {
        (i, j, r): int(pulp.value(var) is not None and pulp.value(var) > 0.5)
        for (i, j, r), var in y.items()
    }
    return x_star, y_star, status, status != "Optimal"


def greedy_repair_hierarchy(
    objects,
    rel_cands,
    rel_type_constraints,
    label_compatible,
):
    x_star, chosen = {}, {}
    for obj in objects:
        best_lbl, best_val = None, -1e30
        for lbl, sdet in obj.candidates:
            value = sdet + obj.onto_scores.get(lbl, 0.0)
            if value > best_val:
                best_lbl, best_val = lbl, value
        chosen[obj.idx] = best_lbl
        for lbl, _ in obj.candidates:
            x_star[(obj.idx, lbl)] = int(lbl == best_lbl)

    y_star, seen_pairs = {}, set()
    for rc in sorted([r for r in rel_cands if r.feasible],
                     key=lambda z: z.score, reverse=True):
        pair = (rc.i, rc.j)
        if pair in seen_pairs:
            continue
        dom, ran = rel_type_constraints.get(rc.rel, (None, None))
        li, lj = chosen.get(rc.i), chosen.get(rc.j)
        if dom and dom != "DetectedObject" and not label_compatible(li, dom):
            continue
        if ran and ran != "DetectedObject" and not label_compatible(lj, ran):
            continue
        y_star[(rc.i, rc.j, rc.rel)] = 1
        seen_pairs.add(pair)
    return x_star, y_star


def solve_problem_v3(
    oc,
    objects,
    rel_cands,
    args,
    rel_constraints,
    label_compatible,
):
    budget = args.rel_budget if args.rel_budget > 0 else max(1, 2 * len(objects))
    t0 = time.time()
    try:
        x_star, y_star, status, fallback = solve_ilp_hierarchy(
            oc,
            objects,
            rel_cands,
            rel_constraints,
            label_compatible,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            time_limit_s=args.time_limit,
            rel_budget=budget,
        )
    except Exception:
        x_star, y_star = greedy_repair_hierarchy(
            objects, rel_cands, rel_constraints, label_compatible
        )
        status, fallback = "FallbackHierarchyGreedy", True
    dt = time.time() - t0
    selected_labels, enriched = oc.enrich_graph(objects, x_star, y_star)
    return x_star, y_star, selected_labels, enriched, status, fallback, dt


def load_gt_df(oi_root: Path) -> pd.DataFrame:
    path = oi_root / "openimages_relationships.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"ImageID": str})


def get_gt_relations(gt_df: pd.DataFrame, image_id: str, image_path: Path) -> List[dict]:
    with Image.open(image_path) as im:
        width, height = im.size
    rows = gt_df[gt_df["ImageID"].astype(str) == str(image_id)]
    gt = []
    for _, r in rows.iterrows():
        rel = norm_label(r["RelationLabel"])
        if rel == "is":
            continue
        sb = np.asarray(
            [
                float(r["XMin1"]) * width,
                float(r["YMin1"]) * height,
                float(r["XMax1"]) * width,
                float(r["YMax1"]) * height,
            ],
            dtype=float,
        )
        ob = np.asarray(
            [
                float(r["XMin2"]) * width,
                float(r["YMin2"]) * height,
                float(r["XMax2"]) * width,
                float(r["YMax2"]) * height,
            ],
            dtype=float,
        )
        gt.append({"rel": rel, "sub_box": sb, "obj_box": ob})
    return gt


def eval_records(records: List[dict], gt: List[dict], iou_thr: float) -> dict:
    records = sorted(records, key=lambda x: x["score"], reverse=True)
    matched = set()
    tp = 0
    for p in records:
        best_j, best_q = None, -1.0
        for j, target in enumerate(gt):
            if j in matched:
                continue
            if norm_label(p["eval_rel"]) != norm_label(target["rel"]):
                continue
            siou = iou_xyxy(p["eval_sub_box"], target["sub_box"])
            oiou = iou_xyxy(p["eval_obj_box"], target["obj_box"])
            if siou >= iou_thr and oiou >= iou_thr:
                q = min(siou, oiou)
                if q > best_q:
                    best_j, best_q = j, q
        if best_j is not None:
            matched.add(best_j)
            tp += 1

    fp = len(records) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt": len(gt), "pred": len(records), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def matched_candidate_keys(records: List[Tuple[tuple, dict]], gt: List[dict], iou_thr: float):
    ordered = sorted(records, key=lambda z: float(z[1]["score"]), reverse=True)
    used_gt, matched_keys = set(), set()
    for key, rec in ordered:
        best_j, best_quality = None, -1.0
        for j, target in enumerate(gt):
            if j in used_gt:
                continue
            if norm_label(rec["eval_rel"]) != norm_label(target["rel"]):
                continue
            siou = iou_xyxy(rec["eval_sub_box"], target["sub_box"])
            oiou = iou_xyxy(rec["eval_obj_box"], target["obj_box"])
            if siou >= iou_thr and oiou >= iou_thr:
                q = min(siou, oiou)
                if q > best_quality:
                    best_quality, best_j = q, j
        if best_j is not None:
            used_gt.add(best_j)
            matched_keys.add(key)
    return matched_keys


def micro_metrics(df: pd.DataFrame, prefix: str) -> dict:
    tp = int(df[f"{prefix}_tp"].sum())
    fp = int(df[f"{prefix}_fp"].sum())
    fn = int(df[f"{prefix}_fn"].sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f1}


def micro_ocosl_v3(df: pd.DataFrame) -> dict:
    tp = int(df["ocosl_v3_tp"].sum())
    fp = int(df["ocosl_v3_fp"].sum())
    fn = int(df["ocosl_v3_fn"].sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f1}


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = rng.choice(x, size=len(x), replace=True).mean()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def paired_permutation(a: Sequence[float], b: Sequence[float], n_perm: int, seed: int):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(a) != len(b):
        return float("nan")
    d = b - a
    observed = abs(float(d.mean()))
    if np.allclose(d, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        if abs(float((d * signs).mean())) >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def verify_reference(summary: dict, tolerance: float = 1e-9):
    checks = {
        "n_completed": summary["n_completed"] == REFERENCE["n_completed"],
        "reltr_f1": abs(summary["reltr_only_shared_pool"]["F1"] - REFERENCE["reltr_f1"]) <= tolerance,
        "ocosl_f1": abs(summary["reltr_plus_ocosl_v3"]["F1"] - REFERENCE["ocosl_f1"]) <= tolerance,
        "micro_f1_gain": abs(summary["micro_f1_gain"] - REFERENCE["micro_f1_gain"]) <= tolerance,
        "improved": summary["per_image_improved"] == REFERENCE["improved"],
        "equal": summary["per_image_equal"] == REFERENCE["equal"],
        "worse": summary["per_image_worse"] == REFERENCE["worse"],
    }
    return checks


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--oi-root", default="", help="Directory containing oi300_ids.txt and images/")
    p.add_argument("--code-dir", default="", help="Directory containing ocosl_run_v4.py and ocosl_utils_v3.py")
    p.add_argument("--ontology", default="", help="Path to JOCO ontology OWL file")
    p.add_argument("--reltr-repo", default="/kaggle/working/RelTR")
    p.add_argument("--reltr-ckpt", default="/kaggle/working/RelTR/ckpt/checkpoint0149_oi.pth")
    p.add_argument("--reltr-metadata-dir", default="/kaggle/working/reltr_oi_metadata")
    p.add_argument("--prepare-reltr", action="store_true", help="Clone RelTR and download official OI assets if missing")
    p.add_argument("--output-dir", default="/kaggle/working/JOCO/results_v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-images", type=int, default=300)
    p.add_argument("--query-thr", type=float, default=0.15)
    p.add_argument("--joint-score-thr", type=float, default=0.05)
    p.add_argument("--reltr-topk", type=int, default=50)
    p.add_argument("--entity-merge-iou", type=float, default=0.70)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--rel-budget", type=int, default=0, help="0 = C=2n")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.20)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--verify-reference", action="store_true")
    return p


def main():
    cli = build_parser().parse_args()

    random.seed(cli.seed)
    np.random.seed(cli.seed)

    ensure_package("pulp")
    import torch
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    oi_root = locate_oi_root(cli.oi_root)
    code_dir = locate_code_dir(cli.code_dir)
    ontology_path = locate_ontology(code_dir, cli.ontology)

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import ocosl_run_v4 as oc
    import ocosl_utils_v3 as ou

    repo = Path(cli.reltr_repo)
    ckpt = Path(cli.reltr_ckpt)
    metadata_dir = Path(cli.reltr_metadata_dir)
    prepare_reltr_assets(repo, ckpt, metadata_dir, cli.prepare_reltr)

    obj_labels, rel_labels = load_oi_vocab(metadata_dir)
    rel_label_file = Path(cli.output_dir) / "reltr_oi_relation_labels.txt"
    rel_label_file.parent.mkdir(parents=True, exist_ok=True)
    rel_label_file.write_text("\n".join(rel_labels), encoding="utf-8")

    device = cli.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    adapter = oc.RelTRAdapter(
        repo=str(repo),
        checkpoint=str(ckpt),
        dataset="oi",
        device=device,
        relation_labels_path=str(rel_label_file),
        query_threshold=cli.query_thr,
        max_triplets=cli.reltr_topk,
    )

    ids = [
        x.strip()
        for x in (oi_root / "oi300_ids.txt").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ][:cli.n_images]

    if not ids:
        raise RuntimeError("No image IDs found.")
    first_img = oi_root / "images" / f"{ids[0]}.jpg"
    with Image.open(first_img).convert("RGB") as im:
        tensor = adapter.transform(im).unsqueeze(0).to(adapter.device)
        with torch.no_grad():
            out0 = adapter.model(tensor)

    obj_labels = fit_vocab(obj_labels, int(out0["sub_logits"].shape[-1] - 1), "Object")
    rel_labels = fit_vocab(rel_labels, int(out0["rel_logits"].shape[-1] - 1), "Relation")
    adapter.rel_labels = rel_labels

    g, base = oc.load_ontology_schema(str(ontology_path))
    rel_constraints = oc.build_relation_type_constraints(g, base)
    synonym_map = oc.build_synonym_table(g)
    type_map = dict(oc.DEFAULT_TYPE_MAP)
    class_index = build_class_index(g)
    label_compatible = make_label_compatible(g, class_index, ou, synonym_map, type_map)

    # Sanity checks that motivated V3
    for lbl, typ in [("Person", "Person"), ("Boat", "Vehicle"), ("Glasses", "Clothing")]:
        if not label_compatible(lbl, typ):
            raise RuntimeError(f"Hierarchy compatibility failed: {lbl} is-a {typ}")

    args = oc.build_arg_parser().parse_args([])
    args.topk = 3
    args.alpha = cli.alpha
    args.beta = cli.beta
    args.gamma = cli.gamma
    args.time_limit = cli.time_limit
    args.rel_budget = cli.rel_budget
    args.iou_dup = 0.60

    gt_df = load_gt_df(oi_root)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    progress_csv = output / "oi_reltr_ocosl_300_v3_progress.csv"
    impact_progress_csv = output / "oi_reltr_ocosl_300_v3_impact_progress.csv"
    error_csv = output / "oi_reltr_ocosl_300_v3_errors.csv"
    heartbeat = output / "oi_reltr_ocosl_300_v3_running.txt"
    final_csv = output / "oi_reltr_ocosl_controlled_300_v3_hierarchy.csv"
    final_json = output / "oi_reltr_ocosl_controlled_300_v3_hierarchy.json"
    rel_impact_csv = output / "reltr_ocosl_relation_impact_300_v3.csv"
    image_impact_csv = output / "reltr_ocosl_image_impact_300_v3.csv"
    mapping_csv = output / "openimages_ocosl_canonical_mapping_v2.csv"
    bundle_zip = output / "JOCO_RelTR_OCOSL_V3_300_bundle.zip"

    pd.DataFrame(
        [{"openimages_label": k, "ocosl_canonical_label": v}
         for k, v in sorted(OI_OBJECT_CANONICAL_MAP.items())]
    ).to_csv(mapping_csv, index=False)

    heartbeat.write_text(
        f"started={time.ctime()}\nrequested={len(ids)}\n", encoding="utf-8"
    )

    if progress_csv.exists():
        prior = pd.read_csv(progress_csv, dtype={"image_id": str})
        rows = prior.to_dict("records")
        completed = set(prior["image_id"].astype(str))
    else:
        rows, completed = [], set()

    if impact_progress_csv.exists():
        impact_rows = pd.read_csv(impact_progress_csv, dtype={"image_id": str}).to_dict("records")
    else:
        impact_rows = []

    if error_csv.exists():
        error_rows = pd.read_csv(error_csv, dtype={"image_id": str}).to_dict("records")
    else:
        error_rows = []

    pending = [iid for iid in ids if iid not in completed]

    try:
        from tqdm.auto import tqdm
    except ImportError:
        ensure_package("tqdm")
        from tqdm.auto import tqdm

    for image_id in tqdm(pending, desc="OI-300 V3", unit="img"):
        image_path = oi_root / "images" / f"{image_id}.jpg"
        try:
            gt = get_gt_relations(gt_df, image_id, image_path)

            t0 = time.time()
            raw, objects, rel_cands, meta, mapping_usage = build_scene_graph(
                oc, adapter, image_path, obj_labels, rel_labels,
                query_thr=cli.query_thr,
                joint_thr=cli.joint_score_thr,
                reltr_topk=cli.reltr_topk,
                entity_merge_iou=cli.entity_merge_iou,
                args=args,
                synonym_map=synonym_map,
                type_map=type_map,
            )
            generator_time = time.time() - t0

            raw_eval = [
                {
                    "eval_sub_box": p["sub_box_xyxy"],
                    "eval_obj_box": p["obj_box_xyxy"],
                    "eval_rel": norm_label(p["relation"]),
                    "score": float(p["score"]),
                }
                for p in raw
            ]
            raw_m = eval_records(raw_eval, gt, cli.eval_iou)

            baseline_keys = oc.select_learned_relations_without_ocosl(
                rel_cands, len(objects), cli.rel_budget
            )
            baseline_keys = [k for k in baseline_keys if k in meta]
            base_m = eval_records([meta[k] for k in baseline_keys], gt, cli.eval_iou)
            baseline_tp = matched_candidate_keys(
                [(k, meta[k]) for k in baseline_keys], gt, cli.eval_iou
            )

            x_star, y_star, selected_labels, enriched, status, fallback, solve_time = solve_problem_v3(
                oc, objects, rel_cands, args, rel_constraints, label_compatible
            )
            ocosl_keys = [
                k for k, value in y_star.items()
                if float(value) >= 0.5 and k in meta
            ]
            opt_m = eval_records([meta[k] for k in ocosl_keys], gt, cli.eval_iou)
            ocosl_tp = matched_candidate_keys(
                [(k, meta[k]) for k in ocosl_keys], gt, cli.eval_iou
            )

            base_set, opt_set = set(baseline_keys), set(ocosl_keys)
            lost = base_set - opt_set
            added = opt_set - base_set

            for key in lost:
                rec = meta[key]
                impact_rows.append(
                    {
                        "image_id": image_id,
                        "event": "removed_by_ocosl",
                        "source_relation": norm_label(rec["eval_rel"]),
                        "canonical_relation": rec.get("canonical_rel", key[2]),
                        "is_true_positive": int(key in baseline_tp),
                        "score": float(rec["score"]),
                    }
                )
            for key in added:
                rec = meta[key]
                impact_rows.append(
                    {
                        "image_id": image_id,
                        "event": "added_by_ocosl",
                        "source_relation": norm_label(rec["eval_rel"]),
                        "canonical_relation": rec.get("canonical_rel", key[2]),
                        "is_true_positive": int(key in ocosl_tp),
                        "score": float(rec["score"]),
                    }
                )

            row = {
                "image_id": str(image_id),
                "n_gt_rel": len(gt),
                "n_raw_reltr": len(raw),
                "n_entities": len(objects),
                "n_candidate_pool": len(rel_cands),
                "canonicalized_endpoints": int(sum(mapping_usage.values())),
                "raw_tp": raw_m["tp"], "raw_fp": raw_m["fp"], "raw_fn": raw_m["fn"],
                "raw_precision": raw_m["precision"], "raw_recall": raw_m["recall"], "raw_f1": raw_m["f1"],
                "reltr_tp": base_m["tp"], "reltr_fp": base_m["fp"], "reltr_fn": base_m["fn"],
                "reltr_precision": base_m["precision"], "reltr_recall": base_m["recall"], "reltr_f1": base_m["f1"],
                "ocosl_v3_tp": opt_m["tp"], "ocosl_v3_fp": opt_m["fp"], "ocosl_v3_fn": opt_m["fn"],
                "ocosl_v3_precision": opt_m["precision"], "ocosl_v3_recall": opt_m["recall"], "ocosl_v3_f1": opt_m["f1"],
                "delta_f1": opt_m["f1"] - base_m["f1"],
                "reltr_selected": len(base_set),
                "ocosl_selected": len(opt_set),
                "removed_by_ocosl": len(lost),
                "removed_true_positive": len(lost & baseline_tp),
                "added_by_ocosl": len(added),
                "added_true_positive": len(added & ocosl_tp),
                "generator_time_s": generator_time,
                "solver_time_s": solve_time,
                "solver_status": str(status),
                "fallback": int(bool(fallback)),
            }
            rows.append(row)

            progress = (
                pd.DataFrame(rows)
                .drop_duplicates(subset=["image_id"], keep="last")
                .reset_index(drop=True)
            )
            progress.to_csv(progress_csv, index=False)
            pd.DataFrame(impact_rows).to_csv(impact_progress_csv, index=False)
            heartbeat.write_text(
                f"last_completed={image_id}\ncompleted={len(progress)}\n"
                f"remaining={len(ids)-len(progress)}\ntime={time.ctime()}\n",
                encoding="utf-8",
            )

        except Exception as exc:
            error_rows.append(
                {
                    "image_id": str(image_id),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "time": time.ctime(),
                }
            )
            pd.DataFrame(error_rows).to_csv(error_csv, index=False)
            print(f"[ERROR] {image_id}: {type(exc).__name__}: {exc}", flush=True)

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["image_id"], keep="last")
        .reset_index(drop=True)
    )
    df.to_csv(final_csv, index=False)

    raw_summary = micro_metrics(df, "raw")
    reltr_summary = micro_metrics(df, "reltr")
    ocosl_summary = micro_ocosl_v3(df)
    df["delta_f1"] = df["ocosl_v3_f1"] - df["reltr_f1"]

    reltr_ci = bootstrap_mean_ci(df["reltr_f1"], cli.bootstrap, cli.seed)
    ocosl_ci = bootstrap_mean_ci(df["ocosl_v3_f1"], cli.bootstrap, cli.seed)
    p_value = paired_permutation(
        df["reltr_f1"], df["ocosl_v3_f1"], cli.permutations, cli.seed
    )

    n_improved = int((df["delta_f1"] > 1e-12).sum())
    n_equal = int((df["delta_f1"].abs() <= 1e-12).sum())
    n_worse = int((df["delta_f1"] < -1e-12).sum())
    micro_gain = ocosl_summary["F1"] - reltr_summary["F1"]

    image_cols = [
        "image_id", "n_gt_rel", "n_candidate_pool", "reltr_selected", "ocosl_selected",
        "reltr_tp", "ocosl_v3_tp", "removed_by_ocosl", "removed_true_positive",
        "added_by_ocosl", "added_true_positive", "reltr_f1", "ocosl_v3_f1", "delta_f1",
    ]
    df[image_cols].to_csv(image_impact_csv, index=False)

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows).drop_duplicates().reset_index(drop=True)
        grouped = []
        for relation in sorted(impact_df["source_relation"].dropna().unique()):
            sub = impact_df[impact_df["source_relation"] == relation]
            removed = sub[sub["event"] == "removed_by_ocosl"]
            added = sub[sub["event"] == "added_by_ocosl"]
            grouped.append(
                {
                    "relation": relation,
                    "removed_total": int(len(removed)),
                    "removed_true_positive": int(removed["is_true_positive"].sum()),
                    "removed_false_positive": int(len(removed) - removed["is_true_positive"].sum()),
                    "added_total": int(len(added)),
                    "added_true_positive": int(added["is_true_positive"].sum()),
                    "added_false_positive": int(len(added) - added["is_true_positive"].sum()),
                }
            )
        rel_impact = pd.DataFrame(grouped)
    else:
        rel_impact = pd.DataFrame(
            columns=[
                "relation", "removed_total", "removed_true_positive",
                "removed_false_positive", "added_total",
                "added_true_positive", "added_false_positive",
            ]
        )
    rel_impact.to_csv(rel_impact_csv, index=False)

    summary = {
        "experiment": "OI-trained RelTR vs hierarchy-aware OCOSL V3",
        "n_requested": cli.n_images,
        "n_completed": int(len(df)),
        "implementation_change": (
            "Ontology domain/range compatibility uses rdfs:subClassOf transitive "
            "closure rather than flat label_to_type equality."
        ),
        "ontology_changed": False,
        "query_threshold": cli.query_thr,
        "joint_score_threshold": cli.joint_score_thr,
        "reltr_topk": cli.reltr_topk,
        "entity_merge_iou": cli.entity_merge_iou,
        "evaluation_iou": cli.eval_iou,
        "relation_budget": "2n" if cli.rel_budget == 0 else cli.rel_budget,
        "seed": cli.seed,
        "time_limit_s": cli.time_limit,
        "alpha": cli.alpha, "beta": cli.beta, "gamma": cli.gamma,
        "raw_oi_reltr": raw_summary,
        "reltr_only_shared_pool": reltr_summary,
        "reltr_plus_ocosl_v3": ocosl_summary,
        "micro_f1_gain": float(micro_gain),
        "reltr_mean_image_f1_ci95": reltr_ci,
        "ocosl_mean_image_f1_ci95": ocosl_ci,
        "paired_permutation_pvalue": float(p_value),
        "per_image_improved": n_improved,
        "per_image_equal": n_equal,
        "per_image_worse": n_worse,
        "fallback_count": int(df["fallback"].sum()),
        "avg_candidate_pool": float(df["n_candidate_pool"].mean()),
        "avg_entities": float(df["n_entities"].mean()),
        "avg_generator_time_s": float(df["generator_time_s"].mean()),
        "avg_solver_time_s": float(df["solver_time_s"].mean()),
        "canonicalized_endpoint_count": int(df["canonicalized_endpoints"].sum()),
        "relations_removed_total": int(df["removed_by_ocosl"].sum()),
        "true_positives_removed_total": int(df["removed_true_positive"].sum()),
        "relations_added_total": int(df["added_by_ocosl"].sum()),
        "true_positives_added_total": int(df["added_true_positive"].sum()),
    }
    final_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [
            final_csv, final_json, rel_impact_csv, image_impact_csv,
            progress_csv, error_csv, mapping_csv, rel_label_file,
        ]:
            if p.exists():
                z.write(p, arcname=p.name)

    print("\n=== FINAL 300-IMAGE V3 RESULT ===")
    print(json.dumps(summary, indent=2))
    print("CSV:", final_csv)
    print("JSON:", final_json)
    print("Bundle:", bundle_zip)

    if cli.verify_reference and cli.n_images == 300:
        checks = verify_reference(summary)
        print("\nReference verification:")
        for key, ok in checks.items():
            print(f"  {key}: {'PASS' if ok else 'FAIL'}")
        if not all(checks.values()):
            raise SystemExit("Reference verification failed.")


if __name__ == "__main__":
    main()
