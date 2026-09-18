from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom


@dataclass(frozen=True)
class TableSupportReport:
    valid: bool
    reason: str
    object_name: str
    support_geom_id: int | None
    support_body_id: int | None
    object_aabb_min: tuple[float, float, float] | None
    object_aabb_max: tuple[float, float, float] | None
    support_aabb_min: tuple[float, float, float] | None
    support_aabb_max: tuple[float, float, float] | None


def check_table_support(data, object_name: str, margin: float = 0.01) -> TableSupportReport:
    """Hard reset-time support gate: object footprint must be inside its support geom."""
    obj = create_mlspaces_body(data, object_name)
    object_id = int(obj.body_id)
    support_geom_id = get_supporting_geom(data, object_id)
    # Lightweight objects may have no contact record immediately after reset.
    # Use a conservative geometry-only fallback: a collidable non-object geom
    # whose top is close to the object's bottom and whose XY footprint contains it.
    if support_geom_id is None:
        obj_center0, obj_size0 = body_aabb(data.model, data, object_id, visible_only=False)
        obj_min0 = np.asarray(obj_center0) - np.asarray(obj_size0) / 2
        obj_max0 = np.asarray(obj_center0) + np.asarray(obj_size0) / 2
        candidates = []
        for gid in range(data.model.ngeom):
            if int(data.model.geom_bodyid[gid]) == object_id:
                continue
            if data.model.geom_contype[gid] == 0 and data.model.geom_conaffinity[gid] == 0:
                continue
            gc, gs = geom_aabb(data.model, data, [gid])
            gmin = np.asarray(gc) - np.asarray(gs) / 2
            gmax = np.asarray(gc) + np.asarray(gs) / 2
            contains_xy = bool(
                np.all(obj_min0[:2] >= gmin[:2] - 1e-4) and np.all(obj_max0[:2] <= gmax[:2] + 1e-4)
            )
            top_gap = float(obj_min0[2] - gmax[2])
            if contains_xy and -0.03 <= top_gap <= 0.03:
                candidates.append((abs(top_gap), gid))
        if candidates:
            support_geom_id = min(candidates)[1]
        else:
            return TableSupportReport(
                False,
                "no_upward_support_contact_or_geometry",
                object_name,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    support_body_id = int(data.model.body_rootid[data.model.geom_bodyid[support_geom_id]])
    if support_body_id == object_id:
        return TableSupportReport(
            False,
            "self_support",
            object_name,
            support_geom_id,
            support_body_id,
            None,
            None,
            None,
            None,
        )
    obj_center, obj_size = body_aabb(data.model, data, object_id, visible_only=False)
    sup_center, sup_size = geom_aabb(data.model, data, [support_geom_id])
    obj_min = np.asarray(obj_center) - np.asarray(obj_size) / 2
    obj_max = np.asarray(obj_center) + np.asarray(obj_size) / 2
    sup_min = np.asarray(sup_center) - np.asarray(sup_size) / 2
    sup_max = np.asarray(sup_center) + np.asarray(sup_size) / 2
    xy_inside = bool(
        np.all(obj_min[:2] >= sup_min[:2] + margin) and np.all(obj_max[:2] <= sup_max[:2] - margin)
    )
    z_supported = bool(obj_min[2] >= sup_max[2] - 0.03 and obj_min[2] <= sup_max[2] + 0.03)
    valid = xy_inside and z_supported
    reason = (
        "ok"
        if valid
        else (
            "footprint_outside_support_geom"
            if not xy_inside
            else "object_bottom_not_on_support_top"
        )
    )
    return TableSupportReport(
        valid,
        reason,
        object_name,
        support_geom_id,
        support_body_id,
        tuple(map(float, obj_min)),
        tuple(map(float, obj_max)),
        tuple(map(float, sup_min)),
        tuple(map(float, sup_max)),
    )
