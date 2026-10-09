"""Frozen metric/input adapter; lineage recorded in provenance.json."""
def metric_control(ev):
    ev.install_tensorboard_import_stub()
    import torch
    import projects.mmdet3d_plugin
    from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
        IntersectionOverUnion,
        PanopticMetric,
    )
    from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import (
        PlanningMetric,
    )
    from pytorch_lightning.metrics.metric import Metric
    import inspect

    iou = IntersectionOverUnion(2).cpu()
    pred = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    target = torch.tensor([0, 1, 0, 0], dtype=torch.long)
    iou(pred, target)
    if not torch.allclose(iou.compute(), torch.tensor([2.0 / 3.0, 0.5])):
        raise ValueError("real occupancy metric control failed")
    plan = PlanningMetric().cpu()
    panoptic = PanopticMetric(n_classes=2, temporally_consistent=True).cpu()
    if (
        not isinstance(plan, torch.nn.Module)
        or not isinstance(panoptic, torch.nn.Module)
        or not hasattr(plan, "add_state")
    ):
        raise ValueError("inference stub replaced real task metric")
    files = [
        Path(inspect.getfile(c)).resolve()
        for c in (Metric, IntersectionOverUnion, PanopticMetric, PlanningMetric)
    ]
    return {
        "status": "pass",
        "real_lightning_metric": str(files[0]),
        "metric_files_sha256": {str(p): sha(p) for p in files},
        "iou_control": iou.compute().tolist(),
        "export_module_absent": True,
        "planning_and_panoptic_constructed": True,
    }
