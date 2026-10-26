"""Dist-Tracker's box loss (Shape-NWD + dynamic WIoU, "SSQ") for the common detector.

Runs inside the Strong Baseline environment, imported by ``_ultralytics_runner.py``
after the YOLOv12 fork (the SOT weights only load in that code). The fork's files are
not changed: ``patch()`` swaps ``ultralytics.utils.loss.BboxLoss`` at run time.

Adapted from Dist-Tracker ``ultralytics/utils/loss.py`` (``BboxLoss``, used by
``v8DetectionLoss`` with ``ratio_power=0.6``), with three fixes so it behaves as the
paper describes:

1. the Wasserstein term is normalised, ``1 - exp(-sqrt(W2) / C)`` with
   ``C = 5 * mean target size`` (the normalisation of the repository's own, unused
   ``NWDLoss``); the published code adds the raw squared distance, which overflows
   under mixed precision (the NaN box loss seen in the smoke test);
2. the training progress reaches the loss: the published code reads
   ``batch['epoch']``, which Ultralytics never sets, so the NWD -> WIoU schedule
   stayed at its first-epoch weights. Here ``on_train_epoch_start`` sets it;
3. the WIoU term gets the boxes in their real format: the published code calls
   ``bbox_iou(...)`` with the default ``xywh=True`` on ``xyxy`` boxes.
"""

from __future__ import annotations

STATE = {"epoch_ratio": 0.0}


def _make_loss_class(base_module):
    import torch
    from torch import nn

    bbox_iou = base_module.bbox_iou
    bbox2dist = base_module.bbox2dist
    xyxy2xywh = base_module.xyxy2xywh
    DFLoss = base_module.DFLoss

    class SSQBboxLoss(nn.Module):
        """Shape-NWD + dynamic WIoU box loss; DFL as in Ultralytics."""

        def __init__(self, reg_max=16, use_dfl=True, alpha=2.5, ratio_power=0.6, gamma_scale=0.3):
            super().__init__()
            self.reg_max = reg_max
            self.use_dfl = use_dfl
            self.alpha = alpha
            self.ratio_power = ratio_power
            self.gamma_scale = gamma_scale
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        @staticmethod
        def wasserstein_distance(mu1, mu2, sigma1, sigma2):
            center_distance = torch.sum(torch.pow(mu1 - mu2, 2), dim=1)
            wh_distance = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=1)
            return center_distance + 0.8 * wh_distance

        def shape_aware_penalty(self, pred_wh, target_wh, eps=1e-7):
            w_pow = torch.pow(target_wh[..., 0], self.ratio_power)
            h_pow = torch.pow(target_wh[..., 1], self.ratio_power)
            ww = 2 * w_pow / (w_pow + h_pow + eps)
            hh = 2 * h_pow / (w_pow + h_pow + eps)
            omiga_w = torch.abs(pred_wh[..., 0] - target_wh[..., 0]) / (
                torch.max(pred_wh[..., 0], target_wh[..., 0]) + eps)
            omiga_h = torch.abs(pred_wh[..., 1] - target_wh[..., 1]) / (
                torch.max(pred_wh[..., 1], target_wh[..., 1]) + eps)
            omiga_w = torch.clamp(omiga_w, max=5.0)
            omiga_h = torch.clamp(omiga_h, max=5.0)
            return (ww * omiga_w.pow(2) + hh * omiga_h.pow(2)) * self.gamma_scale

        def dynamic_wiou(self, iou):
            beta = (1 - iou).detach() / (1 - iou.mean().detach() + 1e-7)
            r = beta / (self.alpha ** (beta - 1 + 1e-3))
            return r * (1 - iou) ** self.alpha

        def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                    target_scores, target_scores_sum, fg_mask):
            loss_iou = torch.zeros(1, device=pred_dist.device)
            loss_dfl = torch.zeros(1, device=pred_dist.device)
            if fg_mask.sum():
                pred_xywh = xyxy2xywh(pred_bboxes[fg_mask]).float()
                target_xywh = xyxy2xywh(target_bboxes[fg_mask]).float()
                mu_p, mu_t = pred_xywh[..., :2], target_xywh[..., :2]
                sigma_p, sigma_t = pred_xywh[..., 2:], target_xywh[..., 2:]

                w2 = self.wasserstein_distance(mu_p, mu_t, sigma_p, sigma_t)
                norm = 5.0 * (sigma_t.mean(dim=-1) + 1e-7)                      # fix 1
                nwd = 1.0 - torch.exp(-torch.sqrt(w2 + 1e-12) / norm)
                loss_nwd = nwd + self.shape_aware_penalty(sigma_p, sigma_t)

                iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)  # fix 3
                loss_wiou = self.dynamic_wiou(iou)

                ratio = STATE["epoch_ratio"]                                    # fix 2
                loss_iou = 0.7 * (1 - ratio) * loss_nwd.mean() + 0.3 * (1 + ratio) * loss_wiou.mean()

                if self.use_dfl and self.dfl_loss:
                    target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
                    loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max),
                                             target_ltrb[fg_mask]).mean()
            return loss_iou, loss_dfl

    return SSQBboxLoss


def patch() -> None:
    """Replaces the fork's ``BboxLoss`` (looked up by ``v8DetectionLoss`` when the criterion is built)."""
    import ultralytics.utils.loss as loss_module

    if getattr(loss_module.BboxLoss, "_tir_jde_ssq", False):
        return
    cls = _make_loss_class(loss_module)
    cls._tir_jde_ssq = True
    loss_module.BboxLoss = cls
    print("[runner] kutu kaybı: Dist-Tracker SSQ (Shape-NWD + dinamik WIoU), normalize NWD + epoch ilerlemesi",
          flush=True)


def epoch_callback(trainer) -> None:
    """``on_train_epoch_start``: training progress for the NWD -> WIoU schedule."""
    STATE["epoch_ratio"] = trainer.epoch / max(trainer.epochs, 1)
