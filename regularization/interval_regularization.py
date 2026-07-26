import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation

from copy import deepcopy


class IntervalPenalization(nn.Module):
    """
    Loss module for InTAct, adapted for KA-Prompt.

    Penalizes:
        - Variance within interval activations of the current task.
        - Drift in internal representations between tasks.
        - Feature-level drift (using externally-provided old features).
        - Misalignment of new representations relative to previous interval bounds.
    """

    def __init__(self,
            var_loss_scale: float = 0.01,
            internal_repr_drift_loss_scale: float = 1.0,
            feature_loss_scale: float = 1.0,
            use_align_loss: bool = True
        ) -> None:
        super().__init__()
        self.task_id = None

        self.var_loss_scale = var_loss_scale
        self.internal_repr_drift_loss_scale = internal_repr_drift_loss_scale
        self.feature_loss_scale = feature_loss_scale
        self.use_align_loss = use_align_loss

        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None

    def detach_interval_last_batches(self, curr_classifier_head: nn.Sequential) -> None:
        layers = list(curr_classifier_head.children())
        for layer in layers:
            if isinstance(layer, IntervalActivation):
                layer.curr_task_last_batch = None

    def setup_task(
        self,
        task_id: int,
        curr_classifier_head: nn.Sequential,
    ) -> None:
        """
        Sets up the penalization module for the current task.

        Clones previous task parameters, freezes them, and resets
        interval activation bounds for the current classifier head.
        """
        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head

        if task_id > 0:
            self.params_buffer = {
                name: p.detach().clone()
                for name, p in self.curr_classifier_head.named_parameters()
            }

            self.detach_interval_last_batches(curr_classifier_head)
            self.old_classifier_head = deepcopy(curr_classifier_head)
            for p in self.old_classifier_head.parameters():
                p.requires_grad = False

            for idx, layer in enumerate(self.curr_classifier_head):
                if isinstance(layer, IntervalActivation):
                    layer.reset_range()
                    if layer.min is not None and layer.max is not None:
                        print(
                            f"Volume of the cumulative hypercube for {idx+1}-th layer "
                            f"in classification head: {torch.mean(layer.max - layer.min).item()}"
                        )

    def forward(self, x: torch.Tensor, loss: torch.Tensor, old_features: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the interval-based penalization loss and adds it to the task loss.

        Args:
            x: Input batch tensor.
            loss: Original task-specific loss to augment.
            old_features: Pre-computed old model CLS features (detached), used for
                          interval drift loss. Should be in the same representation
                          space as IntervalActivation inputs (post-fc_norm).

        Returns:
            Loss augmented with interval penalization terms.
        """
        layers = list(self.curr_classifier_head.children())
        interval_act_layers = [i for i, layer in enumerate(layers) if isinstance(layer, IntervalActivation)]

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        var_loss = zero.clone()
        output_reg_loss = zero.clone()
        interval_drift_loss = zero.clone()
        align_repr_loss = zero.clone()

        for idx in interval_act_layers:
            acts = layers[idx].curr_task_last_batch
            if not isinstance(acts, torch.Tensor):
                continue

            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            layer_min = layers[idx].min
            layer_max = layers[idx].max
            if self.task_id > 0 and layer_min is not None and layer_max is not None:
                lb = layer_min.to(x.device)
                ub = layer_max.to(x.device)

                # Interval drift loss at the FIRST IntervalActivation
                if idx == interval_act_layers[0] and old_features is not None:
                    y_old = old_features.to(x.device)
                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )

                if idx + 1 < len(layers) and isinstance(layers[idx + 1], torch.nn.Linear):
                    next_layer = layers[idx + 1]
                    layer_prefix = f"{idx + 1}"

                    out_dim = next(next_layer.parameters()).shape[0]
                    total_lower = torch.zeros(out_dim, device=x.device)
                    total_upper = torch.zeros(out_dim, device=x.device)

                    for name, p in next_layer.named_parameters():
                        full_name = f"{layer_prefix}.{name}"
                        if full_name in self.params_buffer:
                            prev_param = self.params_buffer[full_name]
                            diff = p - prev_param

                            if "weight" in name:
                                diff_pos = torch.relu(diff)
                                diff_neg = torch.relu(-diff)

                                total_lower += (diff_pos @ lb - diff_neg @ ub)
                                total_upper += (diff_pos @ ub - diff_neg @ lb)

                            elif "bias" in name:
                                total_lower += diff
                                total_upper += diff

                    output_reg_loss += (total_lower.pow(2).mean() + total_upper.pow(2).mean())

                if self.use_align_loss:
                    prev_center = (ub + lb) / 2.0
                    prev_radii = (ub - lb) / 2.0

                    lb_prev_hypercube = prev_center - prev_radii
                    ub_prev_hypercube = prev_center + prev_radii

                    new_lb, _ = acts_flat.min(dim=0)
                    new_ub, _ = acts_flat.max(dim=0)

                    non_overlap_mask = (new_lb > ub_prev_hypercube) | (new_ub < lb_prev_hypercube)
                    new_center = (new_ub + new_lb) / 2.0

                    center_loss = torch.norm(new_center[non_overlap_mask] - prev_center[non_overlap_mask], p=2)

                    align_repr_loss += center_loss / (prev_radii.mean() + 1e-8)

        loss = (
            loss
            + self.var_loss_scale * var_loss
            + self.internal_repr_drift_loss_scale * output_reg_loss
            + self.feature_loss_scale * interval_drift_loss
            + align_repr_loss
        )
        return loss
