import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits).float()
    target = target.float()
    inter = (pred * target).sum(dim=(-2, -1))
    den = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def visible_mask_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float()) + dice_loss(logits, target)


def port_heatmap_loss(pred, target):
    return F.mse_loss(torch.sigmoid(pred), target.float())


def mdl_complexity_loss(num_nodes, num_rules, num_relations, num_queries, node_cost=0.02, rule_cost=0.03, relation_cost=0.01, query_cost=0.005, device=None):
    return torch.tensor(float(node_cost * num_nodes + rule_cost * num_rules + relation_cost * num_relations + query_cost * num_queries), device=device)
