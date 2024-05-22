import numpy as np
import torch
from torch import Tensor, nn

from typing import Optional
from functools import partial

from torch.nn import functional as F
from torchattacks import SparseFool

from utils.rs_attacks import RSAttack
from utils.brendel_bethge import L0BrendelBethgeAttack

import foolbox as fb
from foolbox.attacks.ead import EADAttack
from foolbox.attacks.dataset_attack import DatasetAttack


class SingleChannelModel():
    # code has been adapted from https://github.com/fra31/sparse-rs/blob/master/utils.py
    """ reshapes images to rgb before classification
        i.e. [N, 1, H, W x 3] -> [N, 3, H, W]
    """

    def __init__(self, model, desired_shape):
        if isinstance(model, nn.Module):
            assert not model.training
        self.model = model
        self.c, self.h, self.w = desired_shape

    def __call__(self, x):
        return self.model(x.view(x.shape[0], self.c, self.h, self.w))

    def to(self, device):
        # Call the to(device) method on the model
        return self.model.to(device)


def get_predictions_and_gradients(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    with torch.enable_grad():
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y.cpu()).numpy()

    return pred, grad


def get_predictions(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)

    model = model.to(device)
    xdev = x.to(device)

    with torch.no_grad():
        output = model(xdev)

    return (output.cpu().max(dim=-1)[1] == y).numpy()


# ---------- code taken from https://github.com/CityU-MLO/sPGD/blob/main/autoattack/spgd.py
def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    # return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
    #         x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)
    return -(x[u, y] - x[u, y_target])


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    # return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
    #         1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))


def margin_loss(logits, x, y, targeted=False):
    """
        :param y:        correct labels if untargeted else target labels
        """
    u = torch.arange(x.shape[0])
    y_corr = logits[u, y].clone()
    logits[u, y] = -float('inf')
    y_others = logits.max(dim=-1)[0]

    if not targeted:
        return y_corr - y_others
    else:
        return y_others - y_corr


class MaskingA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None


class MaskingB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        # mask_back = mask.clone().view(b, -1).scatter_(1, idx[:, k:], 0).view(b, 1, h, w)
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask_proj)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None


class SparsePGD(object):
    def __init__(self, model, epsilon=255 / 255, k=10, t=30, random_start=True, patience=3, classes=10, alpha=0.25,
                 beta=0.25, unprojected_gradient=True, verbose=False):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.t = t
        self.random_start = random_start
        self.alpha = epsilon * alpha
        self.beta = beta
        self.patience = patience
        self.classes = classes
        self.masking = MaskingA() if unprojected_gradient else MaskingB()
        self.weight_decay = 0.0
        self.p_init = 1.0
        self.verbose = verbose
        self.verbose_interval = 100

    def initial_perturb(self, x, seed=-1):
        if self.random_start:
            if seed != -1:
                torch.random.manual_seed(seed)
            perturb = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new(x.size()).zero_()
        perturb = torch.min(torch.max(perturb, -x), 1 - x)
        return perturb

    def update_perturbation(self, perturb, grad, x):
        b, c, h, w = perturb.size()
        step_size = self.alpha * torch.ones(b, device=perturb.device)
        perturb1 = perturb + step_size.view(b, 1, 1, 1) * grad.sign()
        perturb1 = perturb1.clamp_(-self.epsilon, self.epsilon)
        perturb1 = torch.min(torch.max(perturb1, -x), 1 - x)
        return perturb1

    def update_mask(self, mask, grad):
        # prev_mask = self.project_mask(mask.clone())
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        d = grad / (grad_norm + 1e-10)

        step_size = np.sqrt(h * w * c) * self.beta * torch.ones(b, device=mask.device)
        step_size = step_size.scatter_(0, (grad_norm.view(-1) < 2e-10).nonzero().squeeze(), 0)
        mask = mask + step_size.view(b, 1, 1, 1) * d

        return mask

    def initial_mask(self, x, it=0, prev_mask=None):

        if x.dim() == 3:
            x = x.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0) if prev_mask is not None else None
        b, c, h, w = x.size()

        mask = torch.randn(b, 1, h, w).to(x.device)

        if prev_mask is not None:
            prev_mask = prev_mask.view(b, -1)
            _, idx = torch.sort(prev_mask.view(b, -1), dim=1, descending=True)
            k_idx = idx[:, :self.k]
            for i in range(len(idx)):
                k_idx[i] = k_idx[i][torch.randperm(self.k)]

            # print(rand_idx.shape, idx.shape)
            p = self.p_selection(it)
            p = max(1, int(p * self.k))
            mask_mask = torch.ones_like(prev_mask).scatter_(1, k_idx[:, :p], 0)
            mask = mask_mask * prev_mask + (1 - mask_mask) * mask.view(b, -1)
            mask = mask.view(b, 1, h, w)

        return mask

    def project_mask(self, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :self.k], 1).view(b, c, h, w)
        return mask_proj

    def check_shape(self, x):
        return x if len(x.shape) == 4 else x.unsqueeze(0)

    def check_low_confidence(self, logits, y, threshold=0.5):
        logits_tmp = logits.clone()
        b, c = logits.size()
        u = torch.arange(b).to(logits.device)
        correct_logit = logits[u, y]
        logits_tmp[u, y] = -float('inf')
        wrong_logit = logits_tmp.max(dim=1)[0]
        return (correct_logit - wrong_logit) < threshold

    def __call__(self, x, y, seed=-1, targeted=False, target=None):
        b, c, h, w = x.size()
        it = torch.zeros(b, dtype=torch.long, device=x.device)

        # generate initial perturbation
        perturb = self.initial_perturb(x, seed)

        # generate initial mask
        mask = self.initial_mask(x)

        mask_best = mask.clone()
        perturb_best = perturb.clone()

        training = self.model.training
        if training:
            self.model.eval()

        ind_all = torch.arange(b).to(x.device)
        reinitial_count = torch.zeros(b, dtype=torch.long, device=x.device)
        x_adv_best = x.clone()

        # remove misclassified examples
        with torch.no_grad():
            logits = self.model(x)
            loss_best = F.cross_entropy(logits, y, reduction='none')
        clean_acc = (logits.argmax(dim=1) == y).float()
        ind_fail = (clean_acc == 1).nonzero().squeeze()
        if self.t == 0:
            return x, clean_acc, it
        if ind_fail.numel() == 0:
            ind_fail = torch.arange(b, device=x.device)

        x = self.check_shape(x[ind_fail])
        perturb = self.check_shape(perturb[ind_fail])
        mask = self.check_shape(mask[ind_fail])
        y = y[ind_fail]
        ind_all = ind_all[ind_fail]
        reinitial_count = reinitial_count[ind_fail]
        if target is not None:
            target = target[ind_fail]
        if ind_fail.numel() == 1:
            y.unsqueeze_(0)
            ind_all.unsqueeze_(0)
            reinitial_count.unsqueeze_(0)
            if target is not None:
                target.unsqueeze_(0)

        if self.verbose:
            acc_list = []

        # First loop
        perturb.requires_grad_()
        mask.requires_grad_()
        proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
        # proj_perturb = self.masking.apply(perturb, mask, self.k)
        with torch.no_grad():
            assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
            assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(
                x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(
                torch.min(x + proj_perturb).item(),
                torch.max(x + proj_perturb).item())
        logits = self.model(x + proj_perturb)

        loss = F.cross_entropy(logits, y, reduction='none')

        loss.sum().backward()
        grad_perturb = perturb.grad.clone()
        grad_mask = mask.grad.clone()

        for i in range(self.t):
            it[ind_all] += 1

            perturb = perturb.detach()
            mask = mask.detach()

            # update mask
            prev_mask = mask.clone()
            mask = self.update_mask(mask, grad_mask)

            # update perturbation using PGD
            perturb = self.update_perturbation(perturb=perturb, grad=grad_perturb, x=x)

            # forward pass
            perturb.requires_grad_()
            mask.requires_grad_()
            proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
            with torch.no_grad():
                assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
                assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(
                    x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(
                    torch.min(x + proj_perturb).item(),
                    torch.max(x + proj_perturb).item())
            logits = self.model(x + proj_perturb)

            # adaptive loss, calculate DLR loss for examples with low confidence, and use CE loss for the rest
            loss = F.cross_entropy(logits, y, reduction='none')

            # backward pass
            loss.sum().backward()
            grad_perturb = perturb.grad.clone()
            grad_mask = mask.grad.clone()

            logits = logits.detach()

            with torch.no_grad():
                fool_label = logits.argmax(dim=1)
                acc = (fool_label == y).float()

                # save the best adversarial example
                loss = loss.detach()
                loss_improve_idx = (loss >= loss_best[ind_all]).nonzero().squeeze()
                if loss_improve_idx.numel() > 0:
                    loss_best[ind_all[loss_improve_idx]] = loss[loss_improve_idx]
                    x_adv_best[ind_all[loss_improve_idx]] = (x + proj_perturb)[loss_improve_idx].detach().clone()

                ind_success = (acc == 0).nonzero().squeeze()
                if ind_success.numel() > 0:
                    x_adv_best[ind_all[ind_success]] = (x + proj_perturb)[ind_success].detach().clone()

                ind_fail = (acc == 1).nonzero().squeeze()
                if ind_fail.numel() > 0:
                    # count the number of times the mask is not updated
                    delta_mask_norm = torch.norm(
                        self.project_mask(mask[ind_fail]) - self.project_mask(prev_mask[ind_fail]), p=0, dim=(1, 2, 3))
                    ind_unchange = (delta_mask_norm <= 0).nonzero().squeeze()
                    if ind_unchange.numel() > 0:
                        if ind_fail.numel() == 1:
                            reinitial_count[ind_fail] += 1
                        else:
                            reinitial_count[ind_fail[ind_unchange]] += 1
                    else:
                        reinitial_count[ind_fail] = 0

                    # reinitialize mask and perturbation when the mask is not updated for 3 consecutive iterations
                    ind_reinit = (reinitial_count >= self.patience).nonzero().squeeze()
                    if ind_reinit.numel() > 0:
                        mask[ind_reinit] = self.initial_mask(x[ind_reinit])
                        reinitial_count[ind_reinit] = 0

                    # remove successfully attacked examples
                    x = self.check_shape(x[ind_fail])
                    perturb = self.check_shape(perturb[ind_fail])
                    mask = self.check_shape(mask[ind_fail])
                    grad_perturb = self.check_shape(grad_perturb[ind_fail])
                    grad_mask = self.check_shape(grad_mask[ind_fail])
                    y = y[ind_fail]
                    ind_all = ind_all[ind_fail]
                    reinitial_count = reinitial_count[ind_fail]
                    if target is not None:
                        target = target[ind_fail]
                    if ind_fail.numel() == 1:
                        y.unsqueeze_(0)
                        ind_all.unsqueeze_(0)
                        reinitial_count.unsqueeze_(0)
                        if target is not None:
                            target.unsqueeze_(0)

            if self.verbose and (i + 1) % self.verbose_interval == 0:
                acc_list.append(acc.sum().item())
            if torch.sum(acc) == 0.:
                break
        if training:
            self.model.train()
        if self.verbose:
            if len(acc_list) != self.t // self.verbose_interval:
                acc_list += [acc_list[-1]] * (self.t // self.verbose_interval - len(acc_list))
            return x_adv_best, acc, it, acc_list
        return x_adv_best, acc, it

    def perturb(self, x, y):
        if self.verbose:
            x_adv, acc, it, acc_list = self.__call__(x, y, targeted=False)
            return x_adv, acc.sum(), it, acc_list
        x_adv, acc, it = self.__call__(x, y, targeted=False)
        return x_adv, acc.sum(), it

    def change_masking(self):
        if isinstance(self.masking, MaskingA):
            self.masking = MaskingB()
        else:
            self.masking = MaskingA()


# ----------

# ---------- code taken from https://github.com/fra31/sparse-imperceivable-attacks/blob/master/pgd_attacks_pt.py
def project_L0_box(y, k, lb, ub):
    ''' projection of the batch y to a batch x such that:
          - each image of the batch x has at most k pixels with non-zero channels
          - lb <= x <= ub '''

    x = np.copy(y)
    p1 = np.sum(x ** 2, axis=-1)
    p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
    p2 = np.sum(p2 ** 2, axis=-1)
    p3 = np.sort(np.reshape(p1 - p2, [p2.shape[0], -1]))[:, -k]
    x = x * (np.logical_and(lb <= x, x <= ub)) + lb * (lb > x) + ub * (x > ub)
    x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)

    return x


def perturb_L0_box(attack, x_nat, y_nat, lb, ub, device):
    ''' PGD attack wrt L0-norm + box constraints

        it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
        such that:
          - each image of the batch adv differs from the corresponding one of
            x_nat in at most k pixels
          - lb <= adv - x_nat <= ub

        it returns also a vector of flags where 1 means no adversarial example found
        (in this case the original image is returned in adv) '''

    if attack.rs:
        x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
        x2 = np.clip(x2, 0, 1)
    else:
        x2 = np.copy(x_nat)

    adv_not_found = np.ones(y_nat.shape)
    adv = np.zeros(x_nat.shape)

    for i in (range(attack.num_steps)):
        if i > 0:
            pred, grad = get_predictions_and_gradients(attack.model, x2, y_nat, device)
            adv_not_found = np.minimum(adv_not_found, pred.astype(int))
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= (1e-10 + np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True))
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + attack.step_size * grad,
                        casting='unsafe')

        x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)

    return adv, adv_not_found


class PGDattack():
    def __init__(self, model, args):
        self.model = model
        self.type_attack = args['type_attack']  # 'L0', 'L0+Linf', 'L0+sigma'
        self.num_steps = args['num_steps']  # number of iterations of gradient descent for each restart
        self.step_size = args['step_size']  # step size for gradient descent (\eta in the paper)
        self.n_restarts = args['n_restarts']  # number of random restarts to perform
        self.rs = True  # random starting point
        self.epsilon = args['epsilon']  # for L0+Linf, the bound on the Linf-norm of the perturbation
        self.kappa = args[
            'kappa']  # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks
        self.k = args['sparsity']  # maximum number of pixels that can be modified (k_max in the paper)

    def perturb(self, x_nat, y_nat, device):
        adv = np.copy(x_nat)

        for counter in range(self.n_restarts):
            if counter == 0:
                corr_pred = get_predictions(self.model, x_nat, y_nat, device)
                pgd_adv_acc = np.copy(corr_pred)

            if self.type_attack == 'L0':
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, -x_nat, 1.0 - x_nat, device)

            elif self.type_attack == 'L0+Linf':
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.epsilon, -x_nat),
                                                               np.minimum(self.epsilon, 1.0 - x_nat))

            elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 1:
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat,
                                                               np.maximum(-self.kappa * self.sigma, -x_nat),
                                                               np.minimum(self.kappa * self.sigma, 1.0 - x_nat))

            pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
            adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
        return adv, pgd_adv_acc


# ----------

def PGD0(model, inputs, labels, epsilon, steps):
    args_pgd0 = {'type_attack': 'L0',
                 'n_restarts': 1,
                 'num_steps': steps,
                 'step_size': 120000.0 / 255.0,
                 'kappa': -1,
                 'epsilon': -1,
                 'sparsity': epsilon
                 }
    device = inputs.device
    attack = PGDattack(model, args_pgd0)
    x_test = inputs.permute(0, 2, 3, 1).cpu().numpy()
    y_test = labels.cpu().numpy()
    adv, _ = attack.perturb(x_test, y_test, device)
    x_torch = torch.from_numpy(adv).permute(0, 3, 1, 2).to(device)

    return x_torch


def sparse_rs(model: nn.Module, inputs: Tensor, labels: Tensor, epsilon: int = -1, steps: int = 400, norm: str = "L0",
              verbose=False):
    model = SingleChannelModel(model, inputs.shape[1:])
    bs, c, h, w = inputs.shape
    inputs = inputs.reshape(bs, 1, h, w * c)
    args_sparse = {'n_queries': steps, 'eps': epsilon, "norm": norm, "verbose": verbose}
    attack = RSAttack(model, **args_sparse)
    _, res = attack.perturb(inputs, labels)
    return res.reshape(bs, c, h, w)


def sparsefool(model: nn.Module, x: Tensor, y: Tensor, steps=20, lam=3, overshoot=0.02):
    atk = SparseFool(model=model, steps=steps, lam=lam, overshoot=overshoot)
    xadv = atk(x, y)
    return xadv


def BB_attack(model: nn.Module,
              inputs: Tensor,
              labels: Tensor,
              steps: int = 1000,
              **kwargs):
    device = next(model.parameters()).device
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    bb = L0BrendelBethgeAttack(steps=steps)
    _, advs, success = bb(fmodel, inputs, labels, epsilons=None)
    return advs


def dataset_BB_attack(model: nn.Module,
                      inputs: Tensor,
                      labels: Tensor,
                      steps=1000,
                      **kwargs):
    device = next(model.parameters()).device
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    dataset_atk = DatasetAttack()

    dataset_atk.feed(fmodel, inputs)

    atk = L0BrendelBethgeAttack(init_attack=dataset_atk, steps=steps)
    _, advs, success = atk(fmodel, inputs, labels, epsilons=None)

    return advs


def EAD_attack(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               steps=10000,
               **kwargs):
    device = next(model.parameters()).device
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    atk = EADAttack(steps=steps)
    _, advs, success = atk(fmodel, inputs, labels, epsilons=None)
    return advs


def sparse_PGD(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               epsilon: float = 255 / 255,
               k: int = 20,
               steps: int = 10000,
               unprojected_gradient: bool = True,
               patience: int = 3,
               alpha: float = 0.25,
               beta: float = 0.25,
               verbose: bool = False,
               **kwargs):
    model = SingleChannelModel(model, inputs.shape[1:])
    bs, c, h, w = inputs.shape
    inputs = inputs.reshape(bs, 1, h, w * c)
    attacker = SparsePGD(model=model, epsilon=epsilon, k=k, t=steps, unprojected_gradient=unprojected_gradient,
                         patience=patience, alpha=alpha, beta=beta, verbose=verbose)
    if verbose:
        advs, _, _, _ = attacker.perturb(inputs, labels)
    else:
        advs, _, _ = attacker.perturb(inputs, labels)
    return advs.reshape(bs, c, h, w)
