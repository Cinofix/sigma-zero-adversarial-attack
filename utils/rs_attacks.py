# code taken by the official Sparse-RS repo: https://github.com/fra31/sparse-rs/blob/master/rs_attacks.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
import math
import torch.nn.functional as F

import numpy as np
import copy
import sys
from utilities import Logger
import os


class RSAttack():
    """
    Sparse-RS attacks

    :param predict:           forward pass function
    :param norm:              type of the attack
    :param n_restarts:        number of random restarts
    :param n_queries:         max number of queries (each restart)
    :param eps:               bound on the sparsity of perturbations
    :param seed:              random seed for the starting point
    :param alpha_init:        parameter to control alphai
    :param loss:              loss function optimized ('margin', 'ce' supported)
    :param resc_schedule      adapt schedule of alphai to n_queries
    :param device             specify device to use
    :param log_path           path to save logfile.txt
    :param constant_schedule  use constant alphai
    :param targeted           perform targeted attacks
    :param init_patches       initialization for patches
    :param resample_loc       period in queries of resampling images and
                              locations for universal attacks
    :param data_loader        loader to get new images for resampling
    :param update_loc_period  period in queries of updates of the location
                              for image-specific patches
    """

    def __init__(
            self,
            predict,
            norm='L0',
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            verbose=True,
            targeted=False,
            loss='margin',
            resc_schedule=True,
            device=None,
            log_path=None,
            constant_schedule=False,
            init_patches='random_squares',
            resample_loc=None,
            data_loader=None,
            update_loc_period=None):
        """
        Sparse-RS implementation in PyTorch
        """

        self.predict = predict
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        self.logger = Logger(log_path)
        self.constant_schedule = constant_schedule
        self.init_patches = init_patches
        self.resample_loc = n_queries // 10 if resample_loc is None else resample_loc
        self.data_loader = data_loader
        self.update_loc_period = update_loc_period if not update_loc_period is None else 4 if not targeted else 10

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """

        logits = self.predict(x)
        xent = F.cross_entropy(logits, y, reduction='none')
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            return y_others - y_corr, xent

    def init_hyperparam(self, x):
        assert self.norm in ['L0', 'patches', 'frames',
                             'patches_universal', 'frames_universal']
        assert not self.eps is None
        assert self.loss in ['ce', 'margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        if self.targeted:
            self.loss = 'ce'

    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 'patches' in self.norm:
            if 10 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 8
            elif 500 < it <= 1000:
                p = self.p_init / 16
            elif 1000 < it <= 2000:
                p = self.p_init / 32
            elif 2000 < it <= 4000:
                p = self.p_init / 64
            elif 4000 < it <= 6000:
                p = self.p_init / 128
            elif 6000 < it <= 8000:
                p = self.p_init / 256
            elif 8000 < it:
                p = self.p_init / 512
            else:
                p = self.p_init

        elif 'frames' in self.norm:
            if not 'universal' in self.norm:
                tot_qr = 10000 if self.rescale_schedule else self.n_queries
                p = max((float(tot_qr - it) / tot_qr - .5) * self.p_init * self.eps ** 2, 0.)
                return 3. * math.ceil(p)

            else:
                assert self.rescale_schedule
                its = [200, 600, 1200, 1800, 2500, 10000, 100000]
                resc_factors = [1., .8, .6, .4, .2, .1, 0.]
                c = 0
                while it >= its[c]:
                    c += 1
                return resc_factors[c] * self.p_init

        elif 'L0' in self.norm:
            if 0 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 5
            elif 500 < it <= 1000:
                p = self.p_init / 6
            elif 1000 < it <= 2000:
                p = self.p_init / 8
            elif 2000 < it <= 4000:
                p = self.p_init / 10
            elif 4000 < it <= 6000:
                p = self.p_init / 12
            elif 6000 < it <= 8000:
                p = self.p_init / 15
            elif 8000 < it:
                p = self.p_init / 20
            else:
                p = self.p_init

            if self.constant_schedule:
                p = self.p_init / 2

        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t

    def get_init_patch(self, c, s, n_iter=1000):
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 1.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 1.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'sh':
            patch_univ = torch.ones([1, c, s, s]).to(self.device)

        return patch_univ.clamp(0., 1.)

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]

            if self.norm == 'L0':
                eps = self.eps

                x_best = x.clone()
                n_pixels = h * w
                b_all, be_all = torch.zeros([x.shape[0], eps]).long().to(self.device), torch.zeros([x.shape[0], n_pixels - eps]).long().to(self.device)
                for img in range(x.shape[0]):
                    ind_all = torch.randperm(n_pixels)
                    ind_p = ind_all[:eps]
                    ind_np = ind_all[eps:]
                    x_best[img, :, ind_p // w, ind_p % w] = self.random_choice([c, eps]).clamp(0., 1.)
                    b_all[img] = ind_p.clone()
                    be_all[img] = ind_np.clone()

                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)

                for it in range(1, self.n_queries):
                    # check points still to fool
                    idx_to_fool = (margin_min > 0.).nonzero().squeeze().to(self.device)
                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    b_curr, be_curr = b_all[idx_to_fool], be_all[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)
                        b_curr.unsqueeze_(0)
                        be_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)

                    # build new candidate
                    x_new = x_best_curr.clone()
                    eps_it = max(int(self.p_selection(it) * eps), 1)
                    ind_p = torch.randperm(eps)[:eps_it]
                    ind_np = torch.randperm(n_pixels - eps)[:eps_it]

                    for img in range(x_new.shape[0]):
                        p_set = b_curr[img, ind_p]
                        np_set = be_curr[img, ind_np]
                        x_new[img, :, p_set // w, p_set % w] = x_curr[img, :, p_set // w, p_set % w].clone()
                        if eps_it > 1:
                            x_new[img, :, np_set // w, np_set % w] = self.random_choice([c, eps_it]).clamp(0., 1.)
                        else:
                            # if update is 1x1 make sure the sampled color is different from the current one
                            old_clr = x_new[img, :, np_set // w, np_set % w].clone()
                            assert old_clr.shape == (c, 1), print(old_clr)
                            new_clr = old_clr.clone()
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice([c, 1]).clone().clamp(0., 1.)
                            x_new[img, :, np_set // w, np_set % w] = new_clr.clone()

                    # compute loss of the new candidates
                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool] += 1

                    # update best solution
                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        x_best[idx_to_fool[idx_improved]] = x_new[idx_improved].clone()
                        t = b_curr[idx_improved].clone()
                        te = be_curr[idx_improved].clone()

                        if nimpr > 1:
                            t[:, ind_p] = be_curr[idx_improved][:, ind_np] + 0
                            te[:, ind_np] = b_curr[idx_improved][:, ind_p] + 0
                        else:
                            t[ind_p] = be_curr[idx_improved][ind_np] + 0
                            te[ind_np] = b_curr[idx_improved][ind_p] + 0

                        b_all[idx_to_fool[idx_improved]] = t.clone()
                        be_all[idx_to_fool[idx_improved]] = te.clone()

                    # log results_1 current iteration
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      ind_succ.numel(), n_ex_total,
                                                      float(ind_succ.numel()) / n_ex_total),
                                                  '- avg # queries={:.1f}'.format(
                                                      n_queries[ind_succ].mean().item()),
                                                  '- med # queries={:.1f}'.format(
                                                      n_queries[ind_succ].median().item()),
                                                  '- loss={:.3f}'.format(loss_min.mean()),
                                                  '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                      -1).max()),
                                                  '- epsit={:.0f}'.format(eps_it),
                                                  ]))

                    if ind_succ.numel() == n_ex_total:
                        break

            elif self.norm == 'patches':
                ''' assumes square images and patches '''

                s = int(math.ceil(self.eps ** .5))
                x_best = x.clone()
                x_new = x.clone()
                loc = torch.randint(h - s, size=[x.shape[0], 2])
                patches_coll = torch.zeros([x.shape[0], c, s, s]).to(self.device)
                assert abs(self.update_loc_period) > 1
                loc_t = abs(self.update_loc_period)

                # set when to start single channel updates
                it_start_cu = None
                for it in range(0, self.n_queries):
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    if s_it == 1:
                        break
                it_start_cu = it + (self.n_queries - it) // 2
                if self.verbose:
                    self.logger.log('starting single channel updates at query {}'.format(
                        it_start_cu))

                # initialize patches
                if self.verbose:
                    self.logger.log('using {} initialization'.format(self.init_patches))
                for counter in range(x.shape[0]):
                    patches_coll[counter] += self.get_init_patch(c, s).squeeze().clamp(0., 1.)
                    x_new[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                    loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()

                margin_min, loss_min = self.margin_and_loss(x_new, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)

                for it in range(1, self.n_queries):
                    # check points still to fool
                    idx_to_fool = (margin_min > -1e-6).nonzero().squeeze()
                    x_curr = self.check_shape(x[idx_to_fool])
                    patches_curr = self.check_shape(patches_coll[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    loc_curr = loc[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)

                        loc_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)

                    # sample update
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    p_it = torch.randint(s - s_it + 1, size=[2])
                    sh_it = int(max(self.sh_selection(it) * h, 0))
                    patches_new = patches_curr.clone()
                    x_new = x_curr.clone()
                    loc_new = loc_curr.clone()
                    update_loc = int((it % loc_t == 0) and (sh_it > 0))
                    update_patch = 1. - update_loc
                    if self.update_loc_period < 0 and sh_it > 0:
                        update_loc = 1. - update_loc
                        update_patch = 1. - update_patch
                    for counter in range(x_curr.shape[0]):
                        if update_patch == 1.:
                            # update patch
                            if it < it_start_cu:
                                if s_it > 1:
                                    patches_new[counter, :, p_it[0]:p_it[0] + s_it,
                                    p_it[1]:p_it[1] + s_it] += self.random_choice([c, 1, 1])
                                else:
                                    # make sure to sample a different color
                                    old_clr = patches_new[counter, :, p_it[0]:p_it[0] + s_it,
                                              p_it[1]:p_it[1] + s_it].clone()
                                    new_clr = old_clr.clone()
                                    while (new_clr == old_clr).all().item():
                                        new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                                    patches_new[counter, :, p_it[0]:p_it[0] + s_it,
                                    p_it[1]:p_it[1] + s_it] = new_clr.clone()
                            else:
                                assert s_it == 1
                                assert it >= it_start_cu
                                # single channel updates
                                new_ch = self.random_int(low=0, high=3, shape=[1])
                                patches_new[counter, new_ch, p_it[0]:p_it[0] + s_it,
                                p_it[1]:p_it[1] + s_it] = 1. - patches_new[
                                                               counter, new_ch, p_it[0]:p_it[0] + s_it,
                                                               p_it[1]:p_it[1] + s_it]

                            patches_new[counter].clamp_(0., 1.)
                        if update_loc == 1:
                            # update location
                            loc_new[counter] += (torch.randint(low=-sh_it, high=sh_it + 1, size=[2]))
                            loc_new[counter].clamp_(0, h - s)

                        x_new[counter, :, loc_new[counter, 0]:loc_new[counter, 0] + s,
                        loc_new[counter, 1]:loc_new[counter, 1] + s] = patches_new[counter].clone()

                    # check loss of new candidate
                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool] += 1

                    # update best solution
                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        patches_coll[idx_to_fool[idx_improved]] = patches_new[idx_improved].clone()
                        loc[idx_to_fool[idx_improved]] = loc_new[idx_improved].clone()

                    # log results_1 current iteration
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      ind_succ.numel(), n_ex_total,
                                                      float(ind_succ.numel()) / n_ex_total),
                                                  '- avg # queries={:.1f}'.format(
                                                      n_queries[ind_succ].mean().item()),
                                                  '- med # queries={:.1f}'.format(
                                                      n_queries[ind_succ].median().item()),
                                                  '- loss={:.3f}'.format(loss_min.mean()),
                                                  '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                      -1).max()),
                                                  # '- sit={:.0f} - sh={:.0f}'.format(s_it, sh_it),
                                                  '{}'.format(' - loc' if update_loc == 1. else ''),
                                                  ]))

                    if ind_succ.numel() == n_ex_total:
                        break

                # apply patches
                for counter in range(x.shape[0]):
                    x_best[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                    loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()

            elif self.norm == 'patches_universal':
                ''' assumes square images and patches '''

                s = int(math.ceil(self.eps ** .5))
                x_best = x.clone()
                self.n_imgs = x.shape[0]
                x_new = x.clone()
                loc = torch.randint(h - s + 1, size=[x.shape[0], 2])

                # set when to start single channel updates
                it_start_cu = None
                for it in range(0, self.n_queries):
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    if s_it == 1:
                        break
                it_start_cu = it + (self.n_queries - it) // 2
                if self.verbose:
                    self.logger.log('starting single channel updates at query {}'.format(
                        it_start_cu))

                # initialize patch
                if self.verbose:
                    self.logger.log('using {} initialization'.format(self.init_patches))
                patch_univ = self.get_init_patch(c, s)
                it_init = 0

                loss_batch = float(1e10)
                n_succs = 0
                n_iter = self.n_queries

                # init update batch
                assert not self.data_loader is None
                assert not self.resample_loc is None
                assert self.targeted
                new_train_imgs = []
                n_newimgs = self.n_imgs + 0
                n_imgsneeded = math.ceil(self.n_queries / self.resample_loc) * n_newimgs
                tot_imgs = 0
                if self.verbose:
                    self.logger.log('imgs updated={}, imgs needed={}'.format(
                        n_newimgs, n_imgsneeded))
                while tot_imgs < min(100000, n_imgsneeded):
                    x_toupdatetrain, _ = next(self.data_loader)
                    new_train_imgs.append(x_toupdatetrain)
                    tot_imgs += x_toupdatetrain.shape[0]
                newimgstoadd = torch.cat(new_train_imgs, axis=0)
                counter_resamplingimgs = 0

                for it in range(it_init, n_iter):
                    # sample size and location of the update
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    p_it = torch.randint(s - s_it + 1, size=[2])

                    patch_new = patch_univ.clone()

                    if s_it > 1:
                        patch_new[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += self.random_choice([c, 1, 1])
                    else:
                        old_clr = patch_new[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                        new_clr = old_clr.clone()
                        if it < it_start_cu:
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice(new_clr).clone().clamp(0., 1.)
                        else:
                            # single channel update
                            new_ch = self.random_int(low=0, high=3, shape=[1])
                            new_clr[new_ch] = 1. - new_clr[new_ch]

                        patch_new[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()

                    patch_new.clamp_(0., 1.)

                    # compute loss for new candidate
                    x_new = x.clone()

                    for counter in range(x.shape[0]):
                        loc_new = loc[counter]
                        x_new[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] = 0.
                        x_new[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] += patch_new[0]

                    margin_run, loss_run = self.margin_and_loss(x_new, y)
                    if self.loss == 'ce':
                        loss_run += x_new.shape[0]
                    loss_new = loss_run.sum()
                    n_succs_new = (margin_run < -1e-6).sum().item()

                    # accept candidate if loss improves
                    if loss_new < loss_batch:
                        is_accepted = True
                        loss_batch = loss_new + 0.
                        patch_univ = patch_new.clone()
                        n_succs = n_succs_new + 0
                    else:
                        is_accepted = False

                    # sample new locations and images
                    if (it + 1) % self.resample_loc == 0:
                        newimgstoadd_it = newimgstoadd[counter_resamplingimgs * n_newimgs:(
                                                                                                  counter_resamplingimgs + 1) * n_newimgs].clone().cuda()
                        new_batch = [x[n_newimgs:].clone(), newimgstoadd_it.clone()]
                        x = torch.cat(new_batch, dim=0)
                        assert x.shape[0] == self.n_imgs

                        loc = torch.randint(h - s + 1, size=[self.n_imgs, 2])
                        assert loc.shape == (self.n_imgs, 2)

                        loss_batch = loss_batch * 0. + 1e6
                        counter_resamplingimgs += 1

                    # logging current iteration
                    if self.verbose:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      n_succs, n_ex_total,
                                                      float(n_succs) / n_ex_total),
                                                  '- loss={:.3f}'.format(loss_batch),
                                                  '- max pert={:.0f}'.format(((x_new - x).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                      -1).max()),
                                                  ]))

                # apply patches on the initial images
                for counter in range(x_best.shape[0]):
                    loc_new = loc[counter]
                    x_best[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] = 0.
                    x_best[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] += patch_univ[0]

            elif self.norm == 'frames':
                # set width and indices of frames
                mask = torch.zeros(x.shape[-2:])
                s = self.eps + 0
                mask[:s] = 1.
                mask[-s:] = 1.
                mask[:, :s] = 1.
                mask[:, -s:] = 1.
                ind = (mask == 1.).nonzero().squeeze()
                eps = ind.shape[0]
                x_best = x.clone()
                x_new = x.clone()
                mask = mask.view(1, 1, h, w).to(self.device)
                mask_frame = torch.ones([1, c, h, w], device=x.device) * mask
                #

                # set when starting single channel updates
                it_start_cu = None
                for it in range(0, self.n_queries):
                    s_it = int(max(self.p_selection(it), 1))
                    if s_it == 1:
                        break
                it_start_cu = it + (self.n_queries - it) // 2
                # it_start_cu = 10000
                if self.verbose:
                    self.logger.log('starting single channel updates at query {}'.format(
                        it_start_cu))

                # initialize frames
                x_best[:, :, ind[:, 0], ind[:, 1]] = self.random_choice(
                    [x.shape[0], c, eps]).clamp(0., 1.)

                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)

                for it in range(1, self.n_queries):
                    # check points still to fool
                    idx_to_fool = (margin_min > -1e-6).nonzero().squeeze()
                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)

                    # sample update
                    s_it = max(int(self.p_selection(it)), 1)
                    ind_it = torch.randperm(eps)[0]

                    x_new = x_best_curr.clone()
                    if s_it > 1:
                        dir_h = self.random_choice([1]).long().cpu()
                        dir_w = self.random_choice([1]).long().cpu()
                        new_clr = self.random_choice([c, 1]).clamp(0., 1.)

                    for counter in range(x_curr.shape[0]):
                        if s_it > 1:
                            for counter_h in range(s_it):
                                for counter_w in range(s_it):
                                    x_new[counter, :, (ind[ind_it, 0] + dir_h * counter_h).clamp(0, h - 1),
                                    (ind[ind_it, 1] + dir_w * counter_w).clamp(0, w - 1)] = new_clr.clone()
                        else:
                            p_it = ind[ind_it].clone()
                            old_clr = x_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                            new_clr = old_clr.clone()
                            if it < it_start_cu:
                                while (new_clr == old_clr).all().item():
                                    new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                            else:
                                # single channel update
                                new_ch = self.random_int(low=0, high=3, shape=[1])
                                new_clr[new_ch] = 1. - new_clr[new_ch]
                            x_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()

                    x_new.clamp_(0., 1.)
                    x_new = (x_new - x_curr) * mask_frame + x_curr

                    # check loss of new candidate
                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool] += 1

                    # update best solution
                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        x_best[idx_to_fool[idx_improved]] = x_new[idx_improved].clone()

                    # log results_1 current iteration
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      ind_succ.numel(), n_ex_total,
                                                      float(ind_succ.numel()) / n_ex_total),
                                                  '- avg # queries={:.1f}'.format(
                                                      n_queries[ind_succ].mean().item()),
                                                  '- med # queries={:.1f}'.format(
                                                      n_queries[ind_succ].median().item()),
                                                  '- loss={:.3f}'.format(loss_min.mean()),
                                                  '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                      -1).max()),
                                                  # '- min pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                                                  # ).max(1)[0].view(x_new.shape[0], -1).sum(-1).min()),
                                                  # '- sit={:.0f} - indit={}'.format(s_it, ind_it.item()),
                                                  ]))

                    if ind_succ.numel() == n_ex_total:
                        break


            elif self.norm == 'frames_universal':
                # set width and indices of frames
                mask = torch.zeros(x.shape[-2:])
                s = self.eps + 0
                mask[:s] = 1.
                mask[-s:] = 1.
                mask[:, :s] = 1.
                mask[:, -s:] = 1.
                ind = (mask == 1.).nonzero().squeeze()
                eps = ind.shape[0]
                x_best = x.clone()
                x_new = x.clone()
                mask = mask.view(1, 1, h, w).to(self.device)
                mask_frame = torch.ones([1, c, h, w], device=x.device) * mask
                frame_univ = self.random_choice([1, c, eps]).clamp(0., 1.)

                # set when to start single channel updates
                it_start_cu = None
                for it in range(0, self.n_queries):
                    s_it = int(max(self.p_selection(it) * s, 1))
                    if s_it == 1:
                        break
                it_start_cu = it + (self.n_queries - it) // 2
                if self.verbose:
                    self.logger.log('starting single channel updates at query {}'.format(
                        it_start_cu))

                self.n_imgs = x.shape[0]
                loss_batch = float(1e10)
                n_queries = torch.ones_like(y).float()

                # init update batch
                assert not self.data_loader is None
                assert not self.resample_loc is None
                assert self.targeted
                new_train_imgs = []
                n_newimgs = self.n_imgs + 0
                n_imgsneeded = math.ceil(self.n_queries / self.resample_loc) * n_newimgs
                tot_imgs = 0
                if self.verbose:
                    self.logger.log('imgs updated={}, imgs needed={}'.format(
                        n_newimgs, n_imgsneeded))
                while tot_imgs < min(100000, n_imgsneeded):
                    x_toupdatetrain, _ = next(self.data_loader)
                    new_train_imgs.append(x_toupdatetrain)
                    tot_imgs += x_toupdatetrain.shape[0]
                newimgstoadd = torch.cat(new_train_imgs, axis=0)
                counter_resamplingimgs = 0

                for it in range(self.n_queries):
                    # sample update
                    s_it = max(int(self.p_selection(it) * self.eps), 1)
                    ind_it = torch.randperm(eps)[0]

                    mask_frame[:, :, ind[:, 0], ind[:, 1]] = 0
                    mask_frame[:, :, ind[:, 0], ind[:, 1]] += frame_univ

                    if s_it > 1:
                        dir_h = self.random_choice([1]).long().cpu()
                        dir_w = self.random_choice([1]).long().cpu()
                        new_clr = self.random_choice([c, 1]).clamp(0., 1.)

                        for counter_h in range(s_it):
                            for counter_w in range(s_it):
                                mask_frame[0, :, (ind[ind_it, 0] + dir_h * counter_h).clamp(0, h - 1),
                                (ind[ind_it, 1] + dir_w * counter_w).clamp(0, w - 1)] = new_clr.clone()
                    else:
                        p_it = ind[ind_it]
                        old_clr = mask_frame[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                        new_clr = old_clr.clone()
                        if it < it_start_cu:
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                        else:
                            # single channel update
                            new_ch = self.random_int(low=0, high=3, shape=[1])
                            new_clr[new_ch] = 1. - new_clr[new_ch]
                        mask_frame[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()

                    frame_new = mask_frame[:, :, ind[:, 0], ind[:, 1]].clone()
                    frame_new.clamp_(0., 1.)
                    if len(frame_new.shape) == 2:
                        frame_new.unsqueeze_(0)

                    x_new[:, :, ind[:, 0], ind[:, 1]] = 0.
                    x_new[:, :, ind[:, 0], ind[:, 1]] += frame_new

                    margin_run, loss_run = self.margin_and_loss(x_new, y)
                    if self.loss == 'ce':
                        loss_run += x_new.shape[0]
                    loss_new = loss_run.sum()
                    n_succs_new = (margin_run < -1e-6).sum().item()

                    # accept candidate if loss improves
                    if loss_new < loss_batch:
                        # is_accepted = True
                        loss_batch = loss_new + 0.
                        frame_univ = frame_new.clone()
                        n_succs = n_succs_new + 0

                    # sample new images
                    if (it + 1) % self.resample_loc == 0:
                        newimgstoadd_it = newimgstoadd[counter_resamplingimgs * n_newimgs:(
                                                                                                  counter_resamplingimgs + 1) * n_newimgs].clone().cuda()
                        new_batch = [x[n_newimgs:].clone(), newimgstoadd_it.clone()]
                        x = torch.cat(new_batch, dim=0)
                        assert x.shape[0] == self.n_imgs

                        loss_batch = loss_batch * 0. + 1e6
                        x_new = x.clone()
                        counter_resamplingimgs += 1

                    # loggin current iteration
                    if self.verbose:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      n_succs, n_ex_total,
                                                      float(n_succs) / n_ex_total),
                                                  '- loss={:.3f}'.format(loss_batch),
                                                  '- max pert={:.0f}'.format(((x_new - x).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                      -1).max()),
                                                  ]))

                # apply frame on initial images
                x_best[:, :, ind[:, 0], ind[:, 1]] = 0.
                x_best[:, :, ind[:, 0], ind[:, 1]] += frame_univ

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)

        adv = x.clone()
        qr = torch.zeros([x.shape[0]]).to(self.device)
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.predict(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    output = self.predict(x)
                    n_classes = output.shape[-1]
                    y_pred = output.max(1)[1]
                    y = self.random_target_classes(y_pred, n_classes)
        else:
            y = y.detach().clone().long().to(self.device)

        if not self.targeted:
            acc = self.predict(x).max(1)[1] == y
        else:
            acc = self.predict(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        np.random.seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                qr_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.predict(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                qr[ind_to_fool[ind_curr]] = qr_curr[ind_curr].clone()
                if self.verbose:
                    print('restart {} - robust accuracy: {:.2%}'.format(
                        counter, acc.float().mean()),
                        '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

        return qr, adv
