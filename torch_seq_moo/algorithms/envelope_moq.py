import hydra
import wandb
import copy
import math
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from torch_seq_moo.algorithms.base import BaseAlgorithm
from torch_seq_moo.algorithms.mogfn_utils.utils import mean_pairwise_distances, generate_simplex, thermometer, plot_pareto, pareto_frontier
from torch_seq_moo.utils import str_to_tokens, tokens_to_str
from torch_seq_moo.metrics import get_all_metrics

from torch.distributions import Categorical
from tqdm import tqdm
from dataclasses import dataclass
from collections import namedtuple, deque


class EnvelopeMOQ(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(EnvelopeMOQ, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        # EMOQ stuff
        self.train_steps = cfg.train_steps
        self.epsilon = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.therm_n_bins = cfg.therm_n_bins
        self.pref_use_therm = cfg.pref_use_therm
        self.pref_cond = cfg.pref_cond
        self.pref_alpha = cfg.pref_alpha
        
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon_delta = (cfg.epsilon - 0.05) / cfg.train_steps

        self.mem_size = cfg.mem_size
        self.batch_size = cfg.batch_size
        self.weight_num = cfg.weight_num

        self.beta = cfg.beta
        self.beta_init = cfg.beta
        self.homotopy = cfg.homotopy
        self.beta_uplim = 1.00
        self.tau = 1000.
        self.beta_expbase = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./cfg.train_steps))
        self.beta_delta = self.beta_expbase / self.tau

        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        self.w_kept = None
        self.update_count = 0
        self.update_freq = cfg.update_freq
        self.num_actions = cfg.model.num_actions

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.k = cfg.k
        self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.simplex = generate_simplex(self.obj_dim, cfg.simplex_bins)

    def init_policy(self):
        cfg = self.cfg
        pref_dim = self.therm_n_bins * self.obj_dim if self.pref_use_therm else self.obj_dim
        cond_dim = pref_dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim)
        self.model_ = copy.deepcopy(self.model)

        self.model.to(self.device)
        self.model_.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))


    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.obj_dim)
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} HV: {:.3f} R2: {:.3f} HSRI: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))

        for i in pb:
            
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(task, plot=True)
                hv, r2, hsri = mo_metrics["hypervolume"], mo_metrics["r2"], mo_metrics["hsri"]
                
                self.log(dict(
                    topk_rewards=topk_metrics[0].mean(),
                    topk_diversity=topk_metrics[1].mean(),
                    sample_r=rs.mean()
                ), commit=False)

                self.log({key: val for key, val in mo_metrics.items()}, commit=False)

                if fig is not None:
                    self.log(dict(
                        pareto_front=fig
                    ), commit=False)
                table = wandb.Table(columns = ["Sequence", "Rewards", "Prefs"])
                for sample, rew, pref in zip(samples, all_rews, self.simplex):
                    table.add_data(str(sample), str(rew), str(pref))
                self.log({"generated_seqs": table})
            self.log(dict(
                train_loss=loss,
                train_rewards=r,
            ))
            pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        
        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol_rel': hv
        }
    
    def train_step(self, task, batch_size):
        cond_var, prefs = self._get_condition_var(train=True, bs=batch_size)
        states, loss, r = self.run_episodes(batch_size, cond_var, prefs, task)

        return loss, r.mean()

    def run_episodes(self, episodes, cond_var=None, prefs=None, task=None, train=True):
        states = [''] * episodes
        if cond_var is None:
            cond_var, prefs = self._get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1]
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.epsilon).to(self.device)
        prefs = torch.from_numpy(prefs).unsqueeze(0).to(self.device).float()
        # prefs = torch.tile(prefs, (episodes, 1))
        loss = 0
        rewards = torch.zeros(episodes, self.obj_dim)
        for t in (range(self.max_len) if episodes > 0 else []):
            _, Q = self.model(x, cond_var, prefs)
            Q = Q.view(-1, self.num_actions, self.obj_dim)
            if t <= self.min_len:
                Q[:, 0, :] = -100 # Prevent model from stopping
                                     # without having output anything

            # cat = Categorical(logits=logits / self.sampling_temp)
            # actions = cat.sample()
            # Q = prefs @ Q.view(-1, self.obj_dim)
            Q = torch.mv(Q.view(-1, self.obj_dim), prefs[0])
            Q = Q.view(-1, self.num_actions)
            actions = Q.max(1)[1]
            
            if train and self.epsilon > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), Q.shape[1], (episodes, )).to(self.device), actions)
            
            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)
            
            next_x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if train:
                # print(t, active_mask)
                for i in range(len(active_mask)):
                    if active_mask[i] == 1:
                        if actions[i] == 0 or t==self.max_len-1:
                            # import pdb; pdb.set_trace();
                            rewards[i] = torch.from_numpy(task.score(tokens_to_str([x.t()[i]], self.tokenizer)))
                        with torch.no_grad():
                            self.store(
                                x.t()[i],
                                actions[i],
                                next_x.t()[i],
                                rewards[i] if actions[i] == 0 else torch.zeros(self.obj_dim).to(self.device),
                                1 if actions[i] == 0 else 0
                            )
                loss += self.learn()

            x = next_x

            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        return states, loss, rewards

    def learn(self):
        if len(self.trans_mem) > self.batch_size * self.max_len:
            self.update_count += 1

            minibatch = self.sample_transitions(self.trans_mem, self.priority_mem, self.batch_size)
            
            batchify = lambda x: list(x) * self.weight_num
            
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0).t(), minibatch))
            action_batch = batchify(map(lambda x: x.a.unsqueeze(0), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0).t(), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))

            w_batch = np.random.randn(self.weight_num, self.obj_dim)
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).float()
            cond_var = thermometer(w_batch, self.therm_n_bins, 0, 1).reshape(w_batch.shape[0], -1).to(self.device)
            w_batch = w_batch.to(self.device)
            # cond_var = torch.tile(cond_var, (self.weight_num, 1))
            # prefs = torch.tile(prefs, (self.weight_num, 1))
            __, Q = self.model_(pad_sequence(state_batch).squeeze(2), cond_var, w_batch, w_num=self.weight_num)

            # detach since we don't want gradients to propagate
            # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
            # 					  Variable(w_batch, volatile=True), w_num=self.weight_num)
            _, DQ = self.model_(pad_sequence(next_state_batch).squeeze(2), cond_var, w_batch)
            w_ext = w_batch.unsqueeze(2).repeat(1, self.model.num_actions, 1)
            w_ext = w_ext.view(-1, self.obj_dim)
            _, tmpQ = self.model(pad_sequence(next_state_batch).squeeze(2), cond_var, w_batch)

            tmpQ = tmpQ.view(-1, self.obj_dim)
            # print(torch.bmm(w_ext.unsqueeze(1),
            # 			    tmpQ.data.unsqueeze(2)).view(-1, action_size))
            act = torch.bmm(w_ext.unsqueeze(1), tmpQ.unsqueeze(2)).view(-1, self.model.num_actions).max(1)[1]

            HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q = torch.zeros(self.batch_size * self.weight_num,
                                             self.obj_dim).to(self.device)
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                # Tau_Q.volatile = False
                Tau_Q += torch.cat(reward_batch, dim=0)

            actions = torch.cat(action_batch, dim=0)

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                         ).view(-1, self.obj_dim)
            Tau_Q = Tau_Q.view(-1, self.obj_dim)

            wQ = torch.bmm(w_batch.unsqueeze(1), Q.unsqueeze(2)).squeeze()

            wTQ = torch.bmm(w_batch.unsqueeze(1), Tau_Q.unsqueeze(2)).squeeze()

            # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
            loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))

            self.opt.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.opt.step()

            if self.update_count % self.update_freq == 0:
                self.model_.load_state_dict(self.model.state_dict())

            return loss.item()

        return 0.0

    def actmsk(self, num_dim, index):
        mask = torch.ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = torch.ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).long()
        inds = inds[mask.eq(0)]
        return inds.to(self.device)

    def store(self, state, action, next_state, reward, terminal):
        self.trans_mem.append(self.trans(
            state,
            action,
            next_state,
            reward,
            terminal
        ))
        cond_var, prefs = self._get_condition_var(train=True, bs=1)
        prefs = torch.from_numpy(prefs).unsqueeze(0).float().to(self.device)
        _, q = self.model(state.unsqueeze(0).t(), cond_var, prefs)
        q = q.view(-1, self.num_actions, self.obj_dim)
        q = q[0, action, :]
        wq = prefs[0].dot(q)

        wr = prefs[0].dot(reward)
        if not terminal:
            # next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, _ = self.model(next_state.unsqueeze(0).t(), cond_var, prefs)
            hq = hq.data[0]
            whq = prefs[0].dot(hq)
            p = abs(wr + self.gamma * whq - wq)
        else:
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init) * self.beta_expbase + self.beta_init - self.beta
            p = abs(wr - wq)
        p += 1e-5

        self.priority_mem.append(p.cpu().numpy())

        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample_transitions(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def process_reward(self, seqs, prefs, task, rewards=None):
        if rewards is None:
            rewards = task.score(seqs)
            rewards = ((rewards - 0) * 2) + -1 # shape rewards to improve learning
        r = (torch.tensor(prefs) * (rewards)).sum(axis=1)
        return r

    def evaluation(self, task, plot=False):
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        topk_rs = []
        topk_div = []
        for prefs in self.simplex:
            cond_var, _ = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
            samples, _, _ = self.run_episodes(self.num_samples, cond_var, prefs=prefs, task=task, train=False)
            rewards = task.score(samples)
            r = self.process_reward(samples, prefs, task, rewards=rewards)
            
            # topk metrics
            topk_r, topk_idx = torch.topk(r, self.k)
            samples = np.array(samples)
            topk_seq = samples[topk_idx].tolist()
            edit_dist = mean_pairwise_distances(topk_seq)
            topk_rs.append(topk_r.mean().item())
            topk_div.append(edit_dist)
            
            # top 1 metrics
            max_idx = r.argmax()
            new_candidates.append(samples[max_idx])
            all_rewards.append(rewards[max_idx])
            r_scores.append(r.max().item())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)
        
        # filter to get current pareto front 
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        
        mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, r2_prefs=self.simplex, num_obj=self.obj_dim)
        fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False) if plot else None        
        return new_candidates, all_rewards, r_scores, mo_metrics, (np.array(topk_rs), np.array(topk_div)), fig

    def _get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet([self.pref_alpha]*self.obj_dim)
        
        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) 
        else: 
            prefs_enc = torch.from_numpy(prefs)
        
        cond_var = prefs_enc.view(-1).float().to(self.device)
        if bs:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        return cond_var, prefs
