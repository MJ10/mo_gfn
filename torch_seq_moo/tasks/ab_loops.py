import re

import numpy as np

from torch_seq_moo.tasks.base import BaseTask
import torch

MHC_TARGETS = [
    "YAFFMFSGGAILNTLFGQFEYFDIEEVRMHLGMT"
]

class AbLoopTask(BaseTask):
    def __init__(
        self,
        max_len,
        min_len,
        num_start_examples,
        tokenizer,
        objectives,
        transform=lambda x: x,
        **kwargs
    ):
        obj_dim = len(objectives)
        super().__init__(tokenizer, obj_dim, max_len, transform, **kwargs)
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples
        self.score_max = kwargs["score_max"]
        self.objectives = objectives
        self.devel_config_path = kwargs["devel_config_path"]
        self.device = kwargs["device"]
        self.fixed_backbone = kwargs.get("fixed_backbone", None)

    def task_setup(self, *args, **kwargs):
        self.toks = {}
        self.models = {}

        for obj in self.objectives:
            if obj == "mhc2":
                try:
                    import mhc2oracle
                except:
                    raise ImportError(
                        "mhc2oracle not installed."
                    )
                tokenizer, model = mhc2oracle.load_pretrained_model()
                model = model.to(self.device)
                self.toks[obj] = tokenizer
                self.models[obj] = model
            elif obj == "developability":
                try:
                    import devoracle
                except:
                    raise ImportError(
                        "developability oracle not installed."
                    )
                device, config, model, tokenizer = devoracle.load_pretrained_model(self.devel_config_path)
                model = model.to(self.device)
                self.toks[obj] = tokenizer
                self.models[obj] = model
            elif obj == "naturalness":
                try:
                    import lmoracle
                except:
                    raise ImportError(
                        "lmoracle not installed."
                    )
                tokenizer, model = lmoracle.load_pretrained_model()
                model = model.to(self.device)
                self.toks[obj] = tokenizer
                self.models[obj] = model
        return [], []

    def score(self, candidates):
        """
        Computes multi-objective scores for each object in candidates.

        Args
        ----
        candidates : list or np.array
            Aptamer sequences in letter format.

        Returns
        -------
        scores : np.array
            Multi-objective scores. Shape: [n_candidates, n_objectives]
        """
        scores = []
        for obj in self.objectives:
            scores.append(getattr(self, f"_score_{obj}")(candidates))
        # scores_dict = self._score(candidates, objectives=self.objectives)
        # scores = [scores_dict[obj] for obj in self.objectives]
        scores = np.stack(scores, axis=-1).astype(np.float64)
        # Normalize and make positive
        scores = scores / self.score_max
        return scores


    def _score_mhc2(self, candidates):
        # scores = []
        # for tokenizer, model in zip(self.toks, self.models):
        tokenizer = self.toks["mhc2"]
        model = self.models["mhc2"]
        input_ids = tokenizer(candidates.tolist(), return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        
        target_input_ids = tokenizer(MHC_TARGETS, return_tensors="pt", padding=True).input_ids.repeat(len(candidates), 1)
        target_input_ids = target_input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids, target_input_ids)
        scores = torch.exp(-outputs).squeeze().cpu().numpy()
        return scores

    def _score_developability(self, candidates):
        tokenizer = self.toks["developability"]
        model = self.models["developability"]
        if self.fixed_backbone is not None:
            candidates = [self.fixed_backbone[0] + c + self.fixed_backbone[1] for c in candidates]
        input_ids = tokenizer(candidates, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = model(input_ids)
        # import pdb; pdb.set_trace();
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        scores = probs[:, 1].cpu().numpy()
        return scores

    def _score_naturalness(self, candidates, batch_size=32):
        tokenizer = self.toks["naturalness"]
        model = self.models["naturalness"]
        if self.fixed_backbone is not None:
            candidates = [self.fixed_backbone[0] + c + self.fixed_backbone[1] for c in candidates]
        input_ids = tokenizer(candidates, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        scores_all = []
        for i in range(0, len(input_ids), batch_size):
            with torch.no_grad():
                outputs = model(input_ids[i:i+batch_size])
            # do min-max normalization
            scores = (outputs[0].cpu().numpy() + 60) / 30
            scores = scores.clip(0, 1)
            scores_all.append(scores)
        scores = np.concatenate(scores_all)
        # scores = np.exp(outputs[0].cpu().numpy() / 100)
        return scores