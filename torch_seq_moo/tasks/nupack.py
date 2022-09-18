import re

import numpy as np

from torch_seq_moo.tasks.base import BaseTask
from torch_seq_moo.utils import random_strings
from nupack import Strand, Complex, ComplexSet, Model, SetSpec, complex_analysis
from omegaconf import ListConfig


class NupackTask(BaseTask):
    def __init__(
        self,
        regex_list,
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
        self.regex_list = None
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs["max_score_per_dim"]
        self.score_max = kwargs["score_max"]
        self.objectives = objectives

    def task_setup(self, *args, **kwargs):
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
        scores_dict = self.nupack_score(candidates, objectives=self.objectives)
        scores = [scores_dict[obj] for obj in self.objectives]
        scores = np.stack(scores, axis=-1).astype(np.float64)
        # Normalize and make positive
        scores = -1 * scores / self.score_max
        return scores


    def nupack_score(self, sequences, objectives="energy"):
        """
        Computes the score (energy, number of pins, number of pairs) of the (important)
        most probable structure according the nupack. Nupack requires Linux OS. Nupack is
        preferred over seqfold - more stable and higher quality predictions.

        Args
        ----
        candidates : list
            List of sequences.
            TODO: specify format when decided.

        objectives : string or list of strings
            Nupack objective(s) to return. Multiple objectives will be returned if a list
            of strings is used as argument.
        """
        temperature = 310.0  # Kelvin
        ionic_strength = 1.0  # molar

        energies = np.zeros(len(sequences))
        n_pins = np.zeros(len(sequences)).astype(int)
        n_pairs = 0
        ssStrings = np.zeros(len(sequences), dtype=object)

        # parallel evaluation - fast
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name="strand{}".format(i)))
            comps.append(Complex([strandList[-1]], name="comp{}".format(i)))

        set = ComplexSet(strands=strandList, complexes=SetSpec(max_size=1, include=comps))
        model1 = Model(material="dna", celsius=temperature - 273, sodium=ionic_strength)
        results = complex_analysis(set, model=model1, compute=["mfe"])
        for i in range(len(energies)):
            energies[i] = results[comps[i]].mfe[0].energy
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)

        dict_return = {}
        if "pins" in objectives:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        indA += 1
                    elif ssStrings[i][j] == ")":
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            n_pins[i] += 1
            dict_return.update({"pins": -n_pins})
        if "pairs" in objectives:
            n_pairs = np.asarray([ssString.count("(") for ssString in ssStrings]).astype(int)
            dict_return.update({"pairs": -n_pairs})
        if "energy" in objectives:
            dict_return.update(
                {"energy": energies}
            )  # this is already negative by construction in nupack
        if "invlength" in objectives:
            invlength = np.asarray([self.min_len * (1.0 / len(seq)) for seq in sequences])
            dict_return.update({"invlength": -invlength})
        if "explength" in objectives:
            beta_length = 0.05
            explength = np.asarray([np.exp(-1.0 * beta_length * len(seq)) for seq in sequences])
            dict_return.update({"explength": explength})

        if "open loop" in objectives:
            biggest_loop = np.zeros(len(ssStrings))
            for i in range(
                len(ssStrings)
            ):  # measure all the open loops and return the largest
                loops = [0]  # size of loops
                counting = 0
                indA = 0
                # loop completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        counting = 1
                        indA = 0
                    if (ssStrings[i][j] == ".") and (counting == 1):
                        indA += 1
                    if (ssStrings[i][j] == ")") and (counting == 1):
                        loops.append(indA)
                        counting = 0
                biggest_loop[i] = max(loops)
            dict_return.update({"open loop": -biggest_loop})

        if isinstance(objectives, (list, ListConfig)):
            if len(objectives) > 1:
                return dict_return
            else:
                return dict_return[objectives[0]]
        else:
            return dict_return[objectives]
