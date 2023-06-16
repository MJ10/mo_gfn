# MOGFN for Sequence Tasks

Code for sequence-based tasks with Multi-Objective GFlowNets. The code is inspired heavily by the LaMBO code: [https://github.com/samuelstanton/lambo](https://github.com/samuelstanton/lambo). 

This folder consists of code for 2 tasks:
* N-grams (Objectives are the occurences of given n-grams)
* DNA Aptamers (Objectives are properties from [NUPACK](https://docs.nupack.org/))


## Installation
The code is encapsulated in the `torch_seq_moo` library. To install the library along with the dependencies follow the instructions below. Tested with Python 3.9 and CUDA 11.7.

```bash
virtualenv env
source env/bin/activate

pip install -e .
```

**Note**: If you would like to run the experiments on DNA Aptamers, please [NUPACK](https://docs.nupack.org/). You need to register and download the wheel from the website.

If you are on a SLURM cluster - please refer to `job.sh` for an example script.

## Commands

For N-grams
```bash
python main.py algorithm=mogfn task=regex tokenizer=protein algorithm.train_steps=10000 task.min_len=32 task.max_len=36 algorithm.beta_cond=False algorithm.sample_beta=96 algorithm.beta_max=96 algorithm.simplex_bins=50 seed=1 algorithm.state_save_path="<some_path_here>" 
```


For DNA Aptamers
```bash
python main.py algorithm=mogfn task=nupack_energy_pins_pairs tokenizer=aptamer algorithm.train_steps=10000 task.min_len=30 task.max_len=60 algorithm.beta_cond=False algorithm.sample_beta=80 algorithm.beta_max=42 algorithm.simplex_bins=50 seed=1 algorithm.state_save_path="<some_path_here>"
```

