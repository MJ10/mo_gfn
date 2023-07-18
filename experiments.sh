
algo="mogfn"
for seed in 123 42 10
do
    for sample_beta in 96
    do
        for task in regex 
        do
            sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=32 task.max_len=36 wandb_mode=online algorithm.beta_cond=False algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=50 group_name=${algo} exp_name=${algo}_${task}_${seed} exp_tags="[${task},${algo},32-36,paper_final]" seed=${seed} algorithm.state_save_path="/network/scratch/m/moksh.jain/mogfn/${algo}_${sample_beta}_${seed}_${task}.pkl.gz"
        done
    done
done

