# for algo in mogfn
# do
#     for task in regex regex_2 regex_easy regex_easy_3 regex_easy_4
#     do
#         for simplex_bins in 50
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 16 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.use_eval_pref=True algorithm.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="s_mo_comparison" exp_name=comp_${algo}prefs2_${beta_cond}_${simplex_bins}_${sample_beta} exp_tags="[${task},${algo},16-18]"
#                     sleep 5
                    
#                 done
#             done  
#         done
#     done
# done


# for algo in mogfn
# do
#     for task in regex regex_2 regex_easy regex_easy_3 regex_easy_4
#     do
#         for simplex_bins in 50
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 16 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="s_mo_comparison" exp_name=comp_${algo}skewpref_${beta_cond}_${simplex_bins}_${sample_beta} exp_tags="[${task},${algo},32-36]"
#                     sleep 5
                    
#                 done
#             done  
#         done
#     done
# done


# for algo in mogfn
# do
#     for task in regex
#     do
#         for simplex_bins in 50
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 16 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=20000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="sb_test" exp_name=comp_${algo}sbtest_${beta_cond}_${simplex_bins}_${sample_beta} exp_tags="[${task},${algo},16-18]"
#                     sleep 5
                    
#                 done
#             done  
#         done
#     done
# done

# for algo in gfn
# do
#     for task in regex regex_2 regex_easy regex_easy_3 regex_easy_4
#     do
#         for simplex_bins in 10
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 16 48
#                 do
#                     for eval_pref in 0 1 2 3 4
#                     do
#                         echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max} ${eval_pref}"
#                         sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=5000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="comparison" exp_name=comp_${algo}_${beta_cond}_${simplex_bins}_${sample_beta}_${eval_pref} exp_tags="[${task},${algo},16-18]" algorithm.eval_pref_choice=${eval_pref}
#                         sleep 5
#                     done                    
#                 done
#             done  
#         done
#     done
# done

# for algo in mogfn
# do
#     for task in regex regex_2 regex_easy regex_easy_3 regex_easy_4
#     do
#         for simplex_bins in 10
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 16 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.use_eval_pref=True algorithm.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="comparison" exp_name=comp_${algo}_${beta_cond}_${simplex_bins}_${sample_beta} exp_tags="[${task},${algo},12-16]"
#                     sleep 5
                    
#                 done
#             done  
#         done
#     done
# done

# for algo in mogfn
# do
#     for task in regex regex_2 regex_4 regex_easy regex_easy_3 regex_easy_4
#     do
#         for simplex_bins in 50
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=22 task.max_len=24 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="mogfn_2" exp_name=comp_${algo}_${beta_cond}_${simplex_bins}_${sample_beta} exp_tags="[${task},${algo},22-24,baselines]"
#                     sleep 1
#                 done
#             done  
#         done
#     done
# done




# for algo in mogfn
# do
#     for task in regex regex_easy_3
#     do
#         for simplex_bins in 50
#         do
#             for beta_cond in False
#             do
#                 for sample_beta in 48
#                 do
#                     echo "${algo} ${task}, ${simplex_bins}, ${sample_beta}, ${beta_max}"
#                     sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=22 task.max_len=24 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=${simplex_bins} group_name="mogfn_2" exp_name=comp_${algo}_${simplex_bins}_${sample_beta}_{} exp_tags="[${task},${algo},22-24,reward_abl]"
#                     sleep 1
#                 done
#             done  
#         done
#     done
# done


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

# algo="mogfn_fm"
# for seed in 123 42 10
# do
#     for sample_beta in 12 32 48
#     do
#         for task in regex_2 regex_easy regex_4 regex_easy_4
#         do
#             sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=32 task.max_len=36 wandb_mode=online algorithm.beta_cond=False algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=50 group_name=${algo} exp_name=${algo}_${task}_${seed} exp_tags="[${task},${algo},32-36,paper_final]" seed=${seed}
#         done
#     done
# done


# algo="moreinforce"
# for seed in 123 42 10
# do
#     for sample_beta in 1
#     do
#         for task in regex_2 regex_easy regex_4 regex_easy_4
#         do
#             sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=32 task.max_len=36 wandb_mode=online algorithm.beta_cond=False algorithm.sample_beta=${sample_beta} algorithm.beta_max=${sample_beta} algorithm.simplex_bins=50 group_name=${algo} exp_name=${algo}_${task}_${seed} exp_tags="[${task},${algo},32-36,paper_final]" seed=${seed}
#         done
#     done
# done

# algo="envelope_moq"
# for seed in 123 42 10
# do
#     for sample_beta in 1
#     do
#         for task in regex_2 regex_easy regex_4 regex_easy_4
#         do
#             sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=10000 task.min_len=32 task.max_len=36 wandb_mode=online algorithm.simplex_bins=50 group_name=${algo} exp_name=${algo}_${task}_${seed} exp_tags="[${task},${algo},32-36,paper_final]" seed=${seed}
#         done
#     done
# done
