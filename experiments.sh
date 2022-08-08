for algo in mogfn mogfn_fm
do
    for task in regex_2
    do
        for simplex_bins in 20
        do
            for beta_cond in False
            do
                for sample_beta in 16 48
                do
                    if test ${sample_beta} -eq 16 
                    then
                        for beta_max in ${sample_beta}
                        do
                            echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
                            sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=20000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${beta_max} algorithm.simplex_bins=${simplex_bins} group_name=${algo} exp_name=${algo}_${beta_cond}_${simplex_bins}_${sample_beta}_${beta_max} exp_tags="[${task},${algo},16-18]"
                            sleep 5
                        done
                    fi

                    if test ${sample_beta} -eq 48
                    then
                        for beta_max in ${sample_beta}
                        do
                            echo "${algo} ${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
                            sbatch job.sh algorithm=${algo} task=${task} tokenizer=protein algorithm.train_steps=20000 task.min_len=12 task.max_len=16 wandb_mode=online algorithm.beta_cond=${beta_cond} algorithm.sample_beta=${sample_beta} algorithm.beta_max=${beta_max} algorithm.simplex_bins=${simplex_bins} group_name=${algo} exp_name=${algo}_${beta_cond}_${simplex_bins}_${sample_beta}_${beta_max} exp_tags="[${task},${algo},16-18]"
                            sleep 5
                        done
                    fi
                    
                done
            done  
        done
    done
done