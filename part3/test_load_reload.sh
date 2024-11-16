#### NOTE ####
# This is just for testing!
##############

# python ~/staging.py --exe train_rnn_zhh_2_test --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env LYY

# python ~/staging.py --exe train_rnn_zhh_2_reload_test_part1 --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP_reload --gpus 4 --time 8:00:00 --exclusive --conda_env LYY

python ~/staging.py --exe train_rnn_zhh_2_reload_test_part2 --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP_reload --gpus 4 --time 8:00:00 --exclusive --conda_env LYY
