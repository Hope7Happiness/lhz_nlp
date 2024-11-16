# 0.5
# python ~/staging.py --exe train_rnn_zhh_1 --remote_url mitgpu --remote_stage_dir /nobackup/users/zhh24/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env DYY

# 1
python ~/staging.py --exe train_rnn_zhh_2 --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env LYY

# 2
# python ~/staging.py --exe train_rnn_zhh_3 --remote_url mitgpu --remote_stage_dir /nobackup/users/zhh24/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env DYY