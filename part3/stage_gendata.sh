# 0.5
# python ~/staging.py --exe train_rnn_zhh_1 --remote_url mitgpu --remote_stage_dir /nobackup/users/zhh24/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env DYY

# 1
# python ~/staging.py --exe gen --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env LYY

# 2
# python ~/staging.py --exe gen --remote_url satori_baiyu --remote_stage_dir /nobackup/users/baiyuzhu/zhh/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env wgt

# 3
# python ~/staging.py --exe gen --remote_url satori_sqa --remote_stage_dir /nobackup/users/sqa24/zhh/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env wgt

# 4
# python ~/staging.py --exe gen --remote_url satori_djq --remote_stage_dir /nobackup/users/jqdai/zhh/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env ZHH

# 5
# python ~/staging.py --exe gen --remote_url satori_ybw --remote_stage_dir /nobackup/users/bowenyu/zhh/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env ZHH

# 6
python ~/staging.py --exe gen --remote_url satori_da --remote_stage_dir /nobackup/users/dcao2028/zhh/staging --task_name LHZ_NLP_gendata --gpus 1 --time 8:00:00 --exclusive --conda_env ZHH
