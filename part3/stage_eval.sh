# 0.5
# python ~/staging.py --exe train_rnn_zhh_1 --remote_url mitgpu --remote_stage_dir /nobackup/users/zhh24/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env DYY

# 1
# python ~/staging.py --exe train_rnn_zhh_2 --remote_url satori_xibo --remote_stage_dir /nobackup/users/jzc_2007/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env LYY

# 2
# python ~/staging.py --exe train_rnn_zhh_3 --remote_url satori_baiyu --remote_stage_dir /nobackup/users/baiyuzhu/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env wgt

# Hybrid 0.5
# python ~/staging.py --exe train_hybrid_zhh_1 --remote_url satori_sqa --remote_stage_dir /nobackup/users/sqa24/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env wgt

# Hybrid 1
# python ~/staging.py --exe train_hybrid_zhh_2 --remote_url satori_djq --remote_stage_dir /nobackup/users/jqdai/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env ZHH

# Hybrid 2
# python ~/staging.py --exe train_hybrid_zhh_3 --remote_url satori_ybw --remote_stage_dir /nobackup/users/bowenyu/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env ZHH

# Transformer sanity
python ~/staging.py --exe eval_transformer_sanity --remote_url satori_da --remote_stage_dir /nobackup/users/dcao2028/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env ZHH
