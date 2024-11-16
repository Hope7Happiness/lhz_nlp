stage_common_func(){
    TASKNAME=$1
    REMOTE=$2
    STAGEDIR=$3
    CONDAENV=$4
    REPEATS=$5
    # submit 10 same jobs
    for i in $(seq 1 $REPEATS); do
        python ~/staging.py --exe $TASKNAME --remote_url $REMOTE --remote_stage_dir $STAGEDIR --task_name LHZ_NLP_reload --gpus 1 --time 8:00:00 --exclusive --conda_env $CONDAENV
    done
}


# 0.5
# stage_common_func train_rnn_zhh_1 mitgpu /nobackup/users/zhh24/staging DYY 1

# 1
# stage_common_func train_rnn_zhh_2 satori_xibo /nobackup/users/jzc_2007/staging LYY 1

# 2
# stage_common_func train_rnn_zhh_3 satori_baiyu /nobackup/users/baiyuzhu/zhh/staging wgt 1

# Hybrid 0.5
# python ~/staging.py --exe train_hybrid_zhh_1 --remote_url satori_sqa --remote_stage_dir /nobackup/users/sqa24/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env wgt
# stage_common_func train_hybrid_zhh_1 satori_sqa /nobackup/users/sqa24/zhh/staging wgt 3

# Hybrid 1
# python ~/staging.py --exe train_hybrid_zhh_2 --remote_url satori_djq --remote_stage_dir /nobackup/users/jqdai/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env ZHH
# stage_common_func train_hybrid_zhh_2 satori_djq /nobackup/users/jqdai/zhh/staging ZHH 3

# Hybrid 2
# python ~/staging.py --exe train_hybrid_zhh_3 --remote_url satori_ybw --remote_stage_dir /nobackup/users/bowenyu/zhh/staging --task_name LHZ_NLP --gpus 4 --time 8:00:00 --exclusive --conda_env ZHH
# stage_common_func train_hybrid_zhh_3 satori_ybw /nobackup/users/bowenyu/zhh/staging ZHH 3

# Transformer sanity
stage_common_func train_transformer_sanity satori_da /nobackup/users/dcao2028/zhh/staging ZHH 1