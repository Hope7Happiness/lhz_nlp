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

# RNN 0.5
stage_common_func train_rnn_zhh_safe_1 mitgpu /nobackup/users/zhh24/staging DYY 10
# stage_common_func train_rnn_zhh_safe_1 mitgpu /nobackup/users/zhh24/staging DYY 1

# RNN 1
stage_common_func train_rnn_zhh_safe_2 satori_xibo /nobackup/users/jzc_2007/staging LYY 10
# stage_common_func train_rnn_zhh_safe_2 satori_xibo /nobackup/users/jzc_2007/staging LYY 1

# RNN 2
stage_common_func train_rnn_zhh_safe_3 satori_baiyu /nobackup/users/baiyuzhu/zhh/staging wgt 10
# stage_common_func train_rnn_zhh_safe_3 satori_baiyu /nobackup/users/baiyuzhu/zhh/staging wgt 1

# Hybrid 0.5
# stage_common_func TODO satori_sqa /nobackup/users/sqa24/zhh/staging wgt 10
# stage_common_func TODO satori_sqa /nobackup/users/sqa24/zhh/staging wgt 1

# Hybrid 1
# stage_common_func TODO satori_djq /nobackup/users/jqdai/zhh/staging ZHH 10
# stage_common_func TODO satori_djq /nobackup/users/jqdai/zhh/staging ZHH 1

# Hybrid 2
# stage_common_func TODO satori_ybw /nobackup/users/bowenyu/zhh/staging ZHH 10
# stage_common_func TODO satori_ybw /nobackup/users/bowenyu/zhh/staging ZHH 1

# Transformer sanity
# stage_common_func TODO satori_da /nobackup/users/dcao2028/zhh/staging ZHH 10
# stage_common_func TODO satori_da /nobackup/users/dcao2028/zhh/staging ZHH 1
