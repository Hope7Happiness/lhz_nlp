python rnn/data.py \
    --n_nodes 32 \
    --graph_type binary \
    --task_type cot \
    --size 1000000
ln -s $(pwd)/data $HOME/lhz_nlp_data