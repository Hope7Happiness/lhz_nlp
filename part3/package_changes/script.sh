export REMOTE_USER=satori_da
export REMOTE_NAME=dcao2028
export REMOTE_ENV=ZHH

# rsync -av ./modeling_llama.py satori_djq:/nobackup/users/jqdai/anaconda3/envs/ZHH/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py
# rsync -av ./modeling_llama.py satori_sqa:/nobackup/users/sqa24/anaconda3/envs/wgt/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py
# rsync -av ./modeling_llama.py satori_ybw:/nobackup/users/bowenyu/anaconda3/envs/ZHH/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py
# rsync -av ./modeling_llama.py satori_da:/nobackup/users/dcao2028/anaconda3/envs/ZHH/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py

rsync -av ./modeling_llama.py $REMOTE_USER:/nobackup/users/$REMOTE_NAME/anaconda3/envs/$REMOTE_ENV/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py
rsync -av ./util.py $REMOTE_USER:/nobackup/users/$REMOTE_NAME/anaconda3/envs/$REMOTE_ENV/lib/python3.7/site-packages/transformers/generation/utils.py