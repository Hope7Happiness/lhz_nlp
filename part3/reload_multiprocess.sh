echo !!! Doing RELOAD !!!
echo Load from $6
echo "model_type: $1"
echo "dataset_dir: ./data/rnn/cot_binary_32_1000000_right"
echo "output_dir: ./output_$1_$2_$3"
echo "model_config_path: ./configs/$2m_$1.json"
echo "batch_size: $5"
echo "total_training_samples: $3"
echo "lr: $4"
python3 rnn/train.py \
    --model_type $1 \
    --dataset_dir $HOME/lhz_nlp_data/rnn/cot_binary_32_1000000_right \
    --output_dir ./output_$1_$2_$3 \
    --model_config_path ./configs/$2m_$1.json \
    --batch_size $5 \
    --total_training_samples $3 \
    --lr $4 \
    --previous_model_path $HOME/lhz_nlp_newest_model
if [ $? -ne 0 ]; then
    echo "Training already finish!"
else
    rm $HOME/lhz_nlp_newest_model
    ln -s $(pwd)/output_$1_$2_$3 $HOME/lhz_nlp_newest_model
    echo "Created symlink to $(pwd)/output_$1_$2_$3 at $HOME/lhz_nlp_newest_model"
    ls -l $HOME/lhz_nlp_newest_model
    echo "Done!"
fi
# --dataset_dir /nobackup/users/zhh24/dev/LHZ_NLP/data/rnn/cot_binary_32_1000000_right \
