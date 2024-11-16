train_transformer:
	bash train.sh transformer 1 5000000 0.0001 64
test_transformer:
	bash eval.sh transformer 0.5 model_best.pt 5000000
train_and_test_transformer: train_transformer test_transformer
	
train_rnn:
	bash train.sh rnn 0.5 1000000 0.0003 64
test_rnn:
	bash eval.sh rnn 0.5 model_best.pt 5000000
train_and_test_rnn: train_rnn test_rnn

train_hybrid:
	bash train.sh hybrid 0.5 5000000 0.0001 64
test_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt 5000000

gen: 
	bash gen.sh
train_all_rnn:
	bash train.sh rnn 0.5 5000000 0.0003 64
	bash train.sh rnn 1 5000000 0.0003 64
	bash train.sh rnn 2 5000000 0.0003 64
eval_all_rnn:
	bash eval.sh rnn 0.5 model_best.pt 5000000
	bash eval.sh rnn 1 model_best.pt 5000000
	bash eval.sh rnn 2 model_best.pt 5000000
train_all_hybrid:
	bash train.sh hybrid 0.5 5000000 0.0001 64
	bash train.sh hybrid 1 5000000 0.0001 64
	bash train.sh hybrid 2 5000000 0.0001 64
eval_all_hybrid:
	bash eval.sh hybrid 0.5 model_best.pt 5000000
	bash eval.sh hybrid 1 model_best.pt 5000000
	bash eval.sh hybrid 2 model_best.pt 5000000
train_all_transformer_1:
	bash train.sh transformer 0.125 5000000 0.0001 64
	bash train.sh transformer 1 5000000 0.0001 64
	bash train.sh transformer 2 5000000 0.0001 64
	bash train.sh transformer 3 5000000 0.0001 64
eval_all_transformer_1:
	bash eval.sh transformer 0.125 model_best.pt 5000000
	bash eval.sh transformer 1 model_best.pt 5000000
	bash eval.sh transformer 2 model_best.pt 5000000
	bash eval.sh transformer 3 model_best.pt 5000000
train_all_transformer_2:
	bash train.sh transformer 5 5000000 0.0001 64
	bash train.sh transformer 7 5000000 0.0001 64
	bash train.sh transformer 9 5000000 0.0001 64
	bash train.sh transformer 0.25 5000000 0.0001 64
eval_all_transformer_2:
	bash eval.sh transformer 5 model_best.pt 5000000
	bash eval.sh transformer 7 model_best.pt 5000000
	bash eval.sh transformer 9 model_best.pt 5000000
	bash eval.sh transformer 0.25 model_best.pt 5000000

train_rnn_zhh_1:
	bash train.sh rnn 0.5 5000000 0.0003 2048

train_rnn_zhh_2:
	bash train.sh rnn 1 5000000 0.0003 1024

train_rnn_zhh_3:
	bash train.sh rnn 2 5000000 0.0003 512

train_rnn_zhh_2_test:
	bash train.sh rnn 1 5000 0.0003 1024

train_rnn_zhh_2_reload_test:
	bash reload.sh rnn 1 10000 0.0003 1024 /nobackup/users/jzc_2007/staging/20241115__230706_LHZ_NLP_dde4ce2-dirty/output_rnn_1_5000

train_hybrid_zhh_1:
	bash train.sh hybrid 0.5 5000000 0.0001 1024

train_hybrid_zhh_2:
	bash train.sh hybrid 1 5000000 0.0001 512

train_hybrid_zhh_3:
	bash train.sh hybrid 2 5000000 0.0001 256

train_transformer_sanity:
# This works
	bash train.sh transformer 0.5 5000000 0.001 256

# Can this work? TODO
#	bash train.sh transformer 0.5 5000000 0.002 512

eval_transformer_sanity:
	bash eval.sh transformer 0.5 /nobackup/users/dcao2028/zhh/staging/20241116__004716_LHZ_NLP_d7f2627-dirty/output_transformer_0.5_5000000/model_4884.pt 5000000