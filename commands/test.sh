### test using logits

python main.py \
--gpu_id 0 \
--n_workers 32 \
--batch_size 8 \
--test_only \
--queries_eval_file 'data/dev.json' \
--test_output_loc 'res/submission_dev.csv' \
--vocab_file_loc 'bslcp_vocab_981.json' \
--test_data_loc 'track3_data/bslcp_challenge_logits/dev' \
--logits_only \

# test using trained MLP model

# python main.py \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 8 \
# --test_only \
# --resume checkpoints/checkpoints/model_0000034190.pt \
# --queries_eval_file 'data/dev.json' \
# --test_output_loc 'res/submission_dev.csv' \
# --vocab_file_loc 'bslcp_vocab_981.json' \
# --test_data_loc 'track3_data/bslcp_challenge_data/dev' \
