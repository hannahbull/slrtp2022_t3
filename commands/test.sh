python main.py \
--gpu_id 0 \
--n_workers 32 \
--batch_size 8 \
--test_only \
--resume checkpoints/checkpoints/model_0000034190.pt \
--queries_eval_file 'data/dev.json' \
--test_data_loc 'bslcp_challenge_data_bobsl/dev' \
--test_output_loc 'res/submission_dev.csv' \
--vocab_file_loc 'bslcp_vocab_981.json' \
# --logits_only \