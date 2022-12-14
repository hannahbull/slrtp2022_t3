### train sign classification MLP model on features

python main.py \
--gpu_id 0 \
--n_workers 32 \
--batch_size 64 \
--n_epochs 5 \
--train_data_loc 'track3_data/bslcp_challenge_data/train' \
--train_labels_loc 'track3_data/bslcp_challenge_labels/train' \
--val_data_loc 'track3_data/bslcp_challenge_data/dev' \
--val_labels_loc 'track3_data/bslcp_challenge_labels/dev' \
--vocab_file_loc 'bslcp_vocab_981.json' \