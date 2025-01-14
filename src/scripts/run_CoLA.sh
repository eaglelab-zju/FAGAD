# nohup bash scripts/run_CoLA.sh 6 >logs/CoLA_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "Facebook"; do
    for embedding_dim in 256 128 64; do
        for lr in 0.01 0.001 0.0001; do
            for drop_prob in 0.2 0.5 0.8; do
                # for alpha in 0.1 0.5 0.9; do
                    python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 CoLA --embedding_dim $embedding_dim --lr $lr --drop_prob $drop_prob --weight_decay 0.00001 --num_epoch 100 --auc_test_rounds 256 --batch_size 1024
done
    done
        done
            done
                # done
