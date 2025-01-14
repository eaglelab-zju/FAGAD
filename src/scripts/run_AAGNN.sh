# nohup bash scripts/run_AAGNN.sh 6 >logs/AAGNN_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "Facebook"; do
    for out_feats in 256 128 64; do
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                # for eta in 0.1 0.5 0.9; do
                #     for theta in 0.1 0.5 0.9; do
                        python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 AAGNN --out_feats $out_feats --out_dim $out_feats --lr $lr --dropout $dropout --weight_decay 0.00001 --num_epoch 100
done
    done
        done
            done
                # done
                #     done
