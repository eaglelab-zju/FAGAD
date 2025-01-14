# nohup bash scripts/run_gadnr_wiki.sh 0 >logs/gadnr_wiki.log &
gpu=$1


for d in "wiki"; do
    for hid_dim in 128 64; do
        #  256 OOM
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for num_layers in 1 2 3 5; do
                    for fea_dec_layers in 2 3; do
                        python -u -m dgld.models.pygod.train.gadnr.gadnr --dataset $d --gpu $gpu --hid_dim $hid_dim --lr $lr --dropout $dropout --num_layers $num_layers --fea_dec_layers $fea_dec_layers
done
    done
        done
            done
                done
                    done
