# nohup bash scripts/run_GUIDE.sh 5 >logs/GUIDE_fb.log &
gpu=$1


# for d in "YelpChi" "YelpChi"; do
for d in "YelpChi"; do
    # for attrb_hid in 256 128 64 16; do
    for attrb_hid in 16; do
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for struct_hid in 16 32; do
                    # for alpha in 0.4 0.9; do
                    for alpha in 1 0.9 0.4; do
                        for num_layers in 2 4; do
                            python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 GUIDE --attrb_hid $attrb_hid --lr $lr --dropout $dropout --num_epoch 100  --alpha $alpha --struct_hid $struct_hid --num_layers $num_layers
done
    done
        done
            done
                done
                    done
                        done
