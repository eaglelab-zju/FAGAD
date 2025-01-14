# nohup bash scripts/run_gadnr_enr.sh 0 >logs/gadnr_enr.log &
gpu=$1


for d in "Enron"; do
    for hid_dim in 16  128 256; do
        for lr in 0.01 0.001 0.0001; do
            for lambda_loss1 in 0.01 0.1 0.8; do
                for lambda_loss2 in 0.01 0.1 0.8; do
                    for lambda_loss3 in 0.01 0.1 0.8; do
                        python -u -m dgld.models.GADNR.gadnr --dataset $d --gpu $gpu --hid_dim $hid_dim --lr $lr --lambda_loss1 $lambda_loss1 --lambda_loss2 $lambda_loss2 --lambda_loss3 $lambda_loss3 --normalize_feat True
done
    done
        done
            done
                done
                    done
