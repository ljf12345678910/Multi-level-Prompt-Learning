
device=1


LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./checkpoint_best/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train3.py --dataset visa --train_data_path /home/ljf/datasets/VisA_20220922 \
        --dataset2 mvtec --test_data_path /home/ljf/datasets/mvtec_anomaly_detection \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --seed 888 \
        --learning_rate 0.001 --learning_rate2 0.001 --distil 1.0
    wait
    done
done

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoint_best/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train3.py --dataset mvtec --train_data_path /home/ljf/datasets/mvtec_anomaly_detection \
        --dataset2 visa --test_data_path /home/ljf/datasets/VisA_20220922 \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --seed 888 \
        --learning_rate 0.005 --learning_rate2 0.001 --distil 0.9
    wait
    done
done
# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
#         save_dir=./checkpoint_linear01adapter4_category_global_conv_lr_end7_1_new/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python train3.py --dataset visa --train_data_path /home/ljf/datasets/VisA_20220922 \
#         --save_path ${save_dir} \
#         --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
#         --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --seed 888 \
#         --learning_rate 0.001 --learning_rate2 0.001 --distil 1.0
#     wait
#     done
# done

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoint_linear01adapter4_category_global_conv_lr_end7_1_news/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python train3.py --dataset mvtec --train_data_path /home/ljf/datasets/mvtec_anomaly_detection \
#         --save_path ${save_dir} \
#         --features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
#         --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --seed 888 \
#         --learning_rate 0.001 --learning_rate2 0.0001 --distil 0.9
#     wait
#     done
# done
