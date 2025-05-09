
device=0

LOG=${save_dir}"log.txt"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./exps/checkpoint_best/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test3.py --dataset mvtec \
        --data_path /home/ljf/datasets/mvtec_anomaly_detection --save_path ./results/${base_dir}/zero_shot \
        --checkpoint_path ${save_dir}prompt_learner_epoch_15.pth \
        --checkpoint_layer2_path ${save_dir}prompt_learner_layer2_epoch_15.pth \
        --checkpoint_layer3_path ${save_dir}prompt_learner_layer3_epoch_15.pth \
        --checkpoint_layer4_student_path ${save_dir}prompt_learner_layer4_student_epoch_14.pth \
        --checkpoint_global_path ${save_dir}prompt_learner_global_epoch_15.pth \
        --adapter_layer1_image_path ${save_dir}adapter_layer1_image_epoch_15.pth \
        --adapter_layer2_image_path ${save_dir}adapter_layer2_image_epoch_15.pth \
        --adapter_layer3_image_path ${save_dir}adapter_layer3_image_epoch_15.pth \
        --adapter_layer4_image_path ${save_dir}adapter_layer4_image_epoch_15.pth \
         --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

LOG=${save_dir}"log.txt"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the mvtec dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./exps/checkpoint_best/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test3.py --dataset visa \
        --data_path /home/ljf/datasets/VisA_20220922 --save_path ./results/${base_dir}/zero_shot \
        --checkpoint_path ${save_dir}prompt_learner_epoch_15.pth \
        --checkpoint_layer2_path ${save_dir}prompt_learner_layer2_epoch_15.pth \
        --checkpoint_layer3_path ${save_dir}prompt_learner_layer3_epoch_15.pth \
        --checkpoint_layer4_student_path ${save_dir}prompt_learner_layer4_student_epoch_15.pth \
        --checkpoint_global_path ${save_dir}prompt_learner_global_epoch_15.pth \
        --adapter_layer1_image_path ${save_dir}adapter_layer1_image_epoch_15.pth \
        --adapter_layer2_image_path ${save_dir}adapter_layer2_image_epoch_15.pth \
        --adapter_layer3_image_path ${save_dir}adapter_layer3_image_epoch_15.pth \
        --adapter_layer4_image_path ${save_dir}adapter_layer4_image_epoch_15.pth \
        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done