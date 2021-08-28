for target in ESR1_ant KAT2A
do
for job_type in train_lit train_1000 train_dude train_top50 train_300
do
path/my-envs/dgl-env/bin/python -u /path/model_training/IGN/train_ign.py --target ${target} --job_type ${job_type} >> /apdcephfs/private_xujunzhang/project_5/IGN/train.log 2>&1
path/my-envs/dgl-env/bin/python -u /path/model_training/IGN/test_ign.py --target ${target} --job_type ${job_type} >> /apdcephfs/private_xujunzhang/project_5/IGN/test.log 2>&1
done
echo ${target}
echo end
done

