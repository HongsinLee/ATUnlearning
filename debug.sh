GPU_ID=0

epoch=10
data_name=cifar10
train_or_load=train
model_name=ResNet18

forget_class=4

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --forget_class $forget_class --epoch $epoch --model_name $model_name --train_or_load $train_or_load --data_name $data_name