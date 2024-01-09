GPU_ID=3

ATmethod=PGD
epoch=1
data_name=cifar10
train_or_unlearning=retrain
model_name=ResNet18

forget_class=2

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --forget_class $forget_class --epoch $epoch --ATmethod $ATmethod --model_name $model_name --train_or_unlearning $train_or_unlearning --data_name $data_name