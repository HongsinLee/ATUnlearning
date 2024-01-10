GPU_ID=1

method=AT_BS
ATmethod=PGD
epoch=200
data_name=cifar10
train_or_unlearning=unlearning
model_name=ResNet18
unlearn_innerloss=PGD
unlearn_ATmethod=Nat

forget_class=0

for forget_class in 0
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --forget_class $forget_class --epoch $epoch --ATmethod $ATmethod --model_name $model_name --train_or_unlearning $train_or_unlearning --data_name $data_name --unlearn_innerloss $unlearn_innerloss --unlearn_ATmethod $unlearn_ATmethod --method $method
done