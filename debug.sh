GPU_ID=6

ATmethod=PGD
epoch=200
data_name=cifar10
train_or_unlearning=retrain
model_name=ResNet18
unlearn_innerloss=FGSM
unlearn_ATmethod=PGD

forget_class=0

for forget_class in 8 9 
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --forget_class $forget_class --epoch $epoch --ATmethod $ATmethod --model_name $model_name --train_or_unlearning $train_or_unlearning --data_name $data_name --unlearn_innerloss $unlearn_innerloss --unlearn_ATmethod $unlearn_ATmethod
done