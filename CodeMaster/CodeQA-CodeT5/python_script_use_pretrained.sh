    cd code
lang=python #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=200 
target_length=30 
data_dir=../data
output_dir=../output/
train_file=$data_dir/$lang/train/
dev_file=$data_dir/$lang/dev/
test_file=$data_dir/$lang/test/
epochs=20
pretrained_model=Salesforce/codet5p-770m

CUDA_VISIBLE_DEVICES=0 python run.py \
    --model_type codet5p \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --do_train \
    --load_model_path ../model/pytorch_model.bin





