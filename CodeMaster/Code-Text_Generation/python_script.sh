    cd code
lang=python #programming language
lr=5e-5
batch_size=8
beam_size=10
source_length=512 
target_length=100
data_dir=../data/Code-Docstring-Corpus
output_dir=../output/
train_file=$data_dir/
epochs=10
pretrained_model=Salesforce/codet5p-770m

CUDA_VISIBLE_DEVICES=0 python run.py \
    --model_type codet5p \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --do_train \
    --do_test 
    --load_model_path ../java_model/pytorch_model.bin





