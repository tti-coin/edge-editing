#!/bin/sh
echo "Split corpus to train, dev test following original paper of corpus" && \
python src/split_data.py --output preprocess/data_split --train_files annotated-materials-syntheses/ner-train-fnames.txt --dev_files annotated-materials-syntheses/ner-dev-fnames.txt --test_files annotated-materials-syntheses/ner-test-fnames.txt && \
echo "Apply rule-based model to the corpus" && \
for m in train dev test; do echo $m; python src/extract_rule.py --input_dir preprocess/data_split/$m --output_dir preprocess/rule/$m; done && \
echo "Convert event styled data to relation style" && \
for m in train dev test; do echo $m; python src/convert_event.py --input_dir preprocess/rule/$m --output_dir preprocess/rule_rel/$m; done && \
for m in train dev test; do echo $m; python src/convert_event.py --input_dir preprocess/data_split/$m --output_dir preprocess/data_rel/$m; done && \
echo "Preprocess for neural edge editing" && \ 
python src/preprocess.py --pkl_path preprocess/pkl --gold_prefix preprocess/data_rel --pred_prefix preprocess/rule_rel --config olivetti.conf
