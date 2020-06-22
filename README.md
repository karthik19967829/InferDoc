# InferDoc
This repo has code to generate synthetic question answering training data using https://github.com/facebookresearch/UnsupervisedQA and training them using https://github.com/deepset-ai/haystack

# SQUAD style QA dataset generation
cd self_supervised_qa && python -m unsupervisedqa.generate_synthetic_qa_data example_input.txt example_output

## Transformer QA Model train ,eval and CLI testing
# Usage:
    qa_model.py train --data_dir=<data_dir> --train_file_name=<train_file_name> --dev_file_name=<dev_file_name>  --save_dir=<save_dir>\
    qa_model.py test --data_dir=<data_dir> --eval_file_name=<eval_file_name> --save_dir=<save_dir>\
    qa_model.py cli --data_dir=<data_dir> --save_dir=<save_dir>

# Options:
  --data_dir=<data_dir>........A namespace to find .txt squad formatted train or eval files\
  --train_file_name=<train_file_name>..............name of the train file in the data dir\
  --dev_file_name=<dev_file_name>..............The file to be used as a development set ,expected in SQUAD json format\
  --eval_file_name=<eval_file_name>..............The file to be used as a evaluation file,expected in SQUAD json format\
  --save_dir=<save_dir> ............The directory to save the trained model or to load the model from
  
  # Todo 
  Add automatic dataset generation within https://github.com/cdqa-suite/cdQA-ui to enable human in loop semi-supervised training
  Make fine-tuning on domain specific data more robust with https://github.com/deepset-ai/FARM/issues/141
