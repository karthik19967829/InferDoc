"""
Usage:
    qa_model.py train --data_dir=<data_dir> --train_file_name=<train_file_name> --dev_file_name=<dev_file_name>  --save_dir=<save_dir>
    qa_model.py test --data_dir=<data_dir> --eval_file_name=<eval_file_name> --save_dir=<save_dir>
    qa_model.py cli --data_dir=<data_dir> --save_dir=<save_dir>

Options:
  --data_dir=<data_dir>........A namespace to find .txt squad formatted train or eval files
  --train_file_name=<train_file_name>..............name of the train file in the data dir
  --dev_file_name=<dev_file_name>..............The file to be used as a development set ,expected in SQUAD json format
  --eval_file_name=<eval_file_name>..............The file to be used as a evaluation file,expected in SQUAD json format
  --save_dir=<save_dir> ............The directory to save the trained model or to load the model from
"""


from haystack.reader.farm import FARMReader
from haystack.database.base import Document
import os
from docopt import docopt
import glob

def main():
    args        = docopt(__doc__)
    data_dir     = args["--data_dir"]
    if args["train"]    : 
        reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)
        reader.train(data_dir=data_dir, train_filename=args["--train_file_name"],dev_filename=args["--dev_file_name"],use_gpu=False, n_epochs=1, save_dir=args["--save_dir"],dev_split=0.05)
    if args["test"]     : 
        reader = FARMReader(model_name_or_path=args["--save_dir"], use_gpu=False)
        print(reader.eval_on_file(data_dir,args["--eval_file_name"],'cpu'))
    if args["cli"]      :
        reader = FARMReader(model_name_or_path=args["--save_dir"], use_gpu=False)
        query_doc_list=[]
        for text_file in list(glob.glob(data_dir+'/*.txt')):
            with open(text_file,"r") as f:
                context=f.read()
            #context=context.split(".")
            context=[context]
            for i,para in enumerate(context):    
                query_doc_list.append(Document(id=str(i),text=para))
        while 1:  
            question=input("CTRL C to exit >")
            prediction=reader.predict(question,query_doc_list)
            print("answer:>> ",prediction['answers'][0]['answer']) 
            print("-----")
            print("context:>> ",prediction['answers'][0]['context'])
            print("-------------")             

if __name__=='__main__':
    main()

