'''
    This script contains the code for fine-tuning Phi-4 for error detection (using the prompt design adapted from Soda-Eval and following the best practices described on the Huggingface model page).
'''
import mlflow
import torch
import json
import random as rd

import error_definitions as ed

from torch.utils.data import DataLoader
from datasets import Dataset as HfDataset
from argparse import Namespace
from tqdm import tqdm

from arguments import Arguments
from utility import accuracy_metrics
from dataset import LLMDataset
from abs_manager import Manager, Model


class LLMFinetuning(Manager):

    def __init__(self, args:Namespace):
        super(LLMFinetuning, self).__init__(args, 
            Model.LLM_FINETUNING)
        """
        This script implements the finetuning LLM baseline using the approach proposed by Mendonça et al. in "SODA-EVAL: Open-Domain Dialogue Evaluation in the age of LLMs". In their work, they used Phi-3-instruct-mini. Since today Phi-4 is available, we will use Phi-4-instuct-mini
        """        
        if self._args.error_types == "abceval":
            self._template = ed.ABCEval_LLM_TEMPLATE_PHI            
        elif self._args.error_types == "fedi":
            self._template = ed.FEDI_LLM_TEMPLATE_PHI            
        elif self._args.error_types == "soda_eval":
            self._template = ed.SODA_EVAL_LLM_TEMPLATE_PHI            
        else:
            raise Exception(f"Unknown type {self._args.error_types}")
        
    def _test(self, dataset:HfDataset, iteration:int=0, 
        n_labels:list[int]=None):

        self._model._model = self._model._model.eval()

        dataset = LLMDataset(dataset, self._args, self._template, 
            True, n_labels=[100], mode="test")          
        dataloader = DataLoader(dataset, batch_size=self._args.batch_size, drop_last=True)

        error_types = list(set([s["error_type"] for s in dataset]))

        preds, targets = [], []
        for batch in tqdm((dataloader)):            
    
            with torch.no_grad():
                print(f"{'-'*100}")
                print(f"targets: {[b.replace("_", " ") for b in batch["error_type_str"]]}\n")

                decisions, reasonings = self._model.tokenize_and_generate(batch)

                decisions = [d.replace("_", " ") for d in decisions]

                print(f"predictions: {decisions}\n")
                print(f"reasonings:\n")
                [print(f"{r}\n") for r in reasonings]

                for d, t1, t2 in zip(decisions, batch["error_type_str"], 
                    batch["error_type"]):
                    
                    preds.append(t2.item() if d.strip() == 
                        t1.replace("_", " ").strip() else rd.choice([e for e in error_types if e != t2.item()]))
                    targets.append(t2.item())                
        
        h_mean, acc_k, acc_n, class_wise_acc = accuracy_metrics(preds, 
                targets, n_labels=n_labels)
                
        metrics = {"h-mean": h_mean, "acc_k": acc_k, "acc_n": acc_n, 
            "class_wise": class_wise_acc}
        print(json.dumps(metrics))
        
        return h_mean if len(n_labels) > 0 else acc_k
           

    def _train(self, epoch:int, dataloader:DataLoader):
        pbar = tqdm(range(len(dataloader)))

        for i, batch in enumerate(dataloader):
            
            loss = self._model.tokenize_and_forward(batch)

            loss.backward()
            self._optimizer.step()            
            self._scheduler.step()
            self._optimizer.zero_grad()

            mlflow.log_metrics({"loss": loss.item()}, 
                i + len(dataloader) * epoch) 
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
        

    def run(self):

        with mlflow.start_run(run_name=self._args.experiment) as _run:

            dataset = LLMDataset(self._dataset["train"], self._args, 
                self._template, True)  
            
            self._init_optimizer_scheduler(len(dataset), 
                self._args.epochs, lr=1e-5)
            
            dataloader = DataLoader(dataset, batch_size=self._args.batch_size, 
                                    shuffle=True, drop_last=True)

            acc_k = self._test(self._dataset["test"], 0, dataset.n_labels)
            exit()
            
            for epoch in range(self._args.epochs): 
                self._model._model = self._model._model.train()

                self._train(epoch, dataloader)        
                acc_k = self._test(self._dataset["test"], epoch, dataset.n_labels)
                self._model.save(self._args.save_dir, self._args.experiment, epoch, acc_k)       
                


if __name__ == "__main__":

    print("LLM-Baseline Phi-4")

    parser = Arguments()
    parser.parser.add_argument("--error_types",
        help="The datatype for loading the correct error definítions and templates.",
        type=str,
        choices=["abceval", "fedi", "soda_eval"],
        required=True)
     
    llm_baseline = LLMFinetuning(parser.parse_args())
    llm_baseline.run()
