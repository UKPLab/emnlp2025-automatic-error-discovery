
'''
    This script contains the code for the abstract Manager class which provides basic funtionality for training and evaluation required in all approaches.
'''
import json
import torch
import mlflow
import argparse
import numpy as np
import torch.optim as optim

from enum import Enum
from datasets import load_dataset, Dataset as HfDataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from dataset import BaseDataset
from nnk_means import NNKMeans
from models import Loop, SimpleContrastiveEncoder, KNNContrastive, LLMFinetuning, DialogueSummaryEncoder
from utility import pp_args, accuracy_metrics, hungarian

class Model(Enum):
    LOOP = "loop"
    SIMPLE_CONTRASTIVE = "simple_contrastive"
    KNN_CONTRASTIVE = "knn_contrastive"
    LLM_FINETUNING = "llm_finetuning"
    DIALOGUE_SUMMARY = "dialogue_summary"

class Manager:

    def __init__(self, args:argparse.Namespace, model:Model):
        pp_args(args)

        self._args = args

        self._dataset = load_dataset(args.dataset, token=args.token)
            #download_mode="force_redownload")       
        self._classes = list(set(self._dataset["train"]["label"])) 
        
        if model == Model.LOOP:
            self._model = Loop(len(self._classes), device=self._args.device)
        elif model == Model.SIMPLE_CONTRASTIVE:
            self._model = SimpleContrastiveEncoder(len(self._classes), device=self._args.device)
        elif model == Model.KNN_CONTRASTIVE:
            self._model = KNNContrastive(len(self._classes), 
                device=self._args.device, positives=self._args.positives)
        elif model == Model.DIALOGUE_SUMMARY:
            self._model = DialogueSummaryEncoder(len(self._classes), 
                device=self._args.device)
        elif model == Model.LLM_FINETUNING:
            self._model = LLMFinetuning(self._args.pretrained) if self._args.pretrained else LLMFinetuning()
        else:
            raise ValueError("Unsupported approach.")

        if model != Model.LLM_FINETUNING and self._args.pretrained:
            print("load pretrained model...")
            original_parameters = self._model.parameters()
            self._model = self._model.load_model(self._args.pretrained)
            loaded = self._check_loading_pretrained(original_parameters, 
                self._model.parameters())
            print(f"loading pretrained model successful: {loaded}")
            if not loaded:
                raise RuntimeError("loaded and pretrained have same weights.")


    def _check_loading_pretrained(self, params_1:torch.Tensor, 
        params_2:torch.Tensor) -> bool:
        for p1, p2 in zip(params_1, params_2):
            if p1.data.ne(p2.data).sum() > 0:
                return True
            return False
        

    def _init_optimizer_scheduler(self, samples:int, epochs:int, 
        lr:float=1e-5) -> None:

        self._optimizer =\
            optim.AdamW(self._model.parameters(), lr=lr)
        num_training_steps =\
            (samples // self._args.batch_size) * epochs
        self._scheduler = get_linear_schedule_with_warmup(self._optimizer, 
            num_warmup_steps=0, num_training_steps=num_training_steps)
        print(f"number of training steps for scheduler: {num_training_steps}")


    def _test(self, dataset:HfDataset, mlflow_log=False, iteration:int=0, n_labels:list[int]=None, field:str="", nnk_means:bool=False, n_clusters:int=100, n_nonzero_coefs:int=50)\
        -> None:

        def _get_context(error_utts:list[str], contexts:list[str]):
            sequences = []
            for eu, context in zip(error_utts, contexts):
                context = context.replace("\nuser: ", " [SEP] ")
                context = context.replace("\nsystem: ", " [SEP] ")
                context = context.replace("user: ", "[CLS] ")
                context = context.replace("system: ", "[CLS] ")
                context = context + " [SEP] " + eu + " [SEP]"
                sequences.append(context)
            return sequences
        
        print("Creating test dataset")


        _dataset = BaseDataset(dataset, self._args, n_labels=[100], 
            shuffle=False)
        
        dataloader = DataLoader(_dataset, batch_size=self._args.batch_size, 
            drop_last=True)
        
        self._model = self._model.eval()
                    
        # first embed all the error user utterances (for 
        # clustering)
        _embeds =\
            torch.empty((0, self._model.hidden_size)).to(self._args.device)
        
        labels, text_labels = [], []
        with torch.no_grad():            
            for batch in dataloader: 
                
                labels += [e.item() for e in batch["error_type"]]
                text_labels += batch["error_type_str"]

                if field == "context_summary":
                    sequences = _get_context(batch["error_utterance"]["text"], 
                        batch["context"])
                    summaries = batch["error_desc_unlabeled"]
                    output = self._model.tokenize_and_forward(sequences, 
                        summaries)["embeddings"]
                else:
                    if field == "context":
                        sequences =\
                            _get_context(batch["error_utterance"]["text"], 
                            batch["context"])
                    else:
                        sequences = batch[field]                
                    output = self._model.tokenize_and_forward(sequences)["avg_hidden"]

                _embeds = torch.cat((_embeds, output))
        
        # create a list of unique label identifiers
        unique_labels = list(set(labels))

        h_mean, acc_k, acc_n, nmi, ari, acc_class_wise = [], [], [], [], [], []

        if nnk_means:
            
            _labels = np.array(labels)
            _max = max(labels)
            mask = _labels < _max
            _embeds = _embeds[mask]
            _labels = _labels[mask]
            labels = _labels.tolist()

            model = NNKMeans(n_clusters=n_clusters, n_nonzero_coefs=n_nonzero_coefs, n_classes = len(unique_labels))\
                .fit(_embeds, torch.tensor(labels).to(self._args.device))
            preds, _ = model.predict(_embeds)
                        
            mapping, kmeans_labels = hungarian(preds, labels)
            _h_mean, _acc_k, _acc_n, _class_wise_acc = accuracy_metrics(preds, 
                labels, mapping=mapping, n_labels=n_labels)
            
            acc_class_wise.append(_class_wise_acc)
            h_mean.append(_h_mean)
            acc_k.append(_acc_k)
            acc_n.append(_acc_n)
            nmi.append(normalized_mutual_info_score(kmeans_labels, preds))
            ari.append(adjusted_rand_score(kmeans_labels, preds))
        else:
            # run kmeans
            # since kmeans results may, repeat 5 times and average the results.
            for i in range(5):                
                km = KMeans(n_clusters = len(unique_labels))\
                    .fit(_embeds.cpu().numpy())
                preds = km.labels_

                if len(unique_labels) != len(np.unique(preds)):
                    # if the number of distinct clusters is not equal to the number 
                    # of clusters, this was an invalid run.
                    continue  

                mapping, kmeans_labels = hungarian(preds, labels)
                _h_mean, _acc_k, _acc_n, _class_wise_acc =\
                    accuracy_metrics(preds, labels, mapping=mapping, n_labels=n_labels)
                
                acc_class_wise.append(_class_wise_acc)
                h_mean.append(_h_mean)
                acc_k.append(_acc_k)
                acc_n.append(_acc_n)
                nmi.append(normalized_mutual_info_score(kmeans_labels, preds))
                ari.append(adjusted_rand_score(kmeans_labels, preds))

        if mlflow_log:

            metrics = {"h-mean": round(sum(h_mean)/len(h_mean), 3),
                "acc_k": round(sum(acc_k)/len(acc_k), 3),
                "acc_n": round(sum(acc_n)/len(acc_n), 3),
                "nmi_score": round(sum(nmi)/len(nmi), 3),
                "ari_score": round(sum(ari)/len(ari), 3)}
            
            _temp = {}
            for class_wise in acc_class_wise:
                for k, v in class_wise.items():
                    if k not in _temp:
                        _temp[k] = [v]
                    else:
                        _temp[k].append(v)            
            for k, v in _temp.items():
                metrics[f"class_{k}_accuracy"] = round(sum(v)/len(v),3) 
            
            print(json.dumps(metrics))
            mlflow.log_metrics(metrics, iteration)
            
        return round(sum(h_mean)/len(h_mean), 3) if sum(h_mean) > 0.0\
            else round(sum(acc_k)/len(acc_k), 3) 