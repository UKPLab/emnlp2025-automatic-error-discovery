'''
    This script contains the implementation for KNNContrastive, following the reference implementation provided at https://github.com/zyh190507/KnnContrastiveForOOD/
'''
import random as rd

import torch
import mlflow

from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
from arguments import Arguments
from dataset import BaseDataset
from abs_manager import Manager, Model

class KNNContrastiveManager(Manager):

    def __init__(self, args:Namespace):
        super(KNNContrastiveManager, self).__init__(args, 
            Model.KNN_CONTRASTIVE)
        #
        # train_mocoknn in trainer.py (https://github.com/zyh190507/KnnContrastiveForOOD/blob/main/trainer.py#L484)
        #
        self._ce_criterion = torch.nn.CrossEntropyLoss()
        self._cl_criterion = torch.nn.CrossEntropyLoss()


    def _get_context(self, context:list[str], error_utterance:dict) -> str:
        context = context.replace("\nuser: ", " [SEP] ")
        context = context.replace("\nsystem: ", " [SEP] ")
        context = context.replace("user: ", "[CLS] ")
        context = context.replace("system: ", "[CLS] ")
        context = context + " [SEP] " + error_utterance + " [SEP]"
        return context
        
                    
    def _knn_momentum_contrastive_learning(self, epoch:int, 
        dataloader:DataLoader, label_sample_mapping:dict[list[dict]]):
        
        # set the model to train mode
        self._model = self._model.train()

        pbar = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):
            
            sequences = [self._get_context(c, e) for c, e in zip(batch["context"], batch["error_utterance"]["text"])]
            error_types = batch["error_type"]

            # in their code, Zhou et al. first something they refer to as 
            # "negative" dataset. This is basically a mapping of all samples 
            # from the train split to their labels (do know why they call it 
            # "negative"). Next, they randomly sample 3 data points from this 
            # dataset per label (error type). They do not exclude data points 
            # if they are equal to samples from the current batch.

            positives = []
            for error_type in error_types.tolist():
                positives += rd.sample(label_sample_mapping[error_type], 
                    self._args.positives)
                                                        
            output = self._model.tokenize_and_forward(sequences, error_types, 
                positives)
                      
            # classification loss
            ce_loss = self._ce_criterion(output["logits_cls"], 
                error_types.to(self._args.device))

            # contrastive loss            
            if output["logits_contrastive"] is not None:
                labels_con = torch.zeros(output["logits_contrastive"].shape[0], 
                    dtype=torch.long).cuda()

                contrastive_loss =\
                    self._cl_criterion(output["logits_contrastive"], labels_con)
            else:
                contrastive_loss = 0.0
            
            loss =\
                contrastive_loss * self._args.contrastive_weighting +\
                    ce_loss * (1 - self._args.contrastive_weighting)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            loss_dict = {"overall_loss": round(loss.item(), 3),
                         "ce_loss": round(ce_loss.item(), 3), 
                         "contrastive_loss": round(contrastive_loss.item(), 3)}
            
            mlflow.log_metrics(loss_dict, i * epoch) 
            pbar.set_postfix(loss_dict)
            pbar.update(1)
   

    def run(self):
                        
        with mlflow.start_run(run_name=self._args.experiment) as _run:
                        
            if self._args.epochs:                                
                print("Starting KNN Contrastive Learning with Momentum")
                
                dataset = BaseDataset(self._dataset["train"], 
                    cli_args=self._args, n_labels=self._args.n_labels,shuffle=True)
                
                label_sample_mapping = {}
                for sample in dataset:
                    if sample["error_type"] not in label_sample_mapping:
                        label_sample_mapping[sample["error_type"]] =\
                            [sample["context"]]
                    else:                        
                        label_sample_mapping[sample["error_type"]].\
                            append(sample["context"])
                                    
                self._init_optimizer_scheduler(len(dataset), 
                    self._args.epochs, lr=5e-5)
                
                dataloader = DataLoader(dataset, 
                    batch_size=self._args.batch_size, shuffle=True,
                    drop_last=True)
                    
                for epoch in range(self._args.epochs):
                    self._knn_momentum_contrastive_learning(epoch, dataloader, 
                        label_sample_mapping)

                    acc = self._test(self._dataset["test"], True, epoch,    
                        dataloader.dataset.n_labels, field="context")
                    
                    self._model.save_model("knn_contrastive", epoch, acc, 
                        self._args.save_dir)


if __name__ == "__main__":

    print("\nKNN-Contrastive Learning for Out-of-Domain Intent Classification")

    parser = Arguments()
    parser.parser.add_argument("--positives",
        help="The number of positive samples per error type.",
        type=int,
        required=True)
    parser.parser.add_argument("--contrastive_weighting",
        help="Weighting factor for contrastive learning",
        type=float,
        default=0.2,
        required=True)
 
    knn_contrastive = KNNContrastiveManager(parser.parse_args())
    knn_contrastive.run()
    