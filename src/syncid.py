'''
    This script contains the code for SynCID adapted from USNID (https://github.com/thuiar/TEXTOIR/tree/main)
'''
import torch
import mlflow
import numpy as np

from itertools import batched

from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader

from sentence_transformers.util import semantic_search

from arguments import Arguments
from losses import SupConLoss
from dataset import BaseDataset
from abs_manager import Manager, Model

# instead of 10%, we keep 50% of the data from known classes as labeled to match the amount of training data SynCID used per intent

class SynCIDManager(Manager):

    def __init__(self, args: Namespace):
        super(SynCIDManager, self).__init__(args, Model.SIMPLE_CONTRASTIVE)

        self._ce_criterion = torch.nn.CrossEntropyLoss()
        self._ucl_criterion = SupConLoss()
        self._scl_criterion = SupConLoss()
        self._sanf_criterion = SupConLoss()


    def _get_context(self, context:list[str], error_utterance:dict) -> str:
        context = context.replace("\nuser: ", " [SEP] ")
        context = context.replace("\nsystem: ", " [SEP] ")
        context = context.replace("user: ", "[CLS] ")
        context = context.replace("system: ", "[CLS] ")
        context = context + " [SEP] " + error_utterance + " [SEP]"
        return context

    
    def _update_knn(self, dataset:BaseDataset):
        
        _embeds_uts = torch.empty((0, self._model.projection_dim))\
                .to(self._args.device)
        _embeds_descs = torch.empty((0, self._model.projection_dim))\
            .to(self._args.device)
        
        with torch.no_grad():          
            for batch in batched(dataset, self._args.batch_size):
                _uts_seqs = [b["error_ut"] for b in batch]
                _descs_seqs = [b["error_desc_unlabeled"] for b in batch]
                _embeds_uts =\
                    torch.cat((_embeds_uts, self._model.tokenize_and_forward(_uts_seqs)["projections"]))
                _embeds_descs =\
                    torch.cat((_embeds_descs, self._model.tokenize_and_forward(_descs_seqs)["projections"]))
        
        return _embeds_uts, _embeds_descs

                    
    def _space_alignment_contrastive_learning(self, epoch:int, 
        dataloader:DataLoader):

        # set the model to train mode
        self._model = self._model.train()  

        pbar = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):

            batch_size = len(batch["error_ut"])
            split_size =  batch_size // 2
            
            contexts = [self._get_context(c, e) for c, e in zip(batch["context"], batch["error_utterance"]["text"])]
            
            #error_uts_l = batch["error_ut"][:split_size]
            error_uts_l = contexts[:split_size]
            o_error_uts_l =\
                self._model.tokenize_and_forward(error_uts_l)
            
            error_desc_l = batch["error_desc_labeled"][:split_size]
            o_error_desc_l =\
                self._model.tokenize_and_forward(error_desc_l)
            
            #error_uts_u = batch["error_ut"][split_size:]
            error_uts_u = contexts[split_size:]
            o_error_uts_u =\
                self._model.tokenize_and_forward(error_uts_u)
            
            error_desc_u = batch["error_desc_unlabeled"][split_size:]
            o_error_desc_u =\
                self._model.tokenize_and_forward(error_desc_u)
            
            # calculate ce loss
            ce_labels = batch["error_type"][:split_size].to(self._args.device)

            ce_loss = self._ce_criterion(o_error_uts_l["logits"],
                ce_labels)
            
            # calculate contrastive loss unlabeled
            ucl_loss =\
                self._ucl_criterion(o_error_uts_u["projections"], 
                o_error_desc_u["projections"], split_size)

            # calculate supervised contrastive loss
            scl_loss =\
                self._scl_criterion(o_error_uts_l["projections"], 
                o_error_desc_l["projections"], split_size, ce_labels)
            
            # calculate overall loss
            loss = ce_loss + self._args.alpha * ucl_loss\
                + self._args.beta * scl_loss
            
            loss.backward()
            self._optimizer.step()            
            self._scheduler.step()
            self._optimizer.zero_grad()
            
            loss_dict = {"overall_loss": round(loss.item(), 3),
                         "ce_loss": round(ce_loss.item(), 3), 
                         "scl_loss": round(scl_loss.item(), 3), 
                         "ucl_loss": round(ucl_loss.item(), 3)}
            
            mlflow.log_metrics(loss_dict, i + len(dataloader) * epoch) 
            pbar.set_postfix(loss_dict)
            pbar.update(1)


    def _space_alignment_neighbor_filtering(self, epoch:int, 
        dataloader:DataLoader, dataset:BaseDataset, topk:int=100):

        # set the model to train mode
        self._model = self._model.train()   

        if epoch % 1 == 0:
            # update embeddings for nearest neighbors (top-k cosine sim 
            # semantic search)
            print (f"Epoch {epoch}: Recalculating nearest neighbors.")
            self._embeds_uts, self._embeds_descs = self._update_knn(dataset)
                                  
        pbar = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):

            contexts = [self._get_context(c, e) for c, e in 
                zip(batch["context"], batch["error_utterance"]["text"])]
            
            # for each sample in batch, select the the top-k from the 
            # intersection as a positive sample. Then consider everything else 
            # in the batch as negative sample
            #error_uts = self._model.tokenize_and_forward(batch["error_ut"])
            error_uts = self._model.tokenize_and_forward(contexts)
            error_descs =\
                self._model.tokenize_and_forward(batch["error_desc_unlabeled"])

            nbrs_uts = semantic_search(error_uts["projections"], 
                self._embeds_uts, top_k=topk+1)            
            nbrs_descs = semantic_search(error_descs["projections"], 
                self._embeds_descs, top_k=topk+1)
            
            # turn the dictionaries into lists and drop the first entry (which 
            # is the sample itself)
            nbrs_uts = [[_l["corpus_id"] for _l in l[1:]] for l in nbrs_uts]
            nbrs_descs = [[_l["corpus_id"] for _l in l[1:]] for l in nbrs_descs]

            # We remove all neighboring utterances whose descriptions are not 
            # part of the neighboring descritpions -> we retain the 
            # intersection of indices for neighboring utterances and 
            # neighboring descriptions
            intersections = [np.intersect1d(nbrs_ut, nbrs_desc).tolist()
                for nbrs_ut, nbrs_desc in zip(nbrs_uts, nbrs_descs)]

            # if no intersection was found for one sample, use the sample idx 
            # as a fallback
            for i in range(len(intersections)):                
                if len(intersections[i]) == 0:                    
                    intersections[i] = [batch["index"][i].item()]

            positives = [dataset[intersection[0]] for intersection in 
                intersections]
            positives = [self._get_context(p["context"], p["error_utterance"]["text"]) for p in positives]            
            positives = self._model.tokenize_and_forward(positives)

            # create labels; considered as a diagonal matrix
            labels =\
                torch.tensor(range(self._args.batch_size)).to(self._args.device)
            
            loss = self._sanf_criterion(error_uts["projections"], 
                positives["projections"], self._args.batch_size, labels)

            loss.backward()            
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            self._optimizer.step()            
            self._optimizer.zero_grad()
            self._scheduler.step()    

            loss_dict = {"sanf_loss": round(loss.item(), 3)}            
            mlflow.log_metrics(loss_dict, i + len(dataloader) * epoch) 
            pbar.set_postfix(loss_dict)
            
            pbar.update(1)  


    def run(self):

        # first, we remove all entries from the dataset with empty description 
        # fields
        for split in self._dataset:
            indices = [idx for idx, sample in enumerate(self._dataset[split])
                if sample["labeled_descriptions"] != "" and sample["unlabeled_descriptions"] != ""]
            self._dataset[split] = self._dataset[split].select(indices)

        # if we shuffle the dataset itself, we get issues with finding the 
        # right entries for neighbor ids in step 2
        if self._args.n_labels:
            _dataset = BaseDataset(self._dataset["train"], cli_args=self._args, 
                n_labels=self._args.n_labels, shuffle=False)
        else:
            _dataset = BaseDataset(self._dataset["train"], cli_args=self._args, 
                shuffle=False)
        
        dataloader = DataLoader(_dataset, batch_size=self._args.batch_size, 
            shuffle=True, drop_last=True)
                                
        with mlflow.start_run(run_name=self._args.experiment) as _run:
            
            if self._args.epochs:
                # step 1 - space alignment with contrastive learning
                print("Starting space alignment with contrastive learning")
                self._init_optimizer_scheduler(len(_dataset), 
                    self._args.epochs, lr=1e-5)
                for epoch in range(self._args.epochs):
                    self._space_alignment_contrastive_learning(epoch, 
                        dataloader)        
                    acc = self._test(self._dataset["test"], True, epoch, 
                        _dataset.n_labels, field="context")
                    print(f"saving model to {self._args.save_dir}")
                    self._model.save_model("syncid", epoch, acc, 
                        self._args.save_dir)

            if self._args.epochs_2:
                print("Starting space alignment with neighbor filtering")
                
                # step 2 - space alignment with neighbor filtering
                dataloader = DataLoader(_dataset, 
                    batch_size=self._args.batch_size, shuffle=True, drop_last=True)
                
                self._init_optimizer_scheduler(len(_dataset),
                    self._args.epochs_2, lr=1e-10)
                
                for epoch in range(self._args.epochs_2):
                    self._space_alignment_neighbor_filtering(epoch, dataloader,
                        _dataset)
                    acc = self._test(self._dataset["test"], True, epoch, 
                        _dataset.n_labels, field="context")
                    self._model.save_model("syncid", epoch, acc, 
                        self._args.save_dir)


if __name__ == "__main__":

    print("\nSynergizing Large Language Models and Pre-Trained Smaller Models for Conversational Intent Discovery (SynCID)")

    parser = Arguments()
    parser.parser.add_argument("--epochs_2", 
        help="The number of epochs for space alignment with neighbor filter.", 
        type=int,
        required=False)
    parser.parser.add_argument("--alpha", 
        help="The weighting factor for unsuper. contrastive loss.", 
        type=float,
        default=0.8,
        required=True)
    parser.parser.add_argument("--beta", 
        help="The weighting factor for super. contrastive loss.", 
        type=float,
        default=0.7,
        required=True)
              
    syncid = SynCIDManager(parser.parse_args())
    syncid.run()
    