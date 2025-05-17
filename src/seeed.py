'''
    This code contains the implementation of SEEED as described in our paper.
'''
import torch
import mlflow

from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
from arguments import Arguments
from dataset import SeeedDataset
from losses import SoftNNLoss
from abs_manager import Manager, Model

class OurApproachManager(Manager):

    def __init__(self, args:Namespace):
        super(OurApproachManager, self).__init__(args, 
            Model.DIALOGUE_SUMMARY)

        self._ce_criterion = torch.nn.CrossEntropyLoss()        
        self._cl_criterion = SoftNNLoss(temperature=0.2, margin=0.3)
            
    def _get_context(self, context:list[str], error_utterance:dict) -> str:
        context = context.replace("\nuser: ", " [SEP] ")
        context = context.replace("\nsystem: ", " [SEP] ")
        context = context.replace("user: ", "[CLS] ")
        context = context.replace("system: ", "[CLS] ")
        context = context + " [SEP] " + error_utterance + " [SEP]"
        return context
    
    def _contrastive_learning_dataloader(self, n_labels:list[int]=None):   

        def my_collate_fn(_batch:list[dict]) -> list[dict]:            
            batch = {}

            def _add(a:dict, b:dict, k:str):
                for _k, _v in b.items():
                    if _k not in a[k]:
                        a[k][_k] = [_v]
                    else:
                        a[k][_k].append(_v)

            for sample in _batch:
                for k, v in sample.items():
                    if k not in batch:
                        if isinstance(v, dict):
                            batch[k] = {}
                            _add(batch, v, k)
                        else:
                            batch[k] = [v]
                    else:
                        if isinstance(v, dict):
                            _add(batch, v, k)
                        else:
                            batch[k].append(v)

            return batch
        
        dataset = SeeedDataset(self._dataset["train"], 
            self._model, cli_args=self._args, shuffle=True, 
            n_labels=n_labels)
        dataloader = DataLoader(dataset, 
            batch_size=self._args.batch_size, shuffle=True, drop_last=True, collate_fn=my_collate_fn)
        return dataloader
                            
    def ours_training(self, epoch:int, dataloader:DataLoader):
        
        # set the model to train mode
        self._model = self._model.train()

        pbar = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):

            targets = torch.tensor(batch["target"], device=self._args.device)

            samples = [self._get_context(c, e) for c, e in zip(batch["context"], batch["error_utterance"])]
            summaries = batch["error_desc_unlabeled"]
            samples_output =\
                self._model.tokenize_and_forward(samples, summaries)

            positives = [self._get_context(c, e) for c, e in 
                zip(batch["pos_context"], batch["pos_error_utts"])]
            negatives = [[self._get_context(c, e) for c, e in 
                zip(_c, _e)][0] for _c, _e in zip(batch["neg_context"], batch["neg_error_utts"]) ]
            negatives_summaries = [b[0] for b in batch["neg_summary"]]    
            negatives_labels = torch.tensor([b[0] for b in batch["neg_labels"]]).to(self._args.device)

            positives_output =\
               self._model.tokenize_and_forward(positives, batch["pos_summary"])
            negatives_output =\
               self._model.tokenize_and_forward(negatives, negatives_summaries)
                                                    
            #
            # this is SofNearestNeighborLoss
            #            
            labeled_cl_loss =\
                self._cl_criterion(samples_output["embeddings"], positives_output["embeddings"], targets, targets, negative_embeddings=negatives_output["embeddings"], negative_labels=negatives_labels)
            
            #
            # cross-entropy loss
            #
            ce_loss = self._ce_criterion(samples_output["logits"], 
                targets)

            # loop style
            loss = 0.5 * ce_loss + labeled_cl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            loss_dict = {"overall_loss": round(loss.item(), 3),
                         "ce_loss": round(ce_loss.item(), 3),
                         "cl_loss": round(labeled_cl_loss.item(), 3)}
            
            mlflow.log_metrics(loss_dict, i * epoch) 
            pbar.set_postfix(loss_dict)
            pbar.update(1)
   
    def run(self):
                        
        with mlflow.start_run(run_name=self._args.experiment) as _run:
                        
            print("Our Approach")

            dataloader = self._contrastive_learning_dataloader()

            self._init_optimizer_scheduler(len(dataloader.dataset), 
                self._args.epochs, lr=1e-5)

            for epoch in range(self._args.epochs):

                if epoch % self._args.resample_rate == 0:
                    dataloader = self._contrastive_learning_dataloader(n_labels=dataloader.dataset.dataset.n_labels)

                self.ours_training(epoch, dataloader)

                acc = self._test(self._dataset["test"], True, 0, 
                    dataloader.dataset.dataset.n_labels, 
                    field="context_summary", nnk_means=True)#, n_clusters=5, n_nonzero_coefs=5)
                
                self._model.save_model("ours", epoch, acc, 
                    self._args.save_dir)

if __name__ == "__main__":

    print("\nKNN-Contrastive Learning for Out-of-Domain Intent Classification")

    parser = Arguments()
    parser.parser.add_argument("--num_negatives",
        help="The number of negative samples per error type.",
        type=int,
        required=True)
    parser.parser.add_argument("--contrastive_weighting",
        help="Weighting factor for contrastive learning",
        type=float,
        default=0.2,
        required=True)
    parser.parser.add_argument("--topk",
        help="The number of top-k samples to consider for calculating inconsistency.",
        type=int,
        required=True)
    parser.parser.add_argument("--positives",
        help="The number of negative samples per error type.",
        type=int,
        required=True)
    parser.parser.add_argument("--resample_rate",
        help="The ratio of epochs after which to resample the dataset for contrastive learning.",
        type=int,
        required=True)    

    our_approach = OurApproachManager(parser.parse_args())
    our_approach.run()
    