'''
    This script contains the code for training and evaluating LOOP, adapted from the reference code provided in GitHub (https://github.com/Lackel/LOOP/tree/main).
'''
import torch
import mlflow

from tqdm import tqdm
from argparse import Namespace
from transformers import BitsAndBytesConfig, LlamaForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from arguments import Arguments
from dataset import BaseDataset, LISDataset
from abs_manager import Manager, Model
from losses import SupConLoss
from error_definitions import FEDI, ABC_EVAL, SODA_EVAL

class LoopManager(Manager):

    def __init__(self, args:Namespace):
        super(LoopManager, self).__init__(args, Model.LOOP)

        print(args)

        if self._args.epochs_2:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self._llm = LlamaForCausalLM.from_pretrained(self._args.llm_path, 
                quantization_config=bnb_config)
            self._llm_tokenizer =\
                AutoTokenizer.from_pretrained(self._args.llm_path)

        self._cl_criterion = SupConLoss()
        self._ce_criterion = torch.nn.CrossEntropyLoss()


    def _contrastive_learning_dataloader(self):
        if self._args.error_types == "FEDI":
            error_definitions = FEDI 
        elif self._args.error_types == "ABC_EVAL":
            error_definitions = ABC_EVAL 
        else:
            error_definitions = SODA_EVAL
        dataset = LISDataset(self._dataset["train"], 
            self._model.foundation_model, self._llm, self._llm_tokenizer, error_definitions, self._args)
        dataloader =\
            DataLoader(dataset, batch_size=self._args.batch_size, 
                shuffle=True, drop_last=True)
        return dataloader


    def _get_contrastive_mask(self, anchor_indices, neighbor_indices, targets):

        # create mask template with dimensions batch_size x batch_size
        mask = torch.zeros(anchor_indices.shape[0], anchor_indices.shape[0])

        for i, neighbor_idx in enumerate(neighbor_indices):
            # neighbors are expected to be of the same class as anchor (or at 
            # least similar)
            mask[i][i] = 1
            for j, anchor_idx in enumerate(anchor_indices):
                if anchor_idx in neighbor_idx:
                    mask[i][j] = 1 # if in neighbors                
                if (targets[i] == targets[j]):
                    mask[i][j] = 1 # if same labels                    
        return mask.to(self._args.device)

                    
                    
    def _get_contrastive_mask(self, anchor_indices, neighbor_indices, targets):

        # create mask template with dimensions batch_size x batch_size
        mask = torch.zeros(anchor_indices.shape[0], anchor_indices.shape[0])

        for i, neighbor_idx in enumerate(neighbor_indices):
            # neighbors are expected to be of the same class as anchor (or at 
            # least similar)
            mask[i][i] = 1
            for j, anchor_idx in enumerate(anchor_indices):
                if anchor_idx in neighbor_idx:
                    mask[i][j] = 1 # if in neighbors                
                if (targets[i] == targets[j]):
                    mask[i][j] = 1 # if same labels                    
        return mask.to(self._args.device)

                    
    def _multi_task_pretraining(self, epoch:int, labeled_dataloader:DataLoader, 
        unlabeled_dataloader:DataLoader):

        # set the model to train mode
        self._model = self._model.train()

        # creating a new iterator doesn't "consume" the original dataloader
        unlabeled_iter = iter(unlabeled_dataloader)

        pbar = tqdm(range(len(labeled_dataloader)))
        for i, labeled_batch in enumerate(labeled_dataloader):
            
            # Here we are following the original implementation. In their code, 
            # they just reiterate the unlabeled dataloader if it is smaller 
            # than the labeled dataloader.
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dataloader)
                unlabeled_batch = next(unlabeled_iter)
                        
            # loop only uses the utterances as input data here
            #labeled_input = labeled_batch["error_ut"]
            #unlabeled_input = unlabeled_batch["error_ut"]

            labeled_input = [self._get_context(c, e) for c, e in zip(labeled_batch["context"], labeled_batch["error_utterance"]["text"])]
            unlabeled_input = [self._get_context(c, e) for c, e in zip(unlabeled_batch["context"], unlabeled_batch["error_utterance"]["text"])]

            #labeled_input = labeled_batch["error_desc_unlabeled"]
            #unlabeled_input = unlabeled_batch["error_desc_unlabeled"]
            target = labeled_batch["error_type"]
                        
            # calculate classification loss
            labeled_output = self._model.tokenize_and_forward(labeled_input)
            ce_loss = self._ce_criterion(labeled_output["logits"],
                target.to(self._args.device))
            
            # get mlm loss; in their code, they are stacking the labeled and 
            # unlabeled input after tokenization. We concatenate the list of 
            # input strings and then pass everything through the tokenizer. 
            # Results in the same but is easier to understand...
            mlm_output = self._model.tokenize_and_forward(unlabeled_input + 
                labeled_input, mlm=True)
            mlm_loss = mlm_output["mlm_loss"]

            loss = ce_loss + mlm_loss

            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            loss_dict = {"overall_loss": round(loss.item(), 3),
                         "ce_loss": round(ce_loss.item(), 3), 
                         "mlm_loss": round(mlm_loss.item(), 3)}
            
            mlflow.log_metrics(loss_dict, i + len(labeled_dataloader) * epoch) 
            pbar.set_postfix(loss_dict)
            pbar.update(1)

    
    def _get_context(self, context:list[str], error_utterance:dict) -> str:
        context = context.replace("\nuser: ", " [SEP] ")
        context = context.replace("\nsystem: ", " [SEP] ")
        context = context.replace("user: ", "[CLS] ")
        context = context.replace("system: ", "[CLS] ")
        context = context + " [SEP] " + error_utterance + " [SEP]"
        return context
   

    def _refined_neighborhood_contrastive_learning(self, epoch:int, 
        dataloader:DataLoader):

        # set the model to train mode
        self._model = self._model.train()

        pbar = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):

            with torch.set_grad_enabled(True):
                samples = []
                for c, e in zip(batch["sample"]["context"], batch["sample"]["error_utterance"]["text"]):
                    samples.append(self._get_context(c, e))
                
                neighbors = []
                for c, e in zip(batch["neighbor"]["context"], batch["neighbor"]["error_utterance"]["text"]):
                    neighbors.append(self._get_context(c, e))
                
                anchor_output =\
                    self._model.tokenize_and_forward(samples, 
                    mlm=True, random_token_replace=True)
                neighbor_output = self._model.tokenize_and_forward(neighbors, mlm=True, random_token_replace=True)

                # get mask for contrastive loss and calculate contrastive loss
                mask = self._cl_criterion.get_contrastive_mask(batch["index"], 
                    batch["possible_neighbors"], batch["target"].tolist())
                
                contrastive_loss =\
                    self._cl_criterion(anchor_output["avg_hidden"], 
                        neighbor_output["avg_hidden"], batch_size=anchor_output["avg_hidden"].shape[0], mask=mask)
                
                # calculate cross-entropy loss
                cross_entropy_loss = self._ce_criterion(anchor_output["logits"],
                    batch["target"].to(self._args.device))

                # calculate overall loss and do the learning
                loss = 0.5 * cross_entropy_loss + contrastive_loss
                loss.backward()
                # in LOOP, they also clip the grad to 1 by default
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()

                loss_dict = {"overall_loss": round(loss.item(), 3),
                    "ce_loss": round(cross_entropy_loss.item(), 3), 
                    "cl_loss": round(contrastive_loss.item(), 3), 
                    "loss": round(loss.item(), 3)}
            
                mlflow.log_metrics(loss_dict, i + len(dataloader) * epoch) 
                pbar.set_postfix(loss_dict)
                pbar.update(1)

    def run(self):
                        
        with mlflow.start_run(run_name=self._args.experiment) as _run:
                        
            if self._args.epochs:                
                # step 1 - multitask pretraining
                print("Starting multitask pretraining")
                
                labeled_dataset = BaseDataset(self._dataset["train"], 
                    cli_args=self._args, shuffle=False)
                
                print(f"labeled_labels: {labeled_dataset.labels}")
                print(f"labeled_n_labels: {labeled_dataset.n_labels}")

                if len(labeled_dataset.n_labels) > 0:
                    print("in if")
                    _labels = [l for l in labeled_dataset.labels if l not in 
                        labeled_dataset.n_labels]
                    unlabeled_dataset = BaseDataset(self._dataset["train"], 
                        cli_args=self._args, n_labels=_labels, shuffle=False)
                else:
                    print("in else")
                    unlabeled_dataset = BaseDataset(self._dataset["train"], 
                        cli_args=self._args, shuffle=False, n_labels=labeled_dataset.n_labels)
                    
                print(f"unlabeled_labels: {unlabeled_dataset.labels}")
                print(f"unlabeled_n_labels: {unlabeled_dataset.n_labels}")
                                    
                labeled_dataloader = DataLoader(labeled_dataset, 
                    batch_size=self._args.batch_size, shuffle=True,
                    drop_last=True)
                
                unlabeled_dataloader = DataLoader(unlabeled_dataset, 
                    batch_size=self._args.batch_size, shuffle=True,
                    drop_last=True)
                
                print(f"labeled_dataloader: {len(labeled_dataloader)}")
                print(f"unlabeled_dataloader: {len(unlabeled_dataloader)}")
                
                self._init_optimizer_scheduler(len(labeled_dataset), 
                    self._args.epochs, lr=1e-5)
                            
                for epoch in range(self._args.epochs):
                    self._multi_task_pretraining(epoch, labeled_dataloader, 
                        unlabeled_dataloader)        
                    acc = self._test(self._dataset["test"], True, epoch, labeled_dataloader.dataset.n_labels, field="context")
                    self._model.save_model("loop", epoch, acc, 
                        self._args.save_dir)

            if self._args.epochs_2:
                # step 2 - refined neighborhood contrastive learning
                print("Starting refined neighborhood contrastive learning")
                dataloader = self._contrastive_learning_dataloader()
                self._init_optimizer_scheduler(len(dataloader.dataset),
                    self._args.epochs_2, lr=1e-5)
                
                print(dataloader.dataset.dataset.n_labels)

                acc = self._test(self._dataset["test"], True, 0, dataloader.dataset.dataset.n_labels, field="context")

                for epoch in range(self._args.epochs_2):
                    
                    # regenerate the dataset every five epochs (as in LOOP)
                    if epoch % 1 == 0:
                        dataloader = self._contrastive_learning_dataloader()
                    self._refined_neighborhood_contrastive_learning(epoch, 
                        dataloader)
                    acc = self._test(self._dataset["test"], True, epoch,  
                        dataloader.dataset.dataset.n_labels, field="context")
                    self._model.save_model("loop", epoch, acc, self._args.save_dir)


if __name__ == "__main__":

    print("\nGeneralized Category Discovery with Large Language Models in the Loop (Loop)")

    parser = Arguments()
    parser.parser.add_argument("--epochs_2", 
        help="The number of epochs for refined neighborhood contrastive learning.", 
        type=int,
        required=False)
    parser.parser.add_argument("--error_types", 
        help="The error types to consider for label generation (either 'FEDI' 'SODA_Eval', or 'ABC_EVAL').", 
        type=str,
        choices=["FEDI", "ABC_EVAL", "SODA_EVAL"],
        required=True)
    parser.parser.add_argument("--llm_path", 
        help="The path to the large language model to use for local inconsistency sampling.", 
        type=str,
        required=True) 
    
    loop = LoopManager(parser.parse_args())
    loop.run()
    