'''
    This script contains all dataset classes.
'''
import re
import random as rd
import math

import torch
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers.util import semantic_search
from argparse import Namespace

from models import BaseModel, DialogueSummaryEncoder
from nnk_means import NNKMeans


class BaseDataset(Dataset):

    def __init__(self, dataset:HfDataset, cli_args:Namespace, 
        n_labels:list[int]=None, shuffle:bool=True):

        print(f"n_labels: {n_labels}")

        """Base dataset class.

        Args:
            dataset (HfDataset): The original dataset as a Huggingface dataset 
                object.
            cli_args (Namespace): The passed cmd arguments.
            n_labels (list[int], optional): The list of labels to treat as 
                novel. If set, novelty from cmd arguments is ignored. This is to make sure, that subsequently created dataset objects can use the same list of novel labels. Defaults to None.
            shuffle (bool, optional): Whether or not to shuffle the dataset.
                Defaults to True.
        """

        self._cli_args = cli_args
        self._shuffle = shuffle

        # openness configuration
        self._labels = list(set(dataset["label"]))
        self._n_labels = n_labels if n_labels else rd.sample(self._labels, 
            k=math.floor(self._cli_args.novelty*len(self._labels)))
        dataset = dataset.select([i for i, l in enumerate(dataset["label"]) 
            if l not in self._n_labels])
        
        self._hf_dataset = dataset.shuffle(seed=42) if self._shuffle\
            else dataset
        self._error_turn = self._cli_args.error_turn
        
        print(f"\nnovelty: {self._cli_args.novelty}; novel labels: {self._n_labels}\n")

    def select(self, indices:list[int]):
        return BaseDataset(self._hf_dataset.select(indices), self._cli_args, 
            n_labels=self._n_labels, shuffle=self._shuffle)
        
    def __len__(self):
        return len(self._hf_dataset)
    
    def __getitem__(self, idx):
        sample = self._hf_dataset[idx]

        if len(sample["context"]) > 0:
            error_ut =\
                sample["error_utterance"]["text"] if not self._error_turn \
                else f"{sample['context'][-1]['text']} [SEP] {sample['error_utterance']['text']}"
        else:
            error_ut = sample["error_utterance"]["text"]

        context = [f"{ut['agent']}: {ut['text']}" for ut in sample["context"]]
        context = "\n".join(context)
        context = re.sub(r"{{\w+}}", "", context)

        return {"error_ut": f"[CLS] {error_ut} [SEP]",
                "error_desc_labeled": f"[CLS] {sample['labeled_descriptions']} [SEP]",
                "error_desc_unlabeled": f"[CLS] {sample['unlabeled_descriptions']} [SEP]",
                "error_type": sample["label"],
                "error_type_str": sample["label_text"],
                "index": idx,
                "error_utterance": sample["error_utterance"],
                "context": context,
                "additional_information": sample["additional_information"]}
    
    @property
    def hf_dataset(self):
        return self._hf_dataset
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def n_labels(self):
        return self._n_labels
    

class LLMDataset(Dataset):

    def __init__(self, dataset:HfDataset, cli_args:Namespace, template:dict, 
        shuffle:bool=True, n_labels:list[str]=None, mode:str="train", examples:dict[str]=None):
        
        self._mode = mode

        self._cli_args = cli_args

        self._n_labels = cli_args.n_labels

        self._examples = examples

        # returns a BaseDataset already considering the novelty
        if self._examples:
            self._dataset = BaseDataset(dataset, self._cli_args, n_labels=[100], shuffle=shuffle)
        else:
            # only for testing
            self._dataset = BaseDataset(dataset, self._cli_args, n_labels=[100], shuffle=shuffle)
        
        self._template = template

        # Consider only "not novel" error types.
        self._error_types =\
            [(sample["error_type"], sample["error_type_str"]) for sample in 
            self._dataset]
        self._error_types = list(set(self._error_types)) 
      
        
    def __len__(self):
        return len(self._dataset)
        
    def __getitem__(self, idx):

        sample = self._dataset[idx]

        error_ut =\
            f"{sample['error_utterance']['agent']}: {sample['error_utterance']['text']}"
        error_ut = re.sub(r"{{\w+}}", "", error_ut)

        # replace placeholder in base template.
        prompt = self._template["base_template_start"].replace("||CONTEXT||", 
                sample["context"])
        prompt = prompt.replace("||ERROR_UTTERANCE||", error_ut)
        for e in self._error_types:
            prompt += "\n\n" + self._template["error_types"][e[0]]
            
            if self._examples:
                print(self._n_labels)
                if e[0] not in self._n_labels:
                    prompt +=  "Here are three examples: "

                    examples = rd.sample(self._examples[e[1]], 1)

                    for ex in examples:
                        prompt += "\n\n" + ex["error_turn"]
        
        prompt += "\n\n" + self._template["base_template_end"]

        if self._examples:

            prompt = prompt.replace("<|end|><|assistant|>", " Here is an example: \n\n Decision: [Ignore Question] \n Reasoning: The system utterance completely disregards the user's query about how to obtain insurance coverage for an accident. The system transitions to unrelated advice on dental insurance without providing any information that addresses the user's original concern. This behavior is a clear example of an Ignore Question error.")

            prompt += "<|end|><|assistant|>"

        target = f"Decision: [{sample['error_type_str']}]"

        if "datatype" in vars(self._cli_args):
            if self._cli_args.datatype == "fedi":
                if len(sample["additional_information"]) > 0:
                    knowledge = f"Given the following knowledge document:\n\n{sample['additional_information']}"
                else:
                    knowledge = ""
                prompt = prompt.replace("{{KNOWLEDGE}}", knowledge)
            if self._cli_args.datatype == "soda_eval":
                if len(sample["additional_information"]) > 0:
                    target += f"\n\nReasoning:\n{sample['additional_information']}<|end|>"

        if self._mode == "train":
            sample["prompt"] = prompt + target
            sample["target"] = sample["prompt"]
        else:            
            sample["prompt"] = prompt
            sample["target"] = target

        return sample
    
    
    @property
    def mapping(self):
        return self._mapping
    
    @property
    def n_labels(self):
        return self._dataset.n_labels

    
class LISDataset(Dataset):

    def __init__(self, dataset:HfDataset, slm:BaseModel, 
        llm:PreTrainedModel, llm_tokenizer:PreTrainedTokenizer, error_definitions:dict, cli_args:Namespace, shuffle:bool=True):

        print(f"Creating LISDataset")

        self._dataset = BaseDataset(dataset, cli_args, 
            shuffle=shuffle, n_labels=cli_args.n_labels)
        self._clusters = len(list(set(dataset["label"])))
        self._error_turn = cli_args.error_turn
        self._shuffle = shuffle
        self._batch_size = cli_args.batch_size
        self._slm = slm
        self._llm = llm
        self._llm_tokenizer = llm_tokenizer
        self._device = cli_args.device
        self._llm_predicted_labels = {}
        self._error_definitions = error_definitions
        self._knn_indices, self._query_indices, self._km_preds =\
            self.local_inconsistency_sampling()      
        

    def local_inconsistency_sampling(self, topk:int=50):

        # preprocess data - create embeddings 
        dataloader = DataLoader(self._dataset, batch_size=self._batch_size, 
            shuffle=False)
        hidden_size = self._slm.hidden_size

        with torch.no_grad():                    
            embeds = torch.empty((0, hidden_size)).to(self._slm.device)
            for batch in dataloader:     
                sequences = []
                for eu, context in zip(batch["error_utterance"]["text"],
                    batch["context"]):
                    context = context.replace("\nuser: ", " [SEP] ")
                    context = context.replace("\nsystem: ", " [SEP] ")
                    context = context.replace("user: ", "[CLS] ")
                    context = context.replace("system: ", "[CLS] ")
                    context = context + " [SEP] " + eu + " [SEP]"
                    sequences.append(context)

                output = self._slm.tokenize_and_forward(sequences)
                embeds = torch.cat((embeds, output["avg_hidden"].detach()))
        
        # perform kMeans
        km = KMeans(n_clusters = self._clusters).fit(embeds.cpu().numpy())
        cluster_centers = km.cluster_centers_
        predicted_labels = km.labels_

        # get k nearest neighbors for each feature using semantic_search 
        # (cosine similarity) from sentence-transformers (as Faiss always 
        # destroys the scikit-learn installation for whatever reason...). It is 
        # still faster than using NearestNeighbors from sklearn.
        # semantic_search returns a list with one entry for each query. Each 
        # entry is again a list of dictionaries (of length topk) with the keys 
        # 'corpus_id' and 'score', sorted by decreasing cosine similarity 
        # scores. (List[List[Dict[str, Union[int, float]]]]).
        knn_result = semantic_search(embeds, embeds, top_k=topk+1)

        # turn the dictionaries into lists and drop the first entry (which is 
        # the sample itself)
        knn_result = [[_l["corpus_id"] for _l in l[1:]] for l in knn_result]

        # calculate the entropy using t-student distribution. Code from LOOP
        # (https://github.com/Lackel/LOOP/blob/main/utils/memory.py#L51)
        alpha =  1
        q = 1.0 / (1.0 + torch.sum((embeds.unsqueeze(1).cpu() -\
            cluster_centers) ** 2, dim=2) / alpha)
        q = q ** (alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, dim=1)).t()
        weight = q ** 2 / torch.sum(q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()
        p = F.softmax(p, dim=-1)
        entropy = -torch.sum(p * torch.log(p), 1)

        # sort for the samples with the highest entropy
        _, inx = torch.sort(entropy, descending=True)

        # collect the predicted classes of the nearest neighbors 
        neighbor_predictions = np.take(predicted_labels, knn_result, axis=0)
        # the first entry in neighbor_predictions is the top-1 neighor. Fill a 
        # "mask" with its predicted class as samples for calculating the local 
        # inconsistency degree.
        sample_pseudos = np.repeat(predicted_labels.reshape(-1,1), topk, axis=1)

        # for each neighbor's predicted class, count the number of deviations 
        # from the sample class (the class predicted for the top-1 neightbor).
        # This is the local inconsistency degree. Samples with locally 
        # inconsistent predictions are near decision boundaries and have a 
        # higher probability of falling into wrong clusters.
        lid = torch.tensor([np.sum(np.not_equal(neighbor_predictions[i], 
            sample_pseudos[i])) for i in range(neighbor_predictions.shape[0])])
        _, index = torch.sort(lid, descending=True)

        # collect the samples that have a high local inconsistency degree and a 
        # high entropy. In contrast to the original work (which only considers the top 500 samples here), we consider all samples, as our datasets are not as alarge as the usual intent discovery datasets.
        query_index = [i for i in index if i in inx]
                
        return knn_result, query_index, predicted_labels


    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, idx):

        # get the sample sample (current sample) 
        sample = self._dataset[idx]

        # collect the predicted classes of the nearest neighbors 
        try:
            neighbor_pred = np.take(self._km_preds, self._knn_indices[idx])
        except IndexError as e:
            print(f"No element found in knn indices for idx {idx}.")
            # if we cannot find neighbors for index idx, we naively assume the 
            # sample as its nearest neighbor. 
            neighbor_pred = np.array([self._km_preds[idx]]*self._batch_size)

        # collect the predicted labels of the two most related neighbors
        res = [neighbor_pred[0]]
        res = res + [i for i in neighbor_pred[1:]]
        res = res[:2] if len(res) >= 2 else res
        
        # randomly choose the two query samples
        np_knn_indices = np.array(self._knn_indices)
        res_0_preds = np.where(neighbor_pred==res[0])
        res_1_preds = np.where(neighbor_pred==res[1])
        q1 = int(np.random.choice(np_knn_indices[idx, res_0_preds][0], 1)[0])

        if len(res) == 1 or idx not in self._query_indices:            
            neighbor_idx = np.random.choice(self._knn_indices[idx], 1)[0]
        else:
            q2 =\
                int(np.random.choice(np_knn_indices[idx, res_1_preds][0], 1)[0])
            if self._llm_predicted_labels.get(idx, -1) == -1:
                neighbor_idx = self.query_llm(sample, q1, q2)
                self._llm_predicted_labels[idx] = neighbor_idx
            else:
                neighbor_idx = self._llm_predicted_labels[idx]
        neighbor = self._dataset[neighbor_idx]
        
        return {"sample": sample,
                "neighbor": neighbor,
                "possible_neighbors": torch.from_numpy(np_knn_indices[idx]),
                "target": sample["error_type"],
                "index": idx}
    

    def query_llm(self, sample:dict, q1:int, q2:int):

        def generate_context(context:str, error_utterance:dict) -> str:
            context += f"\nsystem: {error_utterance['text']}"
                        
            return context

        prompt = f"""Given is the following dialogue between a human user and a dialogue system:

{generate_context(sample["context"], sample["error_utterance"])}

The response generated by the system contains the following error: 
{self._error_definitions[sample['error_type_str']]['description']} 

Which of the following situations contains the same error type?

Choice 1:
{generate_context(self._dataset[q1]["context"], self._dataset[q1]["error_utterance"])}

Choice 2:
{generate_context(self._dataset[q1]["context"], self._dataset[q1]["error_utterance"])}

Please respond with 'Choice 1' or 'Choice 2' without explanation."""

        '''
        prompt = f"""Given is the following summary of an error from a conversation between a human user and a dialogue system:

{generate_context(sample["context"], sample["error_utterance"])}

{self._error_definitions[sample['error_type_str']]['description']} 
The response generated by the system contains such an error. 

Which of the following situations contains the same error type?

Choice 1:
{self._dataset[q1]["error_desc_unlabeled"]}

Choice 2:
{self._dataset[q2]["error_desc_unlabeled"]}

Please respond with 'Choice 1' or 'Choice 2' without explanation."""
        '''
        '''
        prompt = f"""Given is the following excerpt of a conversation between a human user and a dialogue system:

{create_turn(sample["error_ut"])}

{self._error_definitions[sample['error_type_str']]['description']} 
The response generated by the system contains such an error. 

Which of the following conversation turns contains the same error type?

Choice 1:
{create_turn(self._dataset[q1]["error_ut"])}

Choice 2:
{create_turn(self._dataset[q2]["error_ut"])}

Please respond with 'Choice 1' or 'Choice 2' without explanation."""
        '''

        try:
            tokenized = self._llm_tokenizer(prompt, return_attention_mask = True, 
            return_tensors="pt")
            with torch.no_grad():
                output = self._llm.generate(**tokenized.to(self._device),   
                    max_new_tokens=30, do_sample=True, top_p=0.92, top_k=0, early_stopping=True)[0]
                output = self._llm_tokenizer.decode(output, 
                    skip_special_tokens=False)
                output = output.split("without explanation.")[1]
            # in the loop code, they also use q1 as the fallback 
            # (in every case)
            return q2 if 'choice 2' in output.lower() else q1            
        except Exception as e:
            print(e)
            return q1
        
    @property
    def dataset(self):
        return self._dataset
    

class SeeedDataset(Dataset):

    def __init__(self, dataset:HfDataset, slm:DialogueSummaryEncoder, 
        cli_args:Namespace, shuffle:bool=True, n_labels:list[int]=None):

        _n_labels = n_labels if n_labels else cli_args.n_labels

        self._dataset = BaseDataset(dataset, cli_args, 
            shuffle=shuffle, n_labels=_n_labels)
        self._classes = len(list(set(dataset["label"])))        
        self._batch_size = cli_args.batch_size
        self._slm = slm
        self._num_negatives = cli_args.num_negatives

        self._data_class_based = {}
        for i, sample in enumerate(self._dataset):
            if sample["error_type"] not in self._dataset.n_labels:
                if sample["error_type"] in self._data_class_based:
                    self._data_class_based[sample["error_type"]].append(i)
                else:
                    self._data_class_based[sample["error_type"]] = [i]

        self._pos_pos, self._pos, self._soft_neg, self._hard_neg =\
            self.positive_negative_sampling(topk=cli_args.topk)
        

    def _get_context(self, context:list[str], error_utterance:dict) -> str:
        context = context.replace("\nuser: ", " [SEP] ")
        context = context.replace("\nsystem: ", " [SEP] ")
        context = context.replace("user: ", "[CLS] ")
        context = context.replace("system: ", "[CLS] ")
        context = context + " [SEP] " + error_utterance + " [SEP]"
        return context
                

    def _get_embeddings(self)\
        -> tuple[torch.Tensor, list[int]]:        
        dataloader = DataLoader(self._dataset, batch_size=self._batch_size, 
            shuffle=False)
        hidden_size = self._slm.hidden_size
       
        true_labels = []
        with torch.no_grad():                    
            embeds = torch.empty((0, hidden_size)).to(self._slm.device)
            for batch in dataloader:

                contexts = [self._get_context(c, e) for c, e in 
                    zip(batch["context"], batch["error_utterance"]["text"])]
                summaries = batch["error_desc_unlabeled"]
                true_labels += [e.item() for e in batch["error_type"]]

                output = self._slm.tokenize_and_forward(contexts, summaries)
                embeds = torch.cat((embeds, output["embeddings"].detach()))
        
        return embeds, true_labels
        
    
    def _min_max_normalization(self, score:list[float]) -> list[float]:
        _min = min(score)
        _max = max(score)
        
        min_max = []
        for s in score:
            try:
                temp = round((s - _min) / (_max - _min), 2)
            except ZeroDivisionError:
                temp = 0.0
            min_max.append(temp)
        
        return min_max
    
    
    def _entropy(self, embeds:torch.Tensor, cluster_centers:np.array)\
        -> list[float]:
        # calculate the entropy using t-student distribution. Code from LOOP
        # (https://github.com/Lackel/LOOP/blob/main/utils/memory.py#L51)        
        alpha =  1
        q = 1.0 / (1.0 + torch.sum((embeds.unsqueeze(1).cpu() -\
            cluster_centers) ** 2, dim=2) / alpha)
        q = q ** (alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, dim=1)).t()
        weight = q ** 2 / torch.sum(q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()
        p = F.softmax(p, dim=-1)

        # "entropy" represents the probability of a sample to belong to the 
        # predicted class (cluster). If the value is low, the sample may be far 
        # away from the cluster center, meaning that it can be easily 
        # misclassified. It has the shape [1 x len dataset], e.g., [12948] in 
        # the case of Soda-Eval train
        entropy = -torch.sum(p * torch.log(p), 1)

        return self._min_max_normalization(entropy.tolist())
        #return entropy.tolist()
    
    def _local_inconsistency_degree(self, predicted_labels:np.array, 
        embeds:torch.Tensor, topk:int):

        # get k nearest neighbors for each feature using semantic_search 
        # (cosine similarity) from sentence-transformers (as Faiss always 
        # destroys the scikit-learn installation for whatever reason...). It is 
        # still faster than using NearestNeighbors from sklearn.
        # semantic_search returns a list with one entry for each query. Each 
        # entry is again a list of dictionaries (of length topk) with the keys 
        # 'corpus_id' and 'score', sorted by decreasing cosine similarity 
        # scores. (List[List[Dict[str, Union[int, float]]]]).
        knn_result = semantic_search(embeds, embeds, top_k=topk+1)

        # turn the dictionaries into lists and drop the first entry (which is 
        # the sample itself)
        knn_result = [[_l["corpus_id"] for _l in l[1:]] for l in knn_result]

        # neighbor_predictions contains the top-k neighbors for each sample. 
        # it has the shape [len dataset x top-k], e.g., [12948, 50] in the case 
        # of Soda-Eval train
        neighbor_predictions = np.take(predicted_labels, knn_result, axis=0)

        # the first entry in neighbor_predictions is the top-1 neighor. Fill a 
        # "mask" with its predicted class as samples for calculating the local 
        # inconsistency degree.
        sample_pseudos = np.repeat(predicted_labels.reshape(-1,1), topk, axis=1)

        # for each neighbor's predicted class, count the number of deviations 
        # from the sample class (the class predicted for the top-1 neightbor).
        # This is the local inconsistency degree. Samples with locally 
        # inconsistent predictions are near decision boundaries and have a 
        # higher probability of falling into wrong clusters.
        lid = torch.tensor([np.sum(np.not_equal(neighbor_predictions[i], 
            sample_pseudos[i])) for i in range(neighbor_predictions.shape[0])])
        
        return self._min_max_normalization(lid.tolist())
        #return lid.tolist()
    

    def _lis_score(self, embeds:torch.Tensor, preds:np.array, 
        cluster_centers:np.array, topk:int) -> list[float]:

        entropy = self._entropy(embeds, cluster_centers)
        lid = self._local_inconsistency_degree(preds, embeds, topk)

        return [round((a + b)/2, 2) for a, b in zip(entropy, lid)]


    def positive_negative_sampling(self, topk:int=2):
        
        # preprocess data - get embeddings and true labels
        embeds, true_labels = self._get_embeddings()
                
        # perform kMeans
        nnk_means = NNKMeans(n_classes = self._classes)\
            .fit(embeds, torch.tensor(true_labels).to(self._slm.device))
        preds, cluster_centers = nnk_means.predict(embeds)

        # get lis score for each sample
        entropy = self._entropy(embeds, cluster_centers)
        lid = self._local_inconsistency_degree(preds, embeds, topk)
        score = [round((a + b)/2, 2) for a, b in zip(entropy, lid)]

        positives = {k: [] for k in range(self._classes)}
        negatives = {k: [] for k in range(self._classes)}
        positives_positives = {k: [] for k in range(self._classes)}

        # sample positives_positives, positives, and negatives.
        # positives: positives for a class X are all samples which have ground 
        #   truth target X but were wrongly classified
        # negatives: negatives for a class X are all samples which were 
        #   classified as X but have a different ground truth target
        # positives_positives: positives_positives are all samples that were 
        #   correctly classified
        for i, (target, pred, ent, incon, sc) in enumerate(zip(true_labels, 
            preds.tolist(), entropy, lid, score)):
            
            if target == pred:
                positives_positives[target].append((i, incon, ent, sc))
            else:                
                positives[target].append((i, incon, ent, sc))
                negatives[pred].append((i, incon, ent, sc))

        # Positives were misclassified and are sorted descending according to their score.        
        positives = {k: sorted(v, key=lambda item: item[3], reverse=True)
            for k, v in positives.items()}
        # sorting negatives descending based on lis means that those close to the decision boundary are in the first half, while those close to the cluster centers are in the second half.        
        negatives = {k: sorted(v, key=lambda item: item[1], reverse=True)
            for k, v in negatives.items()}
        
        soft_negatives = {}
        hard_negatives = {}

        for key, sorted_list in negatives.items():
            list_length = len(sorted_list)
            half_length = list_length // 2
            soft_negatives[key] = sorted_list[:half_length]
            hard_negatives[key] = sorted_list[half_length:]

        # we sort both soft and hard negatives descending according to their score.
        soft_negatives = {k: sorted(v, key=lambda item: item[3], reverse=True)
            for k, v in soft_negatives.items()}
        hard_negatives = {k: sorted(v, key=lambda item: item[3], reverse=True)
            for k, v in hard_negatives.items()}
        
        # positives_positives with a high lis score were correctly classified, 
        # but are located near decision boundary. Sort them descending 
        # according to their lis score.
        positives_positives = {k: sorted(v, key=lambda item: item[3], 
            reverse=True) for k, v in positives_positives.items()}

        print(f"Soft Positives: {sum([len(v) for k, v in positives_positives.items()])}")
        for k, v in positives_positives.items():
            print(f"\t\t{k}: {len(v)}")

        print(f"Hard Positives: {sum([len(v) for k, v in positives.items()])}")
        for k, v in positives.items():
            print(f"\t\t{k}: {len(v)}")

        print(f"Soft Negatives: {sum([len(v) for k, v in soft_negatives.items()])}")
        for k, v in soft_negatives.items():
            print(f"\t\t{k}: {len(v)}")    

        print(f"Hard Negatives: {sum([len(v) for k, v in hard_negatives.items()])}")
        for k, v in hard_negatives.items():
            print(f"\t\t{k}: {len(v)}")
               
        return positives_positives, positives, soft_negatives, hard_negatives


    def __getitem__(self, idx):

        # get the sample sample (current sample) 
        sample = self._dataset[idx]

        error_type = sample["error_type"]

        valid_classes = [i for i in range(self._classes) if i not in self._dataset.n_labels]

        # init pos to avoid "not initialized warning"
        pos = idx

        # positive
        if len(self._pos[error_type]) > 0:
            if len(self._pos_pos[error_type]) > 0:
                r = rd.choice([0, 1])                
                pos = self._pos[error_type].pop(0) if r == 0 else\
                    rd.choice(self._pos_pos[error_type])                
            else:
                pos = self._pos[error_type].pop(0)                
        else:
            if len(self._pos_pos[error_type]) > 0:
                pos = rd.choice(self._pos_pos[error_type])                

        # if for pos == idx (actually, this should never happen), randomly 
        # sample another positive from the set of soft positives (if available)
        if pos == idx:
            if len(self._pos_pos[error_type]) > 0:
                pos = rd.choice(self._pos_pos[error_type]) 
            
        pos = self._dataset[pos[0]]

        if len(self._hard_neg[error_type]) > 0:
            if len(self._soft_neg[error_type]) > 0:
                r = rd.choice([0, 1])                
                neg = self._hard_neg[error_type].pop(0) if r == 0 else\
                    self._soft_neg[error_type].pop(0)
            else:
                neg = self._hard_neg[error_type].pop(0)
        else:
            if len(self._soft_neg[error_type]) > 0:
                neg = self._soft_neg[error_type].pop(0)
            else:
                _class = rd.choice([i for i in valid_classes 
                    if i != error_type])            
                neg = rd.sample(self._data_class_based[_class], 
                    self._num_negatives)           
        neg = self._dataset[neg[0]]
        
        return {"pos_context": pos["context"],
                "pos_idx": pos["index"],
                "pos_error_utts": pos["error_utterance"]["text"],
                "pos_summary": pos["error_desc_unlabeled"],
                "neg_context": [neg["context"]],
                "neg_summary": [neg["error_desc_unlabeled"]],
                "neg_error_utts": [neg["error_utterance"]["text"]],
                "neg_labels": [neg["error_type"]],
                "target": sample["error_type"],
                "idx": sample["index"],
                "error_utterance": sample["error_utterance"]["text"],
                "context": sample["context"],
                "error_desc_unlabeled": sample["error_desc_unlabeled"]}
       
    
    def __len__(self):
        return len(self._dataset)
    
        
    @property
    def dataset(self):
        return self._dataset
