'''
    This script contains the model classes for the approaches implemented in this project.
'''
import os
import gc
import re
import torch
import numpy as np
import time

from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import Union
from utility import print_model_footprint

class BaseModel(torch.nn.Module):
    """BaseModel class. Provides access to a foundation model, such as BERT.
    """

    def __init__(self, foundation_model:str, device:str="cuda"):         
        super(BaseModel, self).__init__()        
        self._device = device
        self._foundation_model =\
            AutoModelForMaskedLM.from_pretrained(foundation_model) if foundation_model else None
        self._tokenizer = AutoTokenizer.from_pretrained(foundation_model, 
            clean_up_tokenization_spaces=False) if foundation_model else None
        self.to(self._device)


    def tokenize(self, sequences:Union[list[str], str]) -> dict[torch.Tensor]:
        """Returns a dict with the fields input_ids, token_type_ids and 
            attention_mask.

        Args:
            sequences (list[str]): Input sequences (batch)

        Returns:
            dict: Input ids, token type ids and attention mask
        """        

        _max_length = min(max([len(s) for s in sequences]), 512)\
            if isinstance(sequences, list) else 512
        tokenized = self.tokenizer(sequences, truncation=True, padding=True, 
            max_length=_max_length, return_attention_mask=True,return_tensors="pt").to(self._device)
        
        return tokenized
    

    def forward(self, input_ids = None, token_type_ids = None, 
        attention_mask=None, labels=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Passes the passed input ids through the foundation model and returns the averaged hidden states of the last layer.

        Args:
            input_ids (_type_, optional): Input Ids. Defaults to None.
            token_type_ids (_type_, optional): Token type ids. Defaults to None.
            attention_mask (_type_, optional): Attention mask. Defaults to None.
            labels (_type_, optional): Labels for masked language modeling). 
                Defaults to None.
        Returns:
            torch.Tensor: Averaged hidden states of the last layer.
            torch.Tensor: Masked language modeling loss.
        """
        
        input_ids = input_ids.to(self._device)
        token_type_ids = token_type_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        if labels is not None:
            labels = labels.to(self._device)

        outputs = self._foundation_model(input_ids=input_ids, 
            token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        loss = outputs.loss if "loss" in outputs else None
        avg_hidden = outputs.hidden_states[-1][:, 1:, :].mean(dim=1)

        return avg_hidden, loss
    

    def tokenize_and_forward(self, sequences:list[str]) -> dict[torch.Tensor]:
        """Tokenizes the passed list of sequences and passes it through the model.

        Args:
            sequences (list[str]): List of string sequences.

        Returns:
            dict[torch.Tensor]: Output (see return types of HF implementation   
                for the passed foundation model.)
        """    
        _tokenized = self.tokenize(sequences)
        _output = self(**_tokenized)

        return _output
    

    def save_model(self, name:str, epoch:int, acc:float, dir:str):
        """Saves the model's state dict as .pt to the passed directory. File name format: {dir}/{name}_{epoch}_{acc}.pt

        Args:
            epoch (int): The number of the past training epoch.
            acc (float): The accuracy of the past evaluation.
            dir (str): The path of the directory where to save the model.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Get existing models
        models = [f for f in os.listdir(dir) if f.startswith(name) 
            and f.endswith(".pt")]

        print(f"dir: {dir}")
        print(f"models: {models}")

        # max accs
        max_accs = max([float(f.split(".")[0].split("_")[-1]) for f in models])\
            if models else 0.0
        
        print(f"max_accs: {max_accs}")
        
        # Save the new model if no model exists or its acc is higher
        if not models or acc > max_accs:
            torch.save(self, os.path.join(dir, f"{name}_{epoch}_{acc}.pt"))

        # Keep only the 5 best models
        if len(models) >= 5:
            # Sort by accuracy (descending)
            _sorted = sorted(models, 
                key=lambda x: float(x.split("_")[-1].split(".")[1]), reverse=True)
      
            # Delete the models with the lowest acc
            for model in _sorted[5:]:
                os.remove(os.path.join(dir, model))
        
        
    def load_model(self, filename:str):
        return torch.load(filename, weights_only=False)


    @property
    def tokenizer(self):
        return self._tokenizer
    

    @property
    def foundation_model(self):
        return self

    
    @property
    def hidden_size(self):
        return self._foundation_model.config.hidden_size


    @property
    def device(self):
        return self._device


class Loop(BaseModel):

    def __init__(self, num_labels:int, model_path:str="bert-base-uncased",
        device:str="cuda"):
        super(Loop, self).__init__(model_path, device)

        self._device = device
        self._num_labels = num_labels

        self._classifier =\
            torch.nn.Linear(self.hidden_size, 
                self._num_labels)
        self._dropout = torch.nn.Dropout(0.1)
        
        self.to(device)


    def forward(self, input_ids = None, token_type_ids = None, 
        attention_mask = None, labels = None) -> dict[torch.Tensor]:

        avg_hidden, mlm_loss = super().forward(input_ids=input_ids, 
            token_type_ids=token_type_ids, attention_mask=attention_mask,
            labels=labels)
        
        # logits for predicting the error type
        avg_hidden = self._dropout(avg_hidden)
        logits = self._classifier(avg_hidden)

        return {"avg_hidden": avg_hidden, 
                "embeddings": avg_hidden,                
                "logits": logits,
                "mlm_loss": mlm_loss}

    def tokenize_and_forward(self, sequences:list[str], mlm:bool=False, 
        mlm_prob:float=0.15, random_token_replace:bool=False)\
        -> dict[torch.Tensor]:

        if mlm:
            # we reuse the code from LOOP here (https://github.com/Lackel/LOOP/blob/main/utils/tools.py), last accessed 12.12.2024
            _tokenized = self.tokenize(sequences)
            inputs = _tokenized["input_ids"].cpu()
            labels = inputs.clone()

            # create labels
            labels = _tokenized["input_ids"].cpu().clone()
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, 
                dtype=torch.bool)
            probability_matrix = torch.full(labels.shape, mlm_prob)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            probability_matrix[torch.where(inputs==0)] = 0.0
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # only compute loss on masked tokens
            _tokenized["labels"] = labels 

            # mask tokens in the input sequence
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced =\
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] =\
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            
            # 10% of the time, we replace masked input tokens with random word.
            # The rest of the time, we keep the masked input tokens unchanged.
            indices_random =\
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, 
                dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
            _tokenized["input_ids"] = inputs

            if random_token_replace:
                # this is taken from https://github.com/Lackel/LOOP/blob/main/utils/tools.py#L100 (last accessed 17.12.24). Originally, the 
                # solve this separately, although this doesn't make much sense, 
                # as the output of random_token_replace is only required for 
                # the model pass and is an add-on to mlm used in step 2.
                mask_id = self.tokenizer.\
                    convert_tokens_to_ids(self.tokenizer.mask_token)
                random_words = torch.randint(len(self.tokenizer), labels.shape, 
                    dtype=torch.long)
                indices_replaced =\
                    torch.where(_tokenized["input_ids"] == mask_id)
                _tokenized["input_ids"][indices_replaced] =\
                    random_words[indices_replaced]
            
        else:    
            _tokenized = self.tokenize(sequences)            
        
        _output = self(**_tokenized)

        return _output


class DialogueSummaryEncoder(BaseModel):

    def __init__(self, num_labels:int, model_path:str="bert-base-uncased", 
        hidden_size:int=768, dropout_rate:float=0.2, device:str="cuda"):
        
        super(DialogueSummaryEncoder, self).__init__(model_path, device)

        self._summary_encoder = BaseModel(model_path, device)

        self._concat_layer =\
            torch.nn.Linear(self.hidden_size + 
            self._summary_encoder.hidden_size, hidden_size)
        
        self._classifier =\
            torch.nn.Linear(self.hidden_size, num_labels)
        
        #self._concat_dropout = torch.nn.Dropout(dropout_rate)
        #self._relu = torch.nn.ReLU()

        #self._classifier = torch.nn.Sequential(
        #    torch.nn.Linear(hidden_size, hidden_size // 2),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(dropout_rate),
        #    torch.nn.Linear(hidden_size // 2, num_labels)
        #)

        self.to(device)

    
    def forward(self, dialogue_context_input:dict, summary_input:dict)\
        -> dict[torch.Tensor]:

        dialogue_hidden, _ = super().forward(**dialogue_context_input)
        summary_hidden, _ = self._summary_encoder(**summary_input)

        # concatenation
        embeddings = torch.cat((dialogue_hidden, summary_hidden), dim=-1)
        embeddings = self._concat_layer(embeddings)
        #relu_embeddings = self._relu(embeddings)
        #dropout_embeddings = self._concat_dropout(relu_embeddings)
        
        # classification
        #logits = self._classifier(dropout_embeddings)
        logits = self._classifier(embeddings)
        
        return {"embeddings": embeddings,
                "logits": logits}


    def tokenize_and_forward(self, dialogue_context:list[str], 
        summaries:list[str]) -> dict[torch.Tensor]:
        
        _dialogue_context = self.tokenize(dialogue_context)
        _summary = self.tokenize(summaries)       

        _output = self(_dialogue_context, _summary)

        return _output


class SimpleContrastiveEncoder(BaseModel):
    
    def __init__(self, num_labels:int, model_path:str="bert-base-uncased", 
        projection:int=128, device:str="cuda"):
        """as SynCID does not provide their code, we use the impplementation of [USNID](https://github.com/thuiar/TEXTOIR/blob/main/open_intent_discovery/methods/semi_supervised/USNID/pretrain.py) as an orientation (since SynCID refers to them).

        Args:
            num_labels (int): The number of classes to distinguish
            model_path (str, optional): The model path. Defaults to "bert-base-uncased".
            projection (int, optional): The dimension for projection before applying the contrastive loss. Defaults to 128.
        """
        super(SimpleContrastiveEncoder, self).__init__(model_path, device)

        self._device = device
        self._num_labels = num_labels
        self._projection_dim = projection
        
        self._projection =\
            torch.nn.Linear(self.hidden_size, 
                self._projection_dim)
        self._classifier =\
            torch.nn.Linear(self.hidden_size, 
                self._num_labels)
        
        self.to(device)
    

    def forward(self, input_ids = None, token_type_ids = None, 
        attention_mask=None, labels=None) -> dict[torch.Tensor]:

        input_ids = input_ids.to(self._device)
        token_type_ids = token_type_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        avg_hidden, mlm_loss = super().forward(input_ids=input_ids, 
            token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        
        # logits for predicting the error type
        logits = self._classifier(avg_hidden)
        # projection for calculating the contrastive loss
        projections = self._projection(avg_hidden)

        return {"avg_hidden": avg_hidden, 
                "projections": projections, 
                "logits": logits,
                "mlm_loss": mlm_loss}
    

    def tokenize_and_forward(self, sequences:list[str], mlm:bool=False, 
        mlm_prob:float=0.15) -> dict[torch.Tensor]:

        if mlm:
            # we reuse the code from LOOP here (https://github.com/Lackel/LOOP/blob/main/utils/tools.py), last accessed 12.12.2024
            _tokenized = self.tokenize(sequences)
            inputs = _tokenized["input_ids"].cpu()
            labels = inputs.clone()

            # create labels
            labels = _tokenized["input_ids"].cpu().clone()
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, 
                dtype=torch.bool)
            probability_matrix = torch.full(labels.shape, mlm_prob)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            probability_matrix[torch.where(inputs==0)] = 0.0
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # only compute loss on masked tokens
            _tokenized["labels"] = labels 

            # mask tokens in the input sequence
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced =\
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] =\
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            
            # 10% of the time, we replace masked input tokens with random word.
            # The rest of the time, we keep the masked input tokens unchanged.
            indices_random =\
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, 
                dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
            _tokenized["input_ids"] = inputs
        else:    
            _tokenized = self.tokenize(sequences)       

        _output = self(**_tokenized)

        return _output
    
    @property
    def projection_dim(self):
        return self._projection_dim


class KNNContrastiveProjectionLayer(torch.nn.Module):

    def __init__(self, hidden_size:int=768):
        super(KNNContrastiveProjectionLayer, self).__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.1) # bert default
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, feature:torch.Tensor):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class KNNContrastive(BaseModel):
    
    def __init__(self, num_labels:int, model_path:str="bert-base-uncased",
        device:str="cuda", positives:int=3):
        """
        Args:
            num_labels (int): The number of classes to distinguish
            model_path (str, optional): The model path. Defaults to "bert-base-uncased".
            projection (int, optional): The dimension for projection before applying the contrastive loss. Defaults to 128.
        """
        super(KNNContrastive, self).__init__(model_path, device)

        self._device = device
        self._num_labels = num_labels

        # parameter for updating momentum encoder
        self._momentum_update = 0.999

        # queue size
        self._queue_size = 7500

        # topk for knn
        self._topk = 25

        # weighting factor for calculating the contrastive logits
        self._contrastive_weighting = 0.3

        # positive samples per label
        self._update_num = positives

        # k in their code base
        self._momentum_encoder =\
            AutoModelForMaskedLM.from_pretrained(model_path)
        
        self._projection_foundation_model =\
            KNNContrastiveProjectionLayer(self.hidden_size)
        self._projection_momentum_encoder =\
            KNNContrastiveProjectionLayer(self.hidden_size)
        
        self._classifier =\
            torch.nn.Linear(self.hidden_size, 
                self._num_labels)

        # register buffer adds tensors to the state dict (saved and loaded to 
        # disk when saving / loading the model), that are not considered as 
        # model parameters. Buffers can be accessed as attributes using their 
        # given names.
        self.register_buffer("label_queue", 
            torch.randint(0, self._num_labels, [self._queue_size]))
        self.register_buffer("feature_queue", 
            torch.randn(self._queue_size, self.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                
        self.to(device)


    def _l2norm(self, x:torch.Tensor) -> torch.Tensor:
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x
    

    def momentum(self, positives:dict[torch.Tensor], labels:torch.Tensor):

        with torch.no_grad():
            
            # update momentum encoder
            for param_q, param_k in zip(self.foundation_model.parameters(), 
                self._momentum_encoder.parameters()):
                param_k.data =\
                    param_k.data * self._momentum_update +\
                        param_q.data * (1. - self._momentum_update)
            for param_q, param_k in\
                zip(self._projection_foundation_model.parameters(), 
                    self._projection_momentum_encoder.parameters()):
                param_k.data = param_k.data * self._momentum_update +\
                    param_q.data * (1. - self._momentum_update)

            # reshape dict
            for k, v in positives.items():
                shape = v.shape
                positives[k] = v.view([-1, shape[-1]])

            output = self._momentum_encoder(**positives, 
                output_hidden_states=True)
            output = output.hidden_states[-1][:, 1:, :].mean(dim=1)

            projection = self._projection_momentum_encoder(output)
            normalized = self._l2norm(projection)

            # enqueueing and dequeueing labels
            labels = labels.view(-1)
            labels = labels.unsqueeze(-1)
            labels = labels.repeat([1, self._update_num])
            labels = labels.view(-1)

            ptr = int(self.queue_ptr)
            batch_size = normalized.shape[0]
            if ptr + batch_size > self._queue_size:
                batch_size = self._queue_size - ptr
                keys = normalized[:batch_size]
                labels = labels[:batch_size]
            else:
                keys = normalized


            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = labels

            ptr = (ptr + batch_size) % self._queue_size
            self.queue_ptr[0] = ptr

    
    def _knn_contrastive_logits(self, projection:torch.Tensor, 
        error_types:torch.Tensor):
        """This is the original code for calculating the contrastive
           logits (considering KNN) from https://github.com/zyh190507/KnnContrastiveForOOD/blob/main/model.py#L332

        Args:
            projection (torch.Tensor): _description_
            error_types (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """

        label_queue = self.label_queue.clone().detach()        # queue_size
        feature_queue = self.feature_queue.clone().detach()    # queue_size * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * queue_size
        batch_size = error_types.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * queue_size * hidden_size

        # 2.caluate sim
        cos_sim = torch.einsum('nc,nkc->nk', [projection, tmp_feature_queue])

        # 3. get index of postive and negative 
        tmp_label = error_types.unsqueeze(1).detach()
        tmp_label = tmp_label.repeat([1, self._queue_size])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index # ~ turns pos_mask_index into the opposite

        # masked_select returns a new 1-D tensor which indexes the input tensor 
        # according to the boolean mask mask which is a BoolTensor. The shapes 
        # of the mask tensor and the input tensor don't need to match, but they 
        # must be broadcastable.
        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)        
        pos_sample_top_k = pos_sample[:, 0:self._topk]
        
        pos_sample = pos_sample_top_k        
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self._topk)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])

        # 6. calculate logits
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self._contrastive_weighting

        return logits_con    

    def forward(self, sequences:dict[torch.Tensor], 
        positives:dict[torch.Tensor], error_types:torch.Tensor)\
        -> dict[torch.Tensor]:

        self.momentum(positives, error_types)

        avg_hidden, _ =\
            super().forward(**sequences)        
        projections = self._projection_foundation_model(avg_hidden)        
        normalized = self._l2norm(projections)
        logits_cls = self._classifier(projections)
        logits_contrastive = self._knn_contrastive_logits(normalized,
            error_types.to(self._device))

        return {"avg_hidden": avg_hidden, 
                "projections": projections, 
                "logits_cls": logits_cls,
                "logits_contrastive": logits_contrastive}


    def tokenize_and_forward(self, sequences:list[str], 
        error_types:list[int]=None, positives:list[str]=None)\
        -> dict[torch.Tensor]:
        """_summary_

        Args:
            sequences (list[str]): _description_
            error_types (list[int]): The labels for the passed sequences
            positives (list[list[str]]): _description_
            negatives (list[list[str]]): _description_

        Returns:
            dict[torch.Tensor]: _description_
        """
        # sind positives and negatives wirklich listen mehrerer samples, oder jeweils nur ein sample?
    
        if error_types is None and positives is None:
            return self.forward_test(sequences)
        else:
            _tokenized_sequences = self.tokenize(sequences)
            _tokenized_positives = self.tokenize(positives)
            _output = self(_tokenized_sequences, _tokenized_positives, 
                error_types)
            return _output
    
    @property
    def projection_dim(self):
        return self._projection_dim
    

class LLMFinetuning(torch.nn.Module):

    def __init__(self, model_path:str="microsoft/Phi-4-mini-instruct", 
        device:str="cuda"):
        super(LLMFinetuning, self).__init__()   

        print("Model Path: " + model_path)

        self._device = device
        self._model_path = model_path

        peft_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
            "modules_to_save": None,
        }        
        peft_conf = LoraConfig(**peft_config)

        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, 
            trust_remote_code=True)#, quantization_config=bnb_config)
        self._model.gradient_checkpointing_enable()
        self._model = prepare_model_for_kbit_training(self._model)
        self._model = get_peft_model(self._model, peft_conf).to(self._device)

        self._model.print_trainable_parameters()
        print_model_footprint(self._model)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._tokenizer.model_max_length = 2048
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        
        self._model.config.pad_token_id = self._tokenizer.pad_token_id # Also set in the model config

    def forward(self, tokenized:torch.Tensor):

        # all labels set to -100 are ignored; the loss is only
        # calculated for the other tokens.        
        labels = tokenized["input_ids"].clone()
        labels[tokenized["input_ids"][:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self._model(
            input_ids=tokenized["input_ids"].squeeze(),
            attention_mask=tokenized["attention_mask"].squeeze(),
            labels=labels.squeeze(),
            return_dict=True,
        )

        return outputs


    def tokenize_and_forward(self, batch:dict) -> torch.Tensor:
        
        tokenized = self._tokenizer(batch["prompt"], truncation=True, 
            padding=True, return_tensors="pt", 
            max_length=1580, return_attention_mask=True).to(self._device)    
        return self.forward(tokenized)["loss"]
    

    def tokenize_and_generate(self, batch:dict) -> Union[str, str]:

        tokenized = self._tokenizer(batch["prompt"], truncation=True, 
            padding=True, return_tensors="pt").to(self._device)
        _output =\
            self._model.generate(**tokenized, max_new_tokens=200, early_stopping=True, do_sample=False, temperature=0.0)
        
        decisions, reasonings = [], []
        for generation in _output:                  
            decoded = self._tokenizer.decode(generation, 
                skip_special_tokens=False).split("<|assistant|>")
            
            try:
                if len(decoded) > 1: 
                    decoded = decoded[1]            
                
                    decision = re.findall(r"Decision: \[(.*?)\]", 
                        decoded)            
                    decisions.append("" if len(decision) < 1 else 
                        decision[0].lower())                
                    reasoning = re.split(r"Reasoning:", decoded)   
                    if len(reasoning) > 1:
                        reasoning = re.sub(r"<\|endoftext\|>", "", 
                            reasoning[1])         
                        reasonings.append(reasoning)
                    else:
                        reasonings.append("")
                else:
                    decisions.append("")
                    reasonings.append("")
            except IndexError as e:
                    decisions.append("")
                    reasonings.append("")
        
        return decisions, reasonings


    def save(self, path:str, experiment_name:str, epoch: int, score: float):
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'{experiment_name}_{epoch}_{score}'        
                
        self._model.save_pretrained(os.path.join(path, model_name, "lora"), 
            from_pt=True)
        self._tokenizer.save_pretrained(os.path.join(path, model_name, "lora")) 
        
        # reload the model and merge with lora weights
        # (for saving the complete model as finetuned model)
        print("loading the model for merging")
        time.sleep(5)
        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )        
        model = PeftModel.from_pretrained(
            base_model, os.path.join(path, model_name, "lora")
        )
        model = model.merge_and_unload()

        self.tokenizer.save_pretrained(
            os.path.join(path, model_name, "finetuned")
        )
        model.save_pretrained(os.path.join(path, model_name, "finetuned"))
                
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
