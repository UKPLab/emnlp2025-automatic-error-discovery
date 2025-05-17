'''
    This script contains the code for error definition generation.
'''
import re
import torch
import random as rd

from nnk_means import NNKMeans
from argparse import Namespace
from arguments import Arguments
from datasets import load_dataset, Dataset as HfDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

from dataset import BaseDataset
from error_definitions import ABC_EVAL, CLUSTER_INTERPRETATION, SODA_EVAL, FEDI
from models import Loop, SimpleContrastiveEncoder, KNNContrastive,\
    DialogueSummaryEncoder

class ErrorTypeGeneration():
 
    def __init__(self, args:Namespace):
        
        self._args = args

        model = self._args.model.lower()

        self._dataset =\
            load_dataset(self._args.dataset, 
                token=self._args.token)["test"]             
        self._classes = list(set(self._dataset["label"])) 
        self._k_classes =\
            [c for c in self._classes if c not in self._args.n_labels]

        if model == "loop":
            self._model = Loop(len(self._classes), device=self._args.device)
        elif model == "syncid":
            self._model = SimpleContrastiveEncoder(len(self._classes), 
                device=self._args.device)
        elif model == "knn_contrastive":
            self._model = KNNContrastive(len(self._classes), 
                device=self._args.device, positives=self._args.positives)
        elif model == "seeed":            
            self._model = DialogueSummaryEncoder(len(self._classes), 
                device=self._args.device)
        else:
            raise Exception("Model class unknown")

        print("load pretrained model...")
        original_parameters = self._model.parameters()
        self._model = self._model.load_model(self._args.pretrained)
        loaded = self._check_loading_pretrained(original_parameters, 
            self._model.parameters())
        print(f"loading pretrained model successful: {loaded}")
        if not loaded:
            raise RuntimeError("loaded and pretrained have same weights.")
        
        #
        # load generation model
        #

        generation_model = "<path_to_llama-3.1>"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self._gen_model = LlamaForCausalLM.from_pretrained(generation_model, 
            quantization_config=bnb_config).to("cuda")

        self._tokenizer = AutoTokenizer.from_pretrained(generation_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
                
    def _check_loading_pretrained(self, params_1:torch.Tensor, 
        params_2:torch.Tensor) -> bool:
        for p1, p2 in zip(params_1, params_2):
            if p1.data.ne(p2.data).sum() > 0:
                return True
            return False
        
        
    def _get_context(self, error_utts:list[str], contexts:list[str]):
        sequences = []
        for eu, context in zip(error_utts, contexts):
            context = context.replace("\nuser: ", " [SEP] ")
            context = context.replace("\nsystem: ", " [SEP] ")
            context = context.replace("user: ", "[CLS] ")
            context = context.replace("system: ", "[CLS] ")
            context = context + " [SEP] " + eu + " [SEP]"
            sequences.append(context)
        return sequences
    

    def _get_embeddings(self, samples:HfDataset) -> torch.Tensor:
        
        # set the model into eval mode (no gradients)
        self._model = self._model.eval()

        dataloader = DataLoader(BaseDataset(samples, self._args, 
            n_labels=[100], shuffle=False), batch_size=self._args.batch_size, 
            drop_last=True)
        embeds =\
            torch.empty((0, self._model.hidden_size)).to(self._args.device)
        
        labels, text_labels = [], []
        with torch.no_grad():            
            for batch in dataloader:

                labels += [e.item() for e in batch["error_type"]]
                text_labels += batch["error_type_str"] 

                sequences = self._get_context(batch["error_utterance"]["text"], 
                        batch["context"])
                summaries = batch["error_desc_unlabeled"]

                output = self._model.tokenize_and_forward(sequences, 
                    summaries)["embeddings"]
                
                embeds = torch.cat((embeds, output))
        
        return embeds, labels, text_labels
       
    
    def get_cluster_centers(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Calculates the average embedding for each class.

        Args:
            X (torch.tensor): The input embeddings.
            Y (np.array): The predicted labels.

        Returns:
            dict: A dictionary where keys are class labels and values are
                  average embeddings (NumPy arrays).
        """

        class_centers = {}

        for _class_label in torch.unique(Y):
            class_label = _class_label.item()

            class_indices = torch.where(Y == class_label)[0]
            class_embeddings = X[class_indices]

            if len(class_embeddings) > 0:
                avg_embedding = torch.mean(class_embeddings, axis=0)
                class_centers[class_label] = avg_embedding.unsqueeze(0)
            else:
                class_centers[class_label] = None # Or np.array([]) if needed.

        return class_centers
    

    def run(self):

        #
        # get embeddings
        #
        embeds, labels, label_texts = self._get_embeddings(self._dataset)

        print("label-label text assignment:")
        for l, lt in zip(set(labels), set(label_texts)):
            print(f"{l}: {lt}")

        #
        # run NNKMeans
        #
        nnkmeans = NNKMeans(n_classes = len(self._classes))\
                .fit(embeds, torch.tensor(labels).to(self._args.device))
        preds_all, _ = nnkmeans.predict(embeds)
        preds_all = torch.Tensor(preds_all).to(self._args.device)

        #
        # get cluster centers for n_labels
        #
        cluster_centers = {k:v for k, v in self.get_cluster_centers(embeds, 
            preds_all).items() if k in self._args.n_labels}
        
        #
        # get the indices of the top-k embeddings located close to the cluster 
        # centers
        #      
        top_k_indices = {}  
        for k, center in cluster_centers.items():
            
            indices = torch.where(preds_all == k)[0]
            _embeds = embeds[indices]

            _embeds = torch.nn.functional.normalize(_embeds, p=2, dim=1) 
            center = torch.nn.functional.normalize(center, p=2, dim=1)

            sim = F.cosine_similarity(_embeds, 
                center.expand(_embeds.shape[0], -1))

            if sim.shape[0] < 10:
                top_k_sims, _top_k_indices = torch.topk(sim, sim.shape[0])
            else:
                top_k_sims, _top_k_indices = torch.topk(sim, 10)

            top_k_indices[int(k)] = _top_k_indices.tolist()

        prompts = {}
        for n_label in self._classes:

            indices = [i for i, l in enumerate(self._dataset["label"]) 
                if l == n_label]
            samples = self._dataset.select(indices)[:3]

            #
            # build context and summaries
            #
            contexts_summaries = []            
            for i, (context, error_utterance, description) in enumerate(zip(samples["context"], samples["error_utterance"], samples["unlabeled_descriptions"])):                   
                     
                context = [f"{ut['agent']}: {ut['text']}" for ut in context]
                context = "\n".join(context)
                context += f"{error_utterance['agent']}: {error_utterance['text']}"
                context = re.sub(r"{{\w+}}", "", context)
                context_summary = f"Dialogue Context {i}:\n{context}\n\nSummary {i}:\n{description}"
                contexts_summaries.append(context_summary)
            contexts_summaries = "\n\n".join(contexts_summaries)
            

            #
            # sample three error definitions
            #   
            error_idxs = rd.sample(self._k_classes, 3)
            error_types = {k: v for i, (k, v) in enumerate(SODA_EVAL.items())
                if i in error_idxs}

            errors = []
            for et, ed in error_types.items():
                error = f"Name: {et.replace('_', ' ')}\nDescription: {ed['description']}"
                errors.append(error)
            errors = "\n\n".join(errors)

            prompts[n_label] =\
                CLUSTER_INTERPRETATION.replace("{{CONTEXT_SUMMARIES}}", contexts_summaries).replace("{{EXAMPLES}}", errors)

        #
        # create prompts
        #
        
        prompts = {}
        for label, indices in top_k_indices.items():

            print(f"samples for {label}")

            samples = self._dataset.select(indices)

            print(f"samples target classes: {samples['label_text']}")

            #
            # build context and summaries
            #
            contexts_summaries = []            
            for i, s in enumerate(samples):            
                context = [f"{ut['agent']}: {ut['text']}" for ut in s["context"]]
                context = "\n".join(context)
                context += f"{s['error_utterance']['agent']}: {s['error_utterance']['text']}"
                context = re.sub(r"{{\w+}}", "", context)
                context_summary = f"Dialogue Context {i}:\n{context}\n\nSummary {i}:\n{s['unlabeled_descriptions']}"
                contexts_summaries.append(context_summary)
            contexts_summaries = "\n\n".join(contexts_summaries)

            #
            # sample three error definitions
            #   
            error_idxs = rd.sample(self._k_classes, 3)
            error_types = {k: v for i, (k, v) in enumerate(SODA_EVAL.items())
                if i in error_idxs}

            errors = []
            for et, ed in error_types.items():
                error = f"Name: {et.replace('_', ' ')}\nDescription: {ed['description']}"
                errors.append(error)
            errors = "\n\n".join(errors)

            prompts[label] =\
                CLUSTER_INTERPRETATION.replace("{{CONTEXT_SUMMARIES}}", contexts_summaries).replace("{{EXAMPLES}}", errors)

        print(prompts)
        
                
        #
        # generate new error type names and definitions
        #

        with torch.no_grad():
            for label, prompt in prompts.items():                
                tokenized =\
                    self._tokenizer(prompt, return_attention_mask = True, 
                    return_tensors="pt")
                model_input = tokenized.to("cuda")
                output = self._gen_model.generate(**model_input, 
                    max_new_tokens=500, do_sample=True, top_p=0.92, top_k=0, 
                    early_stopping=True)[0]
                decode =\
                    self._tokenizer.decode(output, skip_special_tokens=False)
                decode =\
                    decode.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
                decode = decode.split("<|eot_id|>")[0]
                print(f"prompt: {prompt}\n\nlabel: {label}\n\ngeneration: {decode}")


if __name__ == "__main__":

    print("\nCluster Interpretation")

    parser = Arguments()
    parser.parser.add_argument("--model", 
        help="The model class to use", 
        type=str,
        choices=["loop", "syncid", "knn_contrastive", "seeed"],
        required=True)
    etg = ErrorTypeGeneration(parser.parse_args())
    etg.run()