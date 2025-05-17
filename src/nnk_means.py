'''
    This script contains the code for NNK-Means, adapted from the reference code provided in GitHub (https://github.com/STAC-USC/NNK_Means) to be compatible with the scikit-learn API.
'''
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import kmeans_plusplus
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.cluster import kmeans_plusplus


def approximate_nnk(AtA, b, x_init, x_tol=1e-6, num_iter=100, eta=None):
    if eta is None:
        values, _ = torch.max(torch.linalg.eigvalsh(AtA).abs(), 1, keepdim=True)
        eta = 1. / values.unsqueeze(2)

    b = b.unsqueeze(2)
    x_opt = x_init.unsqueeze(2)
    for t in range(num_iter):
        grad = b.sub(torch.bmm(AtA, x_opt))
        x_opt = x_opt.add(eta * grad).clamp(min=torch.cuda.FloatTensor([0.]), max=b)

    error = 1 - 2*torch.sum(x_opt*b.sub(0.5*torch.bmm(AtA, x_opt)), dim=1)
    return x_opt.squeeze(), error.squeeze()


class NNKMeans(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters=100, n_nonzero_coefs=50, momentum=1.0, 
                 n_classes=0, influence_tol=1e-4, optim_itr=100, optim_lr=None, 
                 optim_tol=1e-6, use_error_based_buffer=True, use_residual_update=False,
                 max_iter=8, tol=1e-4, random_state=None, verbose=0, device="cuda"):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_nonzero_coefs = n_nonzero_coefs
        self.momentum = momentum
        self.n_classes = n_classes
        self.influence_tol = influence_tol
        self.optim_itr = optim_itr
        self.optim_lr = optim_lr
        self.optim_tol = optim_tol
        self.use_error_based_buffer = use_error_based_buffer
        self.use_residual_update = use_residual_update
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        self.dictionary_atoms = None
        self.dictionary_atoms_norm = None
        self.atom_labels = None
        self.data_cache = None
        self.label_cache = None
        self.influence_cache = None
        self.dictionary_data_buffer = None
        self.dictionary_label_buffer = None
        self.associated_error = None

    def fit(self, X:torch.tensor, Y:torch.tensor=None):        
        random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]

        if self.n_classes is not None and Y is None:
            raise ValueError("Labels y must be provided if n_classes > 0")

        _, initial_indices = kmeans_plusplus(X.cpu().numpy(), self.n_clusters, random_state=random_state)
        #initial_data = torch.from_numpy(X[initial_indices]).float().cuda()
        initial_data = X[initial_indices]

        if Y is not None:
          #initial_labels = torch.from_numpy(Y[initial_indices]).long().cuda()
          initial_labels = Y[initial_indices]
        else:
          initial_labels = None

        self.initialize_dictionary(initial_data, initial_labels)  

        batch_size = 64  # Set your desired batch size
        for _ in range(self.max_iter):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = Y[i:i+batch_size] if Y is not None else None
                _, _, _, W = self.forward(X_batch, y_batch, get_codes=True)
            self.update_dict()

        return self
    
    def transform(self, X):
        X = self._check_params(X)
        _, _, _, W = self.forward(torch.from_numpy(X).float().cuda(), get_codes=True)
        return W.cpu().numpy().T

    def predict(self, X:torch.tensor):    
    
        n_samples = X.shape[0]
        predicted_labels = torch.zeros(n_samples, dtype=torch.long).cuda() #Initialize with zeros
        cluster_centers = None  # Initialize cluster_centers

        batch_size = 256  # Use batching for prediction as well
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]
            x_opt, indices, _ = self._sparse_code(X_batch) #Only sparse code
            if self.n_classes > 0:
                label_interpolated = torch.bmm(x_opt.unsqueeze(1), self.atom_labels[indices]).squeeze(1)
                predicted_labels_batch = torch.argmax(label_interpolated, dim=1)
            else:
                interpolated = torch.bmm(x_opt.unsqueeze(1), self.dictionary_atoms[indices]).squeeze(1)
                predicted_labels_batch = interpolated #Regression
            predicted_labels[i:i+batch_size] = predicted_labels_batch

            if cluster_centers is None:
                  cluster_centers = self.dictionary_atoms[indices].clone()
            else:
                cluster_centers = torch.cat((cluster_centers, 
                    self.dictionary_atoms[indices]))

        return predicted_labels.cpu().numpy(), cluster_centers.cpu().numpy()
    

    @torch.no_grad()
    def _process_data(self, data):
        return nn.functional.normalize(data, dim=1)
    
    def _process_labels(self, labels):
        if self.n_classes > 0:
            return nn.functional.one_hot(labels, self.n_classes).float()
        return labels.float()
    
    @torch.no_grad()
    def initialize_dictionary(self, initial_data, initial_labels=None):
        self.dictionary_atoms = initial_data
        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)
        if self.n_classes is not None:
            self.atom_labels = self._process_labels(initial_labels)
        
        self._set_cache()
    
    def _set_cache(self):
        self.dictionary_data_buffer = torch.clone(self.dictionary_atoms)
        self.data_cache = torch.zeros_like(self.dictionary_atoms)
        
        self.associated_error = torch.zeros(self.n_clusters).cuda()
        
        if self.n_classes is not None:
            self.dictionary_label_buffer = torch.clone(self.atom_labels).to(self.device)
            self.label_cache = torch.zeros_like(self.atom_labels).to(self.device)
            
        self.influence_cache = torch.zeros((self.n_clusters, self.n_clusters), dtype=torch.float32).cuda()
    
    def reset_cache(self):
        self._set_cache()

    @torch.no_grad()
    def _update_cache(self, batch_W, batch_data, batch_label):
        self.data_cache = self.data_cache + self.momentum * torch.sparse.mm(batch_W, batch_data)
        self.influence_cache = self.influence_cache + self.momentum * torch.sparse.mm(batch_W, batch_W.t())
        if self.n_classes is not None:
            self.label_cache = self.label_cache + self.momentum * torch.sparse.mm(batch_W, batch_label)

    @torch.no_grad()
    def _update_buffer(self, batch_data, batch_label=None, error=1):
        indices = torch.arange(self.n_clusters) # set default to maintain the data buffer
        if self.use_error_based_buffer:
            if error.min() > self.associated_error.min():
                self.associated_error, indices = torch.topk(torch.cat((self.associated_error, error)), self.n_clusters, sorted=True)
        
        else: # Randomly substitute elements in buffer with elements from batch_data
            indices = torch.randint(0, self.n_clusters + batch_data.shape[0], size=(self.n_clusters,), 
                                    device=self.dictionary_data_buffer.device)
        
        temp_data_buffer = torch.cat((self.dictionary_data_buffer, batch_data))
        self.dictionary_data_buffer = temp_data_buffer[indices]
        
        if self.n_classes is not None:
            temp_label_buffer = torch.cat((self.dictionary_label_buffer, batch_label))
            self.dictionary_label_buffer = temp_label_buffer[indices]

    @torch.no_grad()
    def _sparse_code(self, batch_data):

        similarities = self._calculate_similarity(batch_data, self.dictionary_atoms_norm)
        sub_similarities, sub_indices = torch.topk(similarities, self.n_nonzero_coefs, dim=1)
        support_matrix = self.dictionary_atoms_norm[sub_indices]
        support_similarites = self._calculate_similarity(support_matrix, support_matrix, batched_inputs=True)
        if self.n_nonzero_coefs == 1:
            x_opt = torch.ones_like(sub_similarities)
            error = (1 - sub_similarities).squeeze()
        else:
            x_opt, error = approximate_nnk(support_similarites, sub_similarities, sub_similarities, x_tol=self.optim_tol, num_iter=self.optim_itr)
            x_opt = nn.functional.normalize(x_opt, p=1, dim=1)

        return x_opt, sub_indices, error

    @torch.no_grad()
    def _update_dict_inv(self):
        nonzero_indices = torch.nonzero(self.influence_cache.diag() > self.influence_tol).squeeze()
        n_nonzero = len(nonzero_indices)
        if n_nonzero < self.n_clusters:
            # Replacing self.n_clusters - n_nonzero unused atoms with buffered data
            influence_subset_inv = torch.linalg.inv(self.influence_cache[nonzero_indices, :][:, nonzero_indices])
            data_cache_subset = self.data_cache[nonzero_indices, :]
            label_cache_subset = self.label_cache[nonzero_indices, :]
            self.dictionary_atoms[:n_nonzero] = influence_subset_inv @ data_cache_subset
            self.dictionary_atoms[n_nonzero:] = self.dictionary_data_buffer[:self.n_clusters - n_nonzero]
            if self.n_classes is not None:
                self.atom_labels[:n_nonzero] = influence_subset_inv @ label_cache_subset
                self.atom_labels[n_nonzero:] = self.dictionary_label_buffer[:self.n_clusters - n_nonzero]

        else:
            WWt_inv = torch.linalg.inv(self.influence_cache)
            self.dictionary_atoms = WWt_inv @ self.data_cache
            if self.n_classes is not None:
                self.atom_labels = WWt_inv @ self.label_cache

        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)

    @torch.no_grad()
    def _update_dict_residual(self):
        n_nonzero = 0
        for i in range(self.n_clusters):
            influence_i = self.influence_cache[i]
            if influence_i[i] < self.influence_tol:
                self.dictionary_atoms[i] = self.dictionary_data_buffer[n_nonzero]
                if self.n_classes is not None:
                    self.atom_labels[i] = self.dictionary_label_buffer[n_nonzero]
                n_nonzero += 1

            else:
                self.dictionary_atoms[i] += (self.data_cache[i] - influence_i @ self.dictionary_atoms) / influence_i[i]

        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)

    @torch.no_grad()
    def update_dict(self):
        if self.use_residual_update:
            self._update_dict_residual()
        else:
            self._update_dict_inv()

    def forward(self, batch_data, batch_label=None, update_cache=True, update_dict=True, get_codes=False):
        # batch_data = nn.functional.normalize(batch_data, dim=1)
        batch_size = batch_data.shape[0]
        
        x_opt, indices, error = self._sparse_code(batch_data)
    
        if update_cache:
            batch_row_indices = torch.arange(0, batch_size, dtype=torch.long).cuda().unsqueeze(1)
            batch_W = torch.sparse_coo_tensor(torch.stack((indices.ravel(), torch.tile(batch_row_indices, [1, self.n_nonzero_coefs]).ravel()), 0), x_opt.ravel(), (self.n_clusters, batch_size), dtype=torch.float32).to(self.device) #  # batch_row_indices.ravel()
            # import IPython; IPython.embed()
            if self.n_classes is not None:
                batch_label = self._process_labels(batch_label).to(self.device)
                
            self._update_cache(batch_W, batch_data, batch_label)# 
            self._update_buffer(batch_data, batch_label, error)
        if update_dict:
            self.update_dict()
            # self.reset_cache()
            
        interpolated = torch.bmm(x_opt.unsqueeze(1), self.dictionary_atoms[indices]).squeeze(1)
        label_interpolated = None
        if self.n_classes is not None:
            label_interpolated = torch.bmm(x_opt.unsqueeze(1), self.atom_labels[indices]).squeeze(1)
            
        if get_codes: 
            return batch_data, interpolated, label_interpolated, batch_W.t().to_dense()

        return batch_data, interpolated, label_interpolated

    def _calculate_similarity(self, input1, input2, batched_inputs=False):
        if batched_inputs:
            return torch.bmm(input1, input2.transpose(1,2)) 
            
        return input1 @ input2.t()
    
def get_embeddings(model, tokenizer, dataset, batch_size, device:str, 
    field:str="context"):

    dataloader = DataLoader(dataset, batch_size=batch_size)

    _embeds = torch.empty((0, 768)).to(device)
        
    labels, text_labels = [], []
    with torch.no_grad():            
        for batch in dataloader:                
            if field == "context":
                sequences = []
                for eu, context in zip(batch["error_utterance"]["text"], 
                    batch["context"]):
                    context = context.replace("\nuser: ", " [SEP] ")
                    context = context.replace("\nsystem: ", " [SEP] ")
                    context = context.replace("user: ", "[CLS] ")
                    context = context.replace("system: ", "[CLS] ")
                    context = context + " [SEP] " + eu + " [SEP]"
                    sequences.append(context)
            else:
                sequences = batch[field]             

            labels += [e.item() for e in batch["error_type"]]
            text_labels += batch["error_type_str"]

            _max_length = min(max([len(s) for s in sequences]), 512)\
                if isinstance(sequences, list) else 512
            tokenized = tokenizer(sequences, truncation=True, padding=True, max_length=_max_length, return_attention_mask=True,return_tensors="pt").to(device)

            output = model(**tokenized)
            _embeds = torch.cat((_embeds, output["pooler_output"]))
    
    return _embeds, torch.tensor(labels), text_labels