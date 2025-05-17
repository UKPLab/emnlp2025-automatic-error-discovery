'''
    This script contains the code for the loss functions used in this project, including our enhanced Soft Nearest Neighbor Loss.
'''
import torch
import torch.nn.functional as F


class SoftNNLoss(torch.nn.Module):
    """
    This is the Soft Nearest Neighbor Loss using cosine similarity,
    according to https://arxiv.org/pdf/1902.01889.
    I've extended it with an option to include extra negatives per class.
    """

    def __init__(self, temperature: float = 0.2, margin: float = 0.3):
        super(SoftNNLoss, self).__init__()
        self._temperature = temperature
        self._margin = margin

    def forward(self, anchor_embeddings: torch.Tensor, positives_embeddings: torch.Tensor,
                anchor_labels: torch.Tensor, positives_labels: torch.Tensor,
                negative_embeddings: torch.Tensor = None, negative_labels: torch.Tensor = None,
                remove_diagonal: bool = True) -> torch.Tensor:
        """
        Computes the soft nearest neighbors loss using cosine similarity.

        Args:
            anchor_labels: labels associated with the anchor embed.
            anchor_embeddings: Embedded anchor examples.
            positives_labels: labels associated with the positives embed.
            positives_embeddings: Embedded positives examples.
            temperature: Controls relative importance given to the pair of points.
            remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask

        Returns:
            loss: loss value for the current batch.
        """

        device = anchor_embeddings.device  # Get the device
        batch_size = anchor_labels.size(0)  # Get the batch size
        eps = 1e-9  # Small value for numerical stability

        # Normalize embeddings to have unit norm
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positives_embeddings = F.normalize(positives_embeddings, p=2, dim=1)

        # Calculate anchor-positive cosine similarity
        pairwise_ap_sim = torch.matmul(anchor_embeddings, positives_embeddings.transpose(0, 1))

        # Create a mask where labels of anchor and positive are different
        negative_ap_mask = (anchor_labels != positives_labels).float()
        # Subtract margin from similarities of anchor-positive pairs with different labels.  Important change.
        pairwise_ap_sim = pairwise_ap_sim - self._margin * negative_ap_mask
        pairwise_ap_sim = pairwise_ap_sim / self._temperature # Scale by temperature

        # Exponentiate the scaled similarities.
        exp_ap = torch.exp(pairwise_ap_sim)

        # Mask out diagonal entries
        if remove_diagonal:
            diag_ap = torch.diag(torch.ones(batch_size, dtype=torch.bool, device=device))
            diag_mask_ap = (~diag_ap).float()
            exp_ap = exp_ap * diag_mask_ap

        # Create mask for same class anchor-positive pairs
        pos_mask_ap, _ = self._build_masks(
            anchor_labels,
            positives_labels,
            batch_size=batch_size,
            remove_diagonal=remove_diagonal,
            device=device
        )
        pos_mask_ap = pos_mask_ap.float()

        # Calculate anchor-anchor cosine similarity
        pairwise_aa_sim = torch.matmul(anchor_embeddings, anchor_embeddings.transpose(0, 1))
        pairwise_aa_sim = pairwise_aa_sim / self._temperature
        exp_aa = torch.exp(pairwise_aa_sim)

        # Mask out diagonal entries for anchor-anchor pairs
        diag_aa = torch.diag(torch.ones(batch_size, dtype=torch.bool, device=device))
        diag_mask_aa = (~diag_aa).float()
        exp_aa_masked = exp_aa * diag_mask_aa  # Mask out self-comparisons

        # Create mask for same class anchor-anchor pairs (excluding the diagonal)
        pos_mask_aa, _ = self._build_masks(
            anchor_labels,
            anchor_labels,  # Use anchor_labels for both arguments
            batch_size=batch_size,
            remove_diagonal=True,  # Exclude self-similarity
            device=device
        )
        pos_mask_aa = pos_mask_aa.float()

        # Calculate the sum of exponentiated similarities for same class anchor-positive pairs
        pos_sim = torch.sum(exp_ap * pos_mask_ap, dim=1) + torch.sum(exp_aa_masked * pos_mask_aa, dim=1)

        # Calculate the sum of exponentiated similarities for *all* other anchors in the batch.
        anchor_anchor_sim = torch.sum(exp_aa_masked, dim=1)

        # Initialize total similarity with anchor-positive similarities and add anchor-anchor similarities
        all_sim = torch.sum(exp_ap, dim=1) + anchor_anchor_sim

        # Handle explicit negatives if provided
        if negative_embeddings is not None and negative_labels is not None:
            # Normalize negative embeddings
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            negative_mask = (anchor_labels != negative_labels).float()
            # Calculate anchor-negative cosine similarities
            neg_pairwise_sim = torch.matmul(anchor_embeddings, negative_embeddings.transpose(0, 1))
            neg_pairwise_sim = neg_pairwise_sim - self._margin * negative_mask # Apply margin
            neg_pairwise_sim = neg_pairwise_sim / self._temperature
            neg_exp = torch.exp(neg_pairwise_sim)
            all_sim = all_sim + torch.sum(neg_exp, dim=1)

        # Exclude examples with unique class (no other positive in the batch)
        excl = (torch.sum(pos_mask_ap, dim=1) != 0).float()

        # Calculate the ratio of same class neighborhood to all class neighborhood
        loss = pos_sim / (all_sim + eps)
        # Calculate the negative logarithm of the ratio and apply the exclusion mask
        loss = -torch.log(eps + loss) * excl

        return loss.mean()

    def _build_masks(self, anchor_labels, positives_labels, batch_size,
        remove_diagonal, device):
        """Builds positive mask."""
        pos_mask = torch.eq(anchor_labels.unsqueeze(1), positives_labels.unsqueeze(0))

        if remove_diagonal:
            diag = torch.diag(torch.ones(batch_size, dtype=torch.bool,
                device=device))
            pos_mask = pos_mask & (~diag)

        return pos_mask, None
    

class SupConLoss(torch.nn.Module):

    def __init__(self, temperature:int=0.07, device:str="cuda"):
        super(SupConLoss, self).__init__()
        self._temperature = temperature
        self._device = device

    
    def get_contrastive_mask_no_neighbors(self, anchor_indices:list[int], 
            targets:list[int]=None):
        """If one wants to use SupConLoss with mask instead of list of targets.

        Args:
            anchor_indices (_type_): _description_
            neighbor_indices (_type_): _description_
            targets (_type_): _description_

        Returns:
            _type_: torch.Tensor [batch_size x batch_size]
        """

        # create mask template with dimensions batch_size x batch_size
        mask = torch.zeros(len(anchor_indices), len(anchor_indices))

        for i, _ in enumerate(anchor_indices):
            for j, _ in enumerate(anchor_indices):
                if targets[i] == targets[j]:
                    mask[i][j] = 1 # if same labels                
        return mask.to(self._device)

    def get_contrastive_mask(self, anchor_indices:list[int], 
            neighbor_indices:list[int], targets:list[int]=None):
        """If one wants to use SupConLoss with mask instead of list of targets.

        Args:
            anchor_indices (_type_): _description_
            neighbor_indices (_type_): _description_
            targets (_type_): _description_

        Returns:
            _type_: torch.Tensor [batch_size x batch_size]
        """

        print(neighbor_indices)

        # create mask template with dimensions batch_size x batch_size
        mask = torch.zeros(len(anchor_indices), len(anchor_indices))

        for i, neighbor_idx in enumerate(neighbor_indices):
            # neighbors are expected to be of the same class as anchor (or at 
            # least similar)
            mask[i][i] = 1
            for j, anchor_idx in enumerate(anchor_indices):
                if anchor_idx == neighbor_idx:
                    mask[i][j] = 1 # if in neighbors 
                if targets:               
                    if targets[i] == targets[j]:
                        mask[i][j] = 1 # if same labels                    
        return mask.to(self._device)


    def forward(self, embeddings_1:torch.Tensor, embeddings_2:torch.Tensor, 
        batch_size:int, labels:torch.Tensor=None, mask:torch.Tensor=None):
        """(Simplified) contrastive loss from the USNID implementation (https://ieeexplore.ieee.org/abstract/document/10349963) in TEXTOIR (https://github.com/thuiar/TEXTOIR/blob/main/open_intent_discovery/losses/SupConLoss.py). I reduced it to what is relevant for us (contrast_mode='all' from the original implementation). It is basically what SynCID refers to. LOOP is using it directly. It is built on the unsupervised contrastive loss in SimCLR (which is the fallback if no labels are provided).

        It is not allowed to pass both labels and mask. If both are provided, labels are ignored.

        Args:
            embeddings_1: First set of embeddings in the shape [batch_size, 
                hidden size], e.g., [4, 768].
            embeddings_2: Second set of embeddings in the shape [batch_size, 
                hidden size], e.g., [4, 768].
            batch_size: The number of samples per batch.
            labels: Ground truth labels (required for supervised contrastive 
                loss calculation) as a list of integers.
            mask: Mask (matrix) of with labels for positive and negative 
                samples.       
        Returns:
            A loss scalar.
        """

        # normalize (required for cosine similarity)
        norm_embeds_1 = F.normalize(embeddings_1)
        norm_embeds_2 = F.normalize(embeddings_2)

        # stack anchor with positive features ([batch_size, 2, 768]) 
        features = torch.stack([norm_embeds_1, norm_embeds_2], dim=1)
        anchor_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # if mask is none, build one based on the labels. If labels is also 
        # none, use the default one (diagonal with ones, others zero).
        if mask is None:
            if labels is None:
                mask =\
                    torch.eye(batch_size, dtype=torch.float32).to(self._device)
            else:
                labels_expanded = labels.expand(batch_size, batch_size)
                labels_mask =\
                    torch.eq(labels_expanded, labels_expanded.T).long()
                mask = torch.scatter(labels_mask, 0, torch.arange(batch_size)\
                    .unsqueeze(0).to(self._device), 1)    
        

        # compute cos sim
        cos_sim = torch.div(torch.matmul(features, features.T), 
            self._temperature)
        
        # for numerical stability
        cos_sim_max, _ = torch.max(cos_sim, dim=1, keepdim=True)
        logits = cos_sim - cos_sim_max.detach()

        # tile mask and mask-out self-contrast cases
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self._device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss