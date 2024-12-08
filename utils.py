import torch
import torch.nn as nn
from torch.nn import functional as F # loading libraries
import pandas as pd
import numpy as np
import optuna
import ast
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from contextlib import nullcontext
import json
import regex as re
import numpy as np
from collections import defaultdict
import base64
import math
import os
import inspect
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LogisticRegression(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size,              # Maximum length of input sequences
        vector_length,             # Size of each embedding vector
        num_metadata_features=None,  # Number of metadata features (optional)
        padding_idx=None           # Padding index for embedding
    ):
        super().__init__()
        self.padding_idx = padding_idx

        # Embedding layer for token indices
        self.embedding = nn.Embedding(vocab_size + 1, vector_length, padding_idx=padding_idx)

        # Compute input size: flattened token features + metadata features (if any)
        token_input_size = vector_length * context_size
        input_size = token_input_size + (num_metadata_features if num_metadata_features else 0)

        # Linear layer for logistic regression (single output node for binary classification)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, input_ids, attention_mask=None, metadata_features=None, targets=None):
        """
        Forward pass for logistic regression.
        :param input_ids: Input token indices (B, context_size)
        :param attention_mask: Mask for valid tokens (B, context_size)
        :param metadata_features: Metadata features (optional) (B, num_metadata_features)
        :param targets: Target labels for computing loss
        """
        # Embed tokens
        embedded_tokens = self.embedding(input_ids)  # Shape: (B, context_size, vector_length)

        # Apply attention mask to zero out padded tokens
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).float()  # Shape: (B, context_size, 1)
            embedded_tokens = embedded_tokens * attention_mask  # Mask out padded tokens

        # Flatten token features
        token_features = embedded_tokens.view(embedded_tokens.size(0), -1)  # Shape: (B, context_size * vector_length)

        # Concatenate metadata features if provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Pass through the linear layer
        logits = self.linear(combined_features).squeeze(-1)  # Shape: (B,)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())

        return logits, loss

    def predict(self, input_ids, attention_mask=None, metadata_features=None):
        """
        Generate predictions.
        """
        logits, _ = self.forward(input_ids, attention_mask, metadata_features)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long()  # Threshold at 0.5
        return predictions

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
            print(f"Using fused AdamW: {use_fused}")

            return optimizer
    

class MLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        vector_length,             # Size of each embedding vector
        context_size,              # Maximum length of input sequences
        num_metadata_features=None,  # Number of metadata features (optional)
        hidden_layers=None,        # Hidden layers for the MLP
        num_classes=2,             # Number of output classes
        dropout=0.2,               # Dropout rate
        padding_idx=None,          # Padding index for embedding
    ):
        super().__init__()
        self.padding_idx = padding_idx

        # Embedding layer for token indices
        self.embedding = nn.Embedding(vocab_size + 1, vector_length, padding_idx=padding_idx)

        # Input size after flattening (context_size * vector_length)
        token_input_size = context_size * vector_length

        # Adjust input size to include metadata if provided
        input_size = token_input_size
        if num_metadata_features:
            input_size += num_metadata_features

        # Create MLP layers
        mlp_layers = []
        if hidden_layers:
            for hidden_size in hidden_layers:
                mlp_layers.append(nn.Linear(input_size, hidden_size))
                mlp_layers.append(nn.GELU())
                mlp_layers.append(nn.Dropout(dropout))
                input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers) if hidden_layers else nn.Identity()

        # Final classification layer
        self.final_fc = nn.Linear(input_size, num_classes)

    def forward(self, input_ids, attention_mask=None, metadata_features=None, targets=None):
        """
        Forward pass through the model.
        :param input_ids: Input token indices (B, context_size)
        :param attention_mask: Mask for valid tokens (B, context_size)
        :param metadata_features: Metadata features (optional) (B, num_metadata_features)
        :param targets: Target labels for computing loss
        """
        # Embed tokens and apply attention mask
        embedded_tokens = self.embedding(input_ids)  # Shape: (B, context_size, vector_length)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).float()  # Shape: (B, context_size, 1)
            embedded_tokens = embedded_tokens * attention_mask  # Mask out padded tokens

        # Flatten token features
        token_features = embedded_tokens.view(embedded_tokens.size(0), -1)  # Shape: (B, context_size * vector_length)

        # Concatenate metadata if provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Process through the MLP
        features = self.mlp(combined_features)

        # Final classification
        logits = self.final_fc(features)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def predict(self, input_ids, attention_mask=None, metadata_features=None):
        """
        Generate predictions.
        """
        logits, _ = self.forward(input_ids, attention_mask, metadata_features)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
            print(f"Using fused AdamW: {use_fused}")

            return optimizer


class XGBoost(nn.Module):
    def __init__(
        self,
        context_size,               # Length of token sequences (T)
        num_metadata_features=None,  # Number of metadata features (optional)
        max_depth=6,                # Maximum tree depth for XGBoost
        learning_rate=0.1,          # Learning rate for XGBoost
        n_estimators=100,           # Number of boosting rounds
        min_child_weight=1,         # Minimum child weight
        subsample=1.0,              # Subsample ratio of the training instances
        colsample_bytree=1.0,       # Subsample ratio of features
        gamma=0.0,                  # Minimum loss reduction for splitting
        reg_alpha=0.0,              # L1 regularization term
        reg_lambda=1.0,             # L2 regularization term
        objective="binary:logistic",  # Objective function for XGBoost
        num_classes=2               # Number of output classes
    ):
        super().__init__()
        self.context_size = context_size
        self.num_metadata_features = num_metadata_features
        self.num_classes = num_classes

        # Compute total input size (token indices + metadata)
        self.input_size = context_size
        if num_metadata_features:
            self.input_size += num_metadata_features

        # Initialize XGBoost parameters
        self.xgb_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": objective,
            "verbosity": 1,
            "tree_method": "auto",
        }

        # Placeholder for trained XGBoost model
        self.xgb_model = None

    def _apply_attention_mask(self, token_indices, attention_mask):
        """
        Mask out padded token indices using the attention mask.
        :param token_indices: Input token indices (B, T)
        :param attention_mask: Attention mask (B, T)
        :return: Flattened masked token indices (B, T)
        """
        if attention_mask is not None:
            return token_indices * attention_mask  # Mask out padded tokens (set to 0)
        return token_indices

    def forward(self, token_indices, attention_mask=None, metadata_features=None, targets=None):
        """
        Forward pass for the model.
        :param token_indices: Input token indices (B, T)
        :param attention_mask: Mask for valid tokens (B, T)
        :param metadata_features: Metadata features (optional) (B, num_metadata_features)
        :param targets: Target labels for computing loss (optional)
        """
        # Apply attention mask to token indices
        masked_token_indices = self._apply_attention_mask(token_indices, attention_mask)

        # Flatten token indices
        token_features = masked_token_indices.view(masked_token_indices.size(0), -1)  # Shape: (B, T)

        # Concatenate token features and metadata if metadata is provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Ensure input is a NumPy array for XGBoost
        combined_features_np = combined_features.detach().cpu().numpy()

        # Perform prediction if XGBoost model is trained
        logits = None
        if self.xgb_model is not None:
            logits = self.xgb_model.predict_proba(combined_features_np)

        # Convert logits back to PyTorch tensor
        logits = torch.tensor(logits, dtype=torch.float32, device=token_indices.device)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            targets = targets.float()
            loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], targets)

        return logits, loss

    def fit(self, token_indices, attention_mask=None, metadata_features=None, targets=None):
        """
        Fit the XGBoost model on the provided data.
        :param token_indices: Input token indices (N, T)
        :param attention_mask: Mask for valid tokens (N, T)
        :param metadata_features: Metadata features (optional) (N, num_metadata_features)
        :param targets: Target labels for training (N,)
        """
        if targets is None:
            raise ValueError("Targets must be provided for training.")

        # Apply attention mask to token indices
        masked_token_indices = self._apply_attention_mask(token_indices, attention_mask)

        # Flatten token indices
        token_features = masked_token_indices.view(masked_token_indices.size(0), -1)  # Shape: (N, T)

        # Concatenate token features and metadata if metadata is provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Ensure inputs are NumPy arrays for XGBoost
        combined_features_np = combined_features.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Train the XGBoost model
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(combined_features_np, targets_np)

    def predict(self, token_indices, attention_mask=None, metadata_features=None):
        """
        Generate predictions using the trained model.
        """
        logits, _ = self.forward(token_indices, attention_mask, metadata_features)
        predictions = torch.argmax(logits, dim=1)
        return predictions
    

class KNN(nn.Module):
    def __init__(
        self,
        context_size,                  # Length of token sequences (T)
        num_metadata_features=None,    # Number of metadata features (optional)
        n_neighbors=5,                 # Number of neighbors
        weights="uniform",             # Weight function ("uniform" or "distance")
        metric="minkowski",            # Distance metric
        p=2,                           # Power parameter for Minkowski metric
        leaf_size=30                   # Leaf size for tree-based algorithms
    ):
        super().__init__()
        self.context_size = context_size
        self.num_metadata_features = num_metadata_features

        # Compute the input size (token indices + metadata)
        self.input_size = context_size
        if num_metadata_features:
            self.input_size += num_metadata_features

        # Store KNN hyperparameters
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.leaf_size = leaf_size

        # Placeholder for trained KNN model
        self.knn_model = None

    def _apply_attention_mask(self, token_indices, attention_mask):
        """
        Mask out padded token indices using the attention mask.
        :param token_indices: Input token indices (B, T)
        :param attention_mask: Attention mask (B, T)
        :return: Flattened masked token indices (B, T)
        """
        if attention_mask is not None:
            return token_indices * attention_mask  # Mask out padded tokens
        return token_indices

    def forward(self, token_indices, attention_mask=None, metadata_features=None, targets=None):
        """
        Forward pass for the model.
        :param token_indices: Input token indices (B, T)
        :param attention_mask: Mask for valid tokens (B, T)
        :param metadata_features: Metadata features (optional) (B, num_metadata_features)
        :param targets: Target labels for computing loss (optional)
        """
        # Apply attention mask to token indices
        masked_token_indices = self._apply_attention_mask(token_indices, attention_mask)

        # Flatten token indices
        token_features = masked_token_indices.view(masked_token_indices.size(0), -1)  # Shape: (B, T)

        # Concatenate token features and metadata if metadata is provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Ensure input is a NumPy array for KNN
        combined_features_np = combined_features.detach().cpu().numpy()

        # Perform prediction if KNN model is trained
        logits = None
        if self.knn_model is not None:
            logits = self.knn_model.predict_proba(combined_features_np)

        # Convert logits back to PyTorch tensor
        logits = torch.tensor(logits, dtype=torch.float32, device=token_indices.device)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            targets = targets.long()
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def fit(self, token_indices, attention_mask=None, metadata_features=None, targets=None):
        """
        Fit the KNN model on the provided data.
        :param token_indices: Input token indices (N, T)
        :param attention_mask: Mask for valid tokens (N, T)
        :param metadata_features: Metadata features (optional) (N, num_metadata_features)
        :param targets: Target labels for training (N,)
        """
        if targets is None:
            raise ValueError("Targets must be provided for training.")

        # Apply attention mask to token indices
        masked_token_indices = self._apply_attention_mask(token_indices, attention_mask)

        # Flatten token indices
        token_features = masked_token_indices.view(masked_token_indices.size(0), -1)  # Shape: (N, T)

        # Concatenate token features and metadata if metadata is provided
        if metadata_features is not None:
            combined_features = torch.cat((token_features, metadata_features), dim=1)
        else:
            combined_features = token_features

        # Ensure inputs are NumPy arrays for KNN
        combined_features_np = combined_features.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Train the KNN model
        self.knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            leaf_size=self.leaf_size
        )
        self.knn_model.fit(combined_features_np, targets_np)

    def predict(self, token_indices, attention_mask=None, metadata_features=None):
        """
        Generate predictions using the trained model.
        """
        logits, _ = self.forward(token_indices, attention_mask, metadata_features)
        predictions = torch.argmax(logits, dim=1)
        return predictions
    

class Simple_Transformer(nn.Module):
    def __init__(self, vocab_size, vector_length, context_size, num_blocks, num_heads, dropout, padding_idx, num_metadata_features=None, metadata_vector_length=None, num_classes=2):
        super().__init__()
        self.context_size = context_size
        self.padding_idx = padding_idx  # Padding token is set to vocab_size + 1

        # Embedding layers for text (URLs)
        self.char_embedding = nn.Embedding(vocab_size + 1, vector_length, padding_idx=self.padding_idx)  # embedding with padding_idx
        self.pos_embedding = nn.Embedding(context_size, vector_length)  # positional embedding lookup

        # Transformer blocks for text processing (no nn.Sequential to allow passing mask)
        self.blocks = nn.ModuleList([Block(vector_length, num_heads, dropout) for _ in range(num_blocks)])  # stacked transformer blocks
        self.norm = nn.LayerNorm(vector_length)  # final normalization layer

        # Metadata layers only if metadata is present
        if num_metadata_features and metadata_vector_length:
            self.metadata_layer = nn.Sequential(
                nn.Linear(num_metadata_features, metadata_vector_length),  # process metadata features
                nn.ReLU(),
                nn.Linear(metadata_vector_length, vector_length)  # map to same vector length as text embeddings
            )
            self.use_metadata = True
            # Final classification layer (after concatenating text and metadata)
            self.final = nn.Linear(2 * vector_length, num_classes)
        else:
            self.use_metadata = False
            # Final classification layer (just text embeddings)
            self.final = nn.Linear(vector_length, num_classes)

    def adjust_length(self, x, max_length, pad_token=0):
        """
        Adjust the length of input sequences to max_length by truncating or padding.
        """
        length = x.size(1)  # Assuming x is of shape [B, T]
        if length > max_length:
            return x[:, :max_length]  # Truncate to max_length
        elif length < max_length:
            pad_size = max_length - length
            padding = torch.full((x.size(0), pad_size), pad_token, dtype=x.dtype, device=x.device)
            return torch.cat((x, padding), dim=1)  # Pad to max_length
        return x

    def forward(self, x_text, attention_mask, x_metadata=None, targets=None):
        B, T = x_text.shape

        # Adjust input length to max_length
        x_text = self.adjust_length(x_text, self.context_size, pad_token=self.padding_idx)
        T = x_text.size(1)  # Update T to reflect the adjusted length

        # Text (URL) embedding + positional encoding
        char_token = self.char_embedding(x_text)  # get character embeddings
        pos_indices = torch.arange(T, device=x_text.device)  # create position indices for max_length
        pos_token = self.pos_embedding(pos_indices)  # get positional embeddings
        token = char_token + pos_token  # sum token and positional embeddings

        # Apply each block in the transformer stack with attention mask
        for block in self.blocks:
            token = block(token, mask=attention_mask)

        # Apply final layer normalization
        x_text = self.norm(token)
        x_text = x_text.mean(dim=1)  # global average pooling over the sequence

        if self.use_metadata and x_metadata is not None:
            # Process metadata (numerical features)
            x_metadata = self.metadata_layer(x_metadata)  # process numerical features
            # Concatenate the text and metadata outputs
            x_combined = torch.cat([x_text, x_metadata], dim=1)
        else:
            x_combined = x_text  # Use text embeddings only if metadata is not provided

        # Final classification
        logits = self.final(x_combined)

        loss = None
        if targets is not None:  # training mode
            # Compute the loss, ignoring padding tokens in the target
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def predict(self, x_text, attention_mask, x_metadata=None):
        """
        Classify the input text based on the model. Optionally, metadata can be used.
        """
        # Use the forward pass for classification
        logits, _ = self.forward(x_text, attention_mask, x_metadata)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions

class Head(nn.Module):
    def __init__(self, vector_length, head_size, dropout):
        super().__init__()
        self.query = nn.Linear(vector_length, head_size, bias=False)  # creating input to query linear layer
        self.key = nn.Linear(vector_length, head_size, bias=False)  # creating input to key linear layer
        self.value = nn.Linear(vector_length, head_size, bias=False)  # creating input to value linear layer
        self.dropout = nn.Dropout(dropout)  # setting dropout

    def forward(self, x, mask=None):
        B, T, C = x.shape  # getting tensor shape of input
        q = self.query(x)  # getting query of inputs
        k = self.key(x)  # getting key of inputs
        v = self.value(x)  # getting value of inputs

        # Scaled dot-product attention
        weights = q @ k.transpose(-2, -1) / (C ** 0.5)  # scaled attention by sqrt of dimension

        if mask is not None:
            # Expand mask to [B, 1, T] and broadcast to [B, T, T] to match weights shape
            mask = mask[:, None, :].expand(-1, T, -1)
            weights = weights.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)  # applying softmax for attention scores
        weights = self.dropout(weights)  # applying dropout to attention weights
        attention = weights @ v  # calculating the attention output
        return attention


class MultiAttention(nn.Module):  # multi-attention head class
    def __init__(self, vector_length, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        head_size = vector_length // num_heads
        self.heads = nn.ModuleList([Head(vector_length, head_size, dropout) for _ in range(num_heads)])  # parallel attention heads
        self.proj = nn.Linear(vector_length, vector_length)  # projection layer for concatenated head outputs
        self.dropout = nn.Dropout(dropout)  # setting dropout

    def forward(self, x, mask=None):
        attention = torch.cat([head(x, mask) for head in self.heads], dim=-1)  # pass mask to each head
        output = self.proj(attention)  # projecting concatenated heads
        return self.dropout(output)


class FeedFwd(nn.Module):  # feed forward network
    def __init__(self, vector_length, dropout):
        super().__init__()
        self.fwd = nn.Sequential(
            nn.Linear(vector_length, 4 * vector_length),
            nn.ReLU(),
            nn.Linear(4 * vector_length, vector_length),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fwd(x)


class Block(nn.Module):  # transformer block that will be looped
    def __init__(self, vector_length, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(vector_length)  # layer normalization before self-attention
        self.attention = MultiAttention(vector_length, num_heads, dropout)  # multi-head attention layer
        self.norm2 = nn.LayerNorm(vector_length)  # layer normalization before feed forward network
        self.ffn = FeedFwd(vector_length, dropout)  # feed forward network

    def forward(self, x, mask=None):
        # Apply normalization and attention with skip connection
        x = x + self.attention(self.norm1(x), mask)
        # Apply normalization and feed-forward network with skip connection
        return x + self.ffn(self.norm2(x))
    

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, vector_length, hidden_length, num_layers, dropout, padding_idx, num_metadata_features=None, metadata_vector_length=None, num_classes=2):
        super().__init__()
        self.padding_idx = padding_idx

        # Embedding and LSTM layers for URL encoding
        self.embedding = nn.Embedding(vocab_size + 1, vector_length, padding_idx=padding_idx)  # +1 for padding token
        self.lstm = nn.LSTM(vector_length, hidden_length, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Metadata processing network (optional)
        if num_metadata_features and metadata_vector_length:
            self.metadata_net = nn.Sequential(
                nn.Linear(num_metadata_features, metadata_vector_length),
                nn.GELU(),
                nn.Linear(metadata_vector_length, hidden_length),
                nn.GELU(),
            )
            self.use_metadata = True
            # Final classification layer (concatenates LSTM and metadata output)
            self.fc = nn.Linear(hidden_length * 2 + hidden_length, num_classes)  # hidden_length*2 for bidirectional LSTM
        else:
            self.use_metadata = False
            # Final classification layer (only LSTM output)
            self.fc = nn.Linear(hidden_length * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, metadata=None, targets=None):
        # Encode inputs with embedding
        embedded = self.embedding(input_ids)  # Shape: [B, T, vector_length]

        # Compute sequence lengths based on attention mask
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).to("cpu")  # Actual lengths of each sequence
        else:
            lengths = (input_ids != self.padding_idx).sum(dim=1).to("cpu")  # Fallback if no attention mask is provided

        # Pack the embedded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # Shape: [B, T, hidden_length*2]

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            attention_mask = attention_mask[:, :lstm_out.size(1)]  # Adjust attention mask to match lstm_out's time dimension
            lstm_out = (lstm_out * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            lstm_out = lstm_out.mean(dim=1)  # Fallback mean pooling if no mask is provided

        # Process metadata if provided
        if self.use_metadata and metadata is not None:
            metadata_out = self.metadata_net(metadata)  # Shape: [B, hidden_length]
            # Concatenate LSTM and metadata outputs
            combined_out = torch.cat((lstm_out, metadata_out), dim=1)  # Shape: [B, hidden_length*2 + hidden_length]
        else:
            combined_out = lstm_out  # Use only LSTM output if metadata is not provided

        # Final classification
        logits = self.fc(combined_out)  # Shape: [B, num_classes]

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def predict(self, input_ids, attention_mask=None, metadata=None):
        """
        Generates class predictions.
        """
        logits, _ = self.forward(input_ids, attention_mask, metadata)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
            print(f"Using fused AdamW: {use_fused}")

            return optimizer

class Transformer(nn.Module):
    def __init__(self, vocab_size, vector_length, context_size, num_blocks, num_heads, dropout, padding_idx, num_metadata_features=None, metadata_vector_length=None, num_classes=2):
        super().__init__()
        self.context_size = context_size
        self.padding_idx = padding_idx  # Padding token is set to vocab_size + 1

        # Embedding layers for text (URLs)
        self.char_embedding = nn.Embedding(vocab_size + 1, vector_length, padding_idx=self.padding_idx)
        self.pos_embedding = nn.Embedding(context_size, vector_length)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(vector_length, num_heads, dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(vector_length)

        if num_metadata_features and metadata_vector_length:
            self.metadata_layer = nn.Sequential(
                nn.Linear(num_metadata_features, metadata_vector_length),
                nn.GELU(),
                nn.Dropout(dropout),  # Add dropout after activation
                nn.Linear(metadata_vector_length, vector_length),
                nn.Dropout(dropout),  # Add dropout after the final linear layer
            )
            self.use_metadata = True
            self.final = nn.Linear(2 * vector_length, num_classes)  # Concatenated input
        else:
            self.use_metadata = False
            self.final = nn.Linear(vector_length, num_classes)

    def adjust_length(self, x, max_length, pad_token=0):
        """
        Adjust the length of input sequences to max_length by truncating or padding.
        """
        length = x.size(1)
        if length > max_length:
            return x[:, :max_length]
        elif length < max_length:
            pad_size = max_length - length
            padding = torch.full((x.size(0), pad_size), pad_token, dtype=x.dtype, device=x.device)
            return torch.cat((x, padding), dim=1)
        return x

    def forward(self, x_text, attention_mask, x_metadata=None, targets=None):
        B, T = x_text.shape

        # Adjust input length
        x_text = self.adjust_length(x_text, self.context_size, pad_token=self.padding_idx)
        T = x_text.size(1)

        # Text (URL) embeddings + positional encoding
        char_token = self.char_embedding(x_text)
        pos_indices = torch.arange(T, device=x_text.device)
        pos_token = self.pos_embedding(pos_indices)
        token = char_token + pos_token
        token = self.dropout(token)

        # Apply transformer blocks
        for block in self.blocks:
            token = block(token, mask=attention_mask)

        # Final normalization
        x_text = self.norm(token)
        x_text = x_text.mean(dim=1)  # Global average pooling

        if self.use_metadata and x_metadata is not None:
            x_metadata = self.metadata_layer(x_metadata)
            x_combined = torch.cat([x_text, x_metadata], dim=1)
        else:
            x_combined = x_text

        # Final classification
        logits = self.final(x_combined)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, ignore_index=self.padding_idx)
        return logits, loss

    def predict(self, x_text, attention_mask, x_metadata=None):
        logits, _ = self.forward(x_text, attention_mask, x_metadata)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
            print(f"Using fused AdamW: {use_fused}")

            return optimizer

class SelfAttention(nn.Module):
    def __init__(self, vector_length, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = vector_length // num_heads
        self.vector_length = vector_length
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(vector_length, 3 * vector_length)
        self.out_proj = nn.Linear(vector_length, vector_length)

        # Check for PyTorch native attention
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Using slower attention. Flash Attention requires PyTorch >= 2.0.")

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # Ensure mask is expanded correctly
        if mask is not None:
            # Expand mask to [batch_size, num_heads, 1, seq_len]
            mask = mask[:, None, None, :].expand(B, self.num_heads, T, T).to(dtype=q.dtype)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            y = attn_weights @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return self.dropout(y)

class FeedFwd(nn.Module):
    def __init__(self, vector_length, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_length, 4 * vector_length),
            nn.GELU(),
            nn.Linear(4 * vector_length, vector_length),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, vector_length, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(vector_length)
        self.attention = SelfAttention(vector_length, num_heads, dropout)
        self.norm2 = nn.LayerNorm(vector_length)
        self.ffn = FeedFwd(vector_length, dropout)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class BPE_Tokenizer:
    def __init__(self, text=None, vocab_size=None, directory=None, logging=None, 
                 special_tokens=None, notify_every=None, save_every=None, save_directory=None):
        self.vocab = {}
        self.merges = {}
        self.padding_idx = None
        self.special_tokens = special_tokens or {}
        self.notify_every = notify_every  # Optional parameter to notify every nth vocab
        self.save_every = save_every      # Save every nth vocab
        self.save_directory = save_directory  # Directory to save vocab checkpoints
        self.text_regex = re.compile(r"<[^>]+>|https?://|www\.|/|\.|=|\?|&|#|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+")

        if (text is None and directory is None) or (text is not None and directory is not None):
            raise Exception("You can only pass either the model's tokenization vocabulary or the training text file.")

        if directory:
            self.load_vocab(directory)
        else:
            if not vocab_size:
                raise Exception("You must specify the tokenization vocabulary size.")
            else:
                self.create_vocab(text, vocab_size, logging)

    def create_vocab(self, text, vocab_size, logging=None):
        # Initialize with 256 basic UTF-8 tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        current_size = len(self.vocab)

        # Merge pairs until the target vocab size
        num_merges = vocab_size - current_size
        tokens = re.findall(self.text_regex, text)
        ids = [np.frombuffer(token.encode("utf-8"), dtype=np.uint8) for token in tokens]

        for i in range(current_size, current_size + num_merges):
            stats = defaultdict(int)
            for chunk_ids in ids:
                chunk_stats = self.get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] += count

            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = i
            ids = [self.merge(chunk_ids, pair, new_id) for chunk_ids in ids]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Notify every nth vocab if notify_every is set
            if self.notify_every and new_id % self.notify_every == 0:  # Adjust for initial vocab size of 256
                print(f"Notification: Vocab ID {new_id} has been created.")

            if logging:
                print(f"New token created: {self.vocab[new_id]}")

            # Save vocabulary every n tokens if save_every and save_directory are set
            if self.save_every and self.save_directory and new_id % self.save_every == 0:
                self.save_vocab(self.save_directory)
                print(f"Checkpoint saved at vocab ID {new_id}.")

        # Add special tokens at the very end
        self.add_special_tokens()

    def add_special_tokens(self):
        """Adds special tokens to the vocabulary at the very end."""
        next_idx = len(self.vocab)  # Start from the current size of the vocabulary
        for token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<URL>', '<TITLE>']:
            if token not in self.special_tokens:
                self.special_tokens[token] = next_idx
                self.vocab[next_idx] = token.encode('utf-8')
                next_idx += 1

        # Set the padding index as vocab_size + 1
        self.padding_idx = next_idx  # The final index after all tokens
        self.special_tokens['<PAD>'] = self.padding_idx

    def load_vocab(self, directory):
        # Load vocab and decode Base64-encoded strings back to bytes
        with open(f"{directory}/vocab.txt", "r") as f:
            vocab_serializable = json.load(f)
        self.vocab = {int(key): base64.b64decode(value) for key, value in vocab_serializable.items()}

        # Load merges and convert JSON lists back to tuples and ints
        with open(f"{directory}/merges.txt", "r") as f:
            merges_serializable = json.load(f)
        self.merges = {tuple([int(k) for k in json.loads(key)]): int(value) for key, value in merges_serializable.items()}

        # Load special tokens if they exist
        special_tokens_file = f"{directory}/special_tokens.json"
        if os.path.exists(special_tokens_file):
            with open(special_tokens_file, "r") as f:
                self.special_tokens = json.load(f)

        # Set the padding token as vocab_size + 1
        self.add_special_tokens()

    def save_vocab(self, directory):
        # Convert vocab bytes to Base64-encoded string
        vocab_serializable = {key: base64.b64encode(value).decode('utf-8') for key, value in self.vocab.items()}

        # Convert tuple keys and uint8 values in merges to lists and ints
        merges_serializable = {json.dumps([int(k) for k in key]): int(value) for key, value in self.merges.items()}

        with open(f"{directory}/vocab.txt", "w") as f:
            json.dump(vocab_serializable, f)

        with open(f"{directory}/merges.txt", "w") as f:
            json.dump(merges_serializable, f)

        # Save special tokens
        with open(f"{directory}/special_tokens.json", "w") as f:
            json.dump(self.special_tokens, f)

    def get_stats(self, stream):
        """Counts the frequency of adjacent pairs in the token stream using numpy for efficiency."""
        counts = defaultdict(int)
        stream = np.array(stream)
        for i in range(len(stream) - 1):
            pair = (stream[i], stream[i + 1])
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, id):
        """Merges token pairs into a single token."""
        merged = []
        i = 0
        ids = np.array(ids)
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(id)
                i += 2  # Skip the next token since it's merged
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def encode(self, text, max_length=None, return_attention_mask=True, add_special_tokens=True):
        is_single_text = isinstance(text, str) or (isinstance(text, list) and len(text) == 1)
        if isinstance(text, str):
            text = [text]

        tokenized_batch = []
        for t in text:
            # Tokenize and encode in one pass
            tokens = [
                self.special_tokens.get(match.group(), np.frombuffer(match.group().encode('utf-8'), dtype=np.uint8).tolist())
                for match in re.finditer(self.text_regex, t)
            ]

            # Flatten list of tokens
            tokens = [item for sublist in tokens for item in (sublist if isinstance(sublist, list) else [sublist])]

            # Add special tokens
            if add_special_tokens:
                tokens = [self.special_tokens['<BOS>']] + tokens + [self.special_tokens['<EOS>']]

            # Truncate to max_length before adding to the batch
            if max_length:
                tokens = tokens[:max_length]

            tokenized_batch.append(tokens)

        # Pad sequences and generate attention masks
        max_length = max_length or max(len(seq) for seq in tokenized_batch)
        tokenized_batch, attention_masks = self.pad_sequences(tokenized_batch, max_length)

        # Convert to tensors
        tokenized_batch_tensor = torch.tensor(tokenized_batch, dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long) if return_attention_mask else None

        if is_single_text:
            return {
                'input_ids': tokenized_batch_tensor[0],
                'attention_mask': attention_masks_tensor[0] if attention_masks_tensor is not None else None
            }

        return {
            'input_ids': tokenized_batch_tensor,
            'attention_mask': attention_masks_tensor
        } if return_attention_mask else {'input_ids': tokenized_batch_tensor}
    
    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        attention_masks = []
        for seq in sequences:
            # Truncate the sequence if it's longer than max_length
            seq = seq[:max_length]

            # Pad the sequence to max_length
            padded_seq = seq + [self.padding_idx] * (max_length - len(seq))
            padded_sequences.append(padded_seq)

            # Create attention mask
            attention_masks.append([1] * len(seq) + [0] * (max_length - len(seq)))
        return padded_sequences, attention_masks

    def create_attention_masks(self, sequences):
        """Creates attention masks for each sequence in a batch (1 for token, 0 for padding)."""
        attention_masks = []
        for seq in sequences:
            # Create a mask of 1s for all tokens that are not the padding_idx, 0 otherwise
            mask = [1 if token != self.padding_idx else 0 for token in seq]
            attention_masks.append(mask)
        return attention_masks

    def decode(self, encoded_input, skip_special_tokens=True):
        """
        Decode token IDs into text, preserving trailing spaces and handling special tokens.

        Args:
            encoded_input (dict, torch.Tensor, or list): Encoded tokens or a dictionary containing 'input_ids'.
            skip_special_tokens (bool): Whether to skip special tokens during decoding.

        Returns:
            str or list of str: The decoded text. Returns a single string if input is a single sequence,
            otherwise returns a list of strings.
        """
        # Extract input IDs
        if isinstance(encoded_input, dict):  # If it's a dictionary
            ids = encoded_input['input_ids']
        else:
            ids = encoded_input

        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Normalize input to batch format
        is_single_sequence = isinstance(ids[0], int)  # Check if it's a single sequence
        if is_single_sequence:
            ids = [ids]  # Wrap single sequence in a list for consistent batch processing

        decoded_texts = []
        for seq in ids:
            if skip_special_tokens:
                # Filter out special tokens
                seq = [id for id in seq if id not in self.special_tokens.values()]

            # Decode tokens
            decoded_tokens = [self.vocab[id] for id in seq if id != self.padding_idx]

            # Combine tokens while preserving spaces exactly as they were encoded
            stream = b"".join(decoded_tokens)
            decoded_text = stream.decode("utf-8", errors="replace")
            decoded_texts.append(decoded_text)

        # Return single string if input was a single sequence
        if is_single_sequence:
            return decoded_texts[0]
        return decoded_texts

def get_lr(step, warmup_steps, max_steps, min_lr, max_lr):
    if step < warmup_steps:
        return ((step+1) / warmup_steps) * max_lr

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

# Function to split the dataset into train, validation, and test sets
def train_val_test_split(dataset, train_val_ratio, val_test_ratio, stratify_column=None):
    # First split: Train/Validation
    train_val = dataset.train_test_split(test_size=train_val_ratio, stratify_by_column=stratify_column) if stratify_column is not None else dataset.train_test_split(test_size=train_val_ratio)
    # Second split: Validation/Test
    val_test = train_val['test'].train_test_split(test_size=val_test_ratio, stratify_by_column=stratify_column) if stratify_column is not None else train_val['test'].train_test_split(test_size=val_test_ratio)

    # Return a dictionary with train, validation, and test splits
    final_dataset = DatasetDict({
        'train': train_val['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    return final_dataset

# Function to prepare data by converting inputs into tensors
def prepare_data(example, feature_columns, target_column):
    inputs = [example[col] for col in feature_columns]  # Select feature columns
    input_tensor = torch.tensor(inputs, dtype=torch.float32)  # Convert inputs to tensor
    return {'inputs': input_tensor, 'labels': example[target_column]}  # Return inputs and labels

# Function to calculate means and standard deviations for normalization
def calculate_means_and_stds(dataset):
    means = torch.mean(dataset, dim=0)  # Calculate means across the dataset
    stds = torch.std(dataset, dim=0)  # Calculate standard deviations across the dataset
    return means, stds  # Return the calculated values

# Adds a stratification label for better class balancing during splitting
def add_stratification_column(example):
    label = 'safe' if example['label'] == 1 else 'vulnerable'
    example['stratify_label'] = label # Copy target to 'stratify_label'
    return example

# Normalizes the inputs for a single example using the means and stds
def normalize_example(example, means, stds):
    example['inputs'] = (example['inputs'] - means) / stds  # Normalize the input
    return example

# Tests the model on a given dataloader and calculates the average loss
def test_model(model, dataloader, device, fp16=False):
    final_loss = 0
    for i, batch in enumerate(dataloader):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].to(device)
        # If using half-precision (fp16), enable automatic mixed precision.
        context_manager = torch.autocast(device_type=device, dtype=torch.bfloat16) if fp16 else nullcontext()
        with context_manager:
          logits, loss = model(inputs, attention_mask, metadata, labels)  # Forward pass through the model
        final_loss += loss.item()  # Sum up the loss
    avg_loss = final_loss / (i + 1)  # Compute the average loss
    return avg_loss

# Runs inference on the model and collects results for evaluation
def run_inference_and_collect_results(model, dataloader, device, fp16=False):
    all_predictions = []
    all_labels = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation needed during inference
        for i, batch in enumerate(dataloader):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)
            # If using half-precision (fp16), enable automatic mixed precision.
            context_manager = torch.autocast(device_type=device, dtype=torch.bfloat16) if fp16 else nullcontext()
            with context_manager:
                outputs = model.predict(inputs, attention_mask, metadata).view(-1)  # Predict outputs
            all_predictions.extend(outputs.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    results = pd.DataFrame({'Predicted Outputs': all_predictions, 'True Labels': all_labels})  # Store results in a DataFrame
    return results

# Computes accuracy, precision, recall, and F1 score for model evaluation
def compute_metrics(results):
    predictions = results['Predicted Outputs']
    labels = results['True Labels']
    accuracy = accuracy_score(predictions, labels)
    precision = precision_score(predictions, labels, zero_division=0)
    recall = recall_score(predictions, labels, zero_division=0)
    f1 = f1_score(predictions, labels, zero_division=0)
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    return metrics

# Evaluates the model by testing and calculating metrics for training and validation sets
def evaluate_model(model, epoch, train_dataloader, val_dataloader, device):
    avg_train_loss = test_model(model, train_dataloader, device)  # Test model on training data
    avg_val_loss = test_model(model, val_dataloader, device)  # Test model on validation data
    results = run_inference_and_collect_results(model, val_dataloader, device)  # Collect validation results
    metrics = compute_metrics(results)  # Compute performance metrics
    log_message = f"Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {metrics['Accuracy'] * 100:.2f}, Precision: {metrics['Precision'] * 100:.2f}%, Recall: {metrics['Recall'] * 100:.2f}%, F1: {metrics['F1 Score']:.2f}"
    return log_message, metrics

def test_model_adversarial(model, dataloader, device, fp16=False):
    """
    Tests the model on adversarially generated examples and calculates average losses.
    Pulls tokenized adversarial examples from the dataset.
    """
    results = {
        "original_loss": 0,
        "similar_loss": 0,
        "case_symbols_loss": 0,
        "unicode_loss": 0,
    }
    total_batches = 0

    with torch.no_grad():  # No gradient computation needed during inference
        for batch in dataloader:
            # Load original data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            # Load adversarial examples
            similar_ids = batch["similar_input_ids"].to(device)
            similar_mask = batch["similar_attention_mask"].to(device)

            case_symbols_ids = batch["case_symbols_input_ids"].to(device)
            case_symbols_mask = batch["case_symbols_attention_mask"].to(device)

            unicode_ids = batch["unicode_input_ids"].to(device)
            unicode_mask = batch["unicode_attention_mask"].to(device)

            # Context manager for mixed precision (if fp16 is enabled)
            context_manager = torch.autocast(device_type=device, dtype=torch.bfloat16) if fp16 else nullcontext()

            with context_manager:
                # Calculate loss for original inputs
                _, original_loss = model(input_ids, attention_mask, metadata, labels)
                results["original_loss"] += original_loss.item()

                # Calculate loss for similar characters adversarial examples
                _, similar_loss = model(similar_ids, similar_mask, metadata, labels)
                results["similar_loss"] += similar_loss.item()

                # Calculate loss for case symbols adversarial examples
                _, case_symbols_loss = model(case_symbols_ids, case_symbols_mask, metadata, labels)
                results["case_symbols_loss"] += case_symbols_loss.item()

                # Calculate loss for unicode adversarial examples
                _, unicode_loss = model(unicode_ids, unicode_mask, metadata, labels)
                results["unicode_loss"] += unicode_loss.item()

            total_batches += 1

    # Compute average losses
    avg_results = {key: value / total_batches for key, value in results.items()}
    return avg_results

# Runs inference on the model and collects results for evaluation
def run_inference_and_collect_results_adversarial(model, dataloader, device, fp16=False):
    """
    Runs inference and collects results for original and adversarial inputs.
    """
    results = {
        "original": {"predictions": [], "labels": []},
        "similar": {"predictions": [], "labels": []},
        "case symbols": {"predictions": [], "labels": []},
        "unicode": {"predictions": [], "labels": []},
    }

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation needed during inference
        for batch in dataloader:
            # Shared metadata and labels
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            # Original inputs
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Adversarial examples
            similar_ids = batch["similar_input_ids"].to(device)
            similar_mask = batch["similar_attention_mask"].to(device)

            case_symbols_ids = batch["case_symbols_input_ids"].to(device)
            case_symbols_mask = batch["case_symbols_attention_mask"].to(device)

            unicode_ids = batch["unicode_input_ids"].to(device)
            unicode_mask = batch["unicode_attention_mask"].to(device)

            # If using half-precision (fp16), enable automatic mixed precision
            context_manager = torch.autocast(device_type=device, dtype=torch.bfloat16) if fp16 else nullcontext()

            with context_manager:
                # Original inputs
                original_outputs = model.predict(inputs, attention_mask, metadata).view(-1)
                results["original"]["predictions"].extend(original_outputs.cpu().numpy())
                results["original"]["labels"].extend(labels.cpu().numpy())

                # Similar characters
                similar_outputs = model.predict(similar_ids, similar_mask, metadata).view(-1)
                results["similar"]["predictions"].extend(similar_outputs.cpu().numpy())
                results["similar"]["labels"].extend(labels.cpu().numpy())

                # Case symbols
                case_symbols_outputs = model.predict(case_symbols_ids, case_symbols_mask, metadata).view(-1)
                results["case symbols"]["predictions"].extend(case_symbols_outputs.cpu().numpy())
                results["case symbols"]["labels"].extend(labels.cpu().numpy())

                # Unicode replacements
                unicode_outputs = model.predict(unicode_ids, unicode_mask, metadata).view(-1)
                results["unicode"]["predictions"].extend(unicode_outputs.cpu().numpy())
                results["unicode"]["labels"].extend(labels.cpu().numpy())

    return results

# Trains the model using gradient descent with optional early stopping
def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, device, log_interval=10, fp16=True, scheduler_config=None, early_stopping=False, patience=5, improvement_threshold=0.001):
    best_val_loss = float('inf')
    steps_without_improvement = 0

    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            avg_train_loss = 0.0
            model.train()
            max_steps = len(train_dataloader)

            for step, batch in enumerate(train_dataloader):
                # Log and evaluate at specified intervals
                if (step % log_interval == 0 or step == max_steps - 1) and step > 0:
                    model.eval()  # Set model to evaluation mode
                    with torch.no_grad():
                        avg_val_loss = test_model(model, val_dataloader, device)  # Test model on validation data
                        results = run_inference_and_collect_results(model, val_dataloader, device)  # Collect validation results
                        metrics = compute_metrics(results)  # Compute performance metrics
                        log_message = f"Epoch {epoch}, Average Training Loss: {avg_train_loss / log_interval:.4f}, Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {metrics['Accuracy'] * 100:.2f}%, Precision: {metrics['Precision'] * 100:.2f}%, Recall: {metrics['Recall'] * 100:.2f}%, F1: {metrics['F1 Score']:.2f}"
                        print(log_message)

                        # Early stopping logic if enabled
                        if early_stopping:
                            if avg_val_loss < best_val_loss - improvement_threshold:
                                best_val_loss = avg_val_loss
                                steps_without_improvement = 0  # Reset patience counter
                            else:
                                steps_without_improvement += 1
                                print(f"No significant improvement in validation loss for {steps_without_improvement} step(s).")

                            if steps_without_improvement >= patience:
                                print(f"Early stopping at epoch {epoch}: Validation loss did not improve for {patience} steps.")
                                return model

                    avg_train_loss = 0.0
                    model.train()  # Set model back to training mode if not early stopping

                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device)

                # If using half-precision (fp16), enable automatic mixed precision.
                context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()
                with context_manager:
                    _, loss = model(inputs, attention_mask, metadata, labels)  # Forward pass through the model

                optimizer.zero_grad(set_to_none=True)  # Clear gradients
                loss.backward()  # Backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
                if scheduler_config is not None:
                    lr = get_lr(step, warmup_steps=scheduler_config['warmup_steps'], max_steps=scheduler_config['max_steps'], min_lr=scheduler_config['min_lr'], max_lr=scheduler_config['max_lr'])
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                optimizer.step()  # Update model parameters
                epoch_loss += loss.detach()  # Accumulate loss
                avg_train_loss += loss.detach()

        return model
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Returning the model as-is.")
        return model

def urls_to_string(url_series):
    return " ".join(url_series)

# Function to find the appropriate bin for a given length
def find_bin(length, bins):
    for bin_size in bins[:-1]:  # Exclude the 'beyond' bin from this loop
        if length <= bin_size:
            return bin_size
    return 'beyond'  # If the length is larger than the largest bin, place it in 'beyond'


def add_metadata(example, columns):
    # Extract the values to normalize from the example
    metadata_values = [example[col] for col in columns]

    # Convert the normalized values to a tensor
    metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32)

    # Add the metadata tensor to the example
    example['metadata'] = metadata_tensor
    return example

def plot_confusion_matrix(model, dataloader, device):
    """
    Computes and plots the confusion matrix for a given model and dataset.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation dataset.
        device (torch.device): Device on which to perform computations.
    """
    class_names = ["vulnerable", "safe"]  # Class 0 = vulnerable, Class 1 = safe
    
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device) if "metadata" in batch else None
            labels = batch["label"].to(device)
            
            # Get model predictions
            if metadata is not None:
                outputs = model.predict(inputs, attention_mask, metadata)
            else:
                outputs = model.predict(inputs, attention_mask)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Flatten lists to 1D arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # Ensure correct label order (vulnerable=0, safe=1)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def calculate_auc_roc(model, dataloader, device):
    """
    Calculate AUC and plot the ROC curve for a model.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run inference on.
        
    Returns:
        float: AUC score.
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch.get("metadata", None)
            if metadata is not None:
                metadata = metadata.to(device)
            labels = batch["label"].to(device)
            
            # Get model predictions
            probs = model.predict(inputs, attention_mask, metadata)
            probs = probs.float()  # Ensure probabilities are float type for ROC
            
            # Append to lists
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # Calculate AUC score
    auc_score = roc_auc_score(all_labels, all_probs)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return auc_score

def multiples_of_two(start, end):
    multiples = []
    power = 1
    while power <= end:
        if power >= start:
            multiples.append(power)
        power *= 2
    return multiples