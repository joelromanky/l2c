from typing import Any
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import random

from .modules import Model

class L2C:
    def __init__(
        self,
        oracle: nn.Module,
        continuous_features: list[str],
        categorical_features: list[str],
        immutable_features: list[str],
        monotonic_increasing_column_names: list[str] | None = None,
        correlated_features: list[tuple[str, str, float]] | None = None,
        categories: list[Any] | None = None,
        HP: int = 50,
        HS: int = 50,
        tau: float = 0.2,
        epochs: int = 100,
        batch_size: int = 50,
        learning_rate: float = 1e-4,
        train_L: int = 1,
        test_L: int = 1000,
        train_smoothing: float = 1.0,
        test_smoothing: float = 0.0,
        num_samples: int = 100,
        max_iter: int = 10000,
        sparsity_threshold: float = 1.0,
        device: str = "cpu"
    ):
        self.oracle = oracle.to(device)
        self.oracle.eval()
        self.continuous = continuous_features
        self.categorical = categorical_features
        self.immutable = immutable_features
        self.categories = categories
        
        self.monotonic_names = monotonic_increasing_column_names or []
        self.correlated_names = correlated_features or []
        
        self.HP = HP
        self.HS = HS
        self.tau = tau
        self.device = device

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_L = train_L
        self.test_L = test_L
        self.train_smoothing = train_smoothing
        self.test_smoothing = test_smoothing

        self.num_samples = num_samples
        self.max_iter = max_iter
        self.sparsity_threshold = sparsity_threshold
        
    def _prepare_data_metadata(self, df: pd.DataFrame):
        self.ordered_columns = self.continuous + self.categorical
        col_to_idx = {name: i for i, name in enumerate(self.ordered_columns)}
        
        transformers = []
        if self.continuous:
            transformers.append(("num", MinMaxScaler(), self.continuous))
        if self.categorical:
            if self.categories is not None:
                typed_categories = [
                    np.asarray(cats, dtype=df[col].dtype)
                    for cats, col in zip(self.categories, self.categorical)
                ]
                ohe = OneHotEncoder(categories=typed_categories, sparse_output=False, handle_unknown="ignore")
            else:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            transformers.append(("cat", ohe, self.categorical))
            
        self.preprocessor = ColumnTransformer(transformers, remainder='drop')
        self.preprocessor.fit(df)
        
        metadata = {
            'num_features': len(self.ordered_columns),
            'feature_indices': [],
            'feature_types': [],
            'feature_ranges': [],
            'monotonic_indices': [],
            'correlated_features': []
        }
        
        self.mask_locations = []
        current_idx = 0
        
        for i, col in enumerate(self.ordered_columns):
            if col in self.continuous:
                metadata['feature_indices'].append((current_idx, current_idx + 1))
                metadata['feature_types'].append('cont')
                metadata['feature_ranges'].append(df[col].max() - df[col].min())
                current_idx += 1
            else:
                cat_encoder = self.preprocessor.named_transformers_['cat']
                # The index within the categorical subset is i - len(self.continuous)
                width = len(cat_encoder.categories_[i - len(self.continuous)])
                metadata['feature_indices'].append((current_idx, current_idx + width))
                metadata['feature_types'].append('cat')
                metadata['feature_ranges'].append(None)
                current_idx += width
                
            if col in self.immutable:
                self.mask_locations.append(i)
            if col in self.monotonic_names:
                metadata['monotonic_indices'].append(i)
                
        # Map (cause_str, effect_str, delta) -> (cause_idx, effect_idx, delta)
        for cause, effect, delta in self.correlated_names:
            if cause in col_to_idx and effect in col_to_idx:
                metadata['correlated_features'].append(
                    (col_to_idx[cause], col_to_idx[effect], delta)
                )

        self.metadata = metadata
        self.input_dim = current_idx

    def fit(self, X_train: pd.DataFrame):
        self._prepare_data_metadata(X_train)
        
        X_trans = self.preprocessor.transform(X_train[self.ordered_columns])
        dataset = TensorDataset(torch.FloatTensor(X_trans))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = Model(
            D=self.input_dim, 
            HP=self.HP, 
            HS=self.HS, 
            metadata=self.metadata, 
            tau=self.tau
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        kwargs = {'mask_locations': self.mask_locations, 'L': self.train_L, 'smoothing': self.train_smoothing}
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                # --- Inside the fit() loop ---
                truth_x, xcf, P, W = self.model(x, truth_x=x, **kwargs)
                
                # Validity Loss
                y_hat = self.oracle(xcf) 
                
                if y_hat.shape[-1] > 1:
                    # Multi-column output (e.g., [Batch, 2])
                    with torch.no_grad():
                        orig_y = self.oracle(x).argmax(dim=-1)
                    
                    # Target the opposite class (Assuming binary 0 and 1)
                    target_y = 1 - orig_y 
                    
                    # Extract the probability the generator assigned to the TARGET class
                    y_hat_target = y_hat.gather(1, target_y.unsqueeze(-1)).squeeze(-1)
                else:
                    # Single-column output [Batch, 1]
                    with torch.no_grad():
                        orig_y = (self.oracle(x) >= 0.5).float()
                    target_y = 1.0 - orig_y
                    y_hat_target = y_hat.flatten()
                    target_y = target_y.flatten()
                
                # Push the probability of the TARGET class to 1.0 (clamped for CUDA BCE stability)
                y_hat_target = torch.clamp(y_hat_target, min=1e-7, max=1.0 - 1e-7)
                loss = nn.functional.binary_cross_entropy(y_hat_target, torch.ones_like(y_hat_target))
                
                loss.backward()
                optimizer.step()

    def generate(self, query: pd.DataFrame) -> pd.DataFrame:
        self.model.eval()
        
        # 1. Prepare Query Tensor
        x_np = self.preprocessor.transform(query[self.ordered_columns])
        x = torch.FloatTensor(x_np).to(self.device)
        
        # Get the original prediction class to flip
        with torch.no_grad():
            orig_preds = self.oracle(x)
            # Assuming output is either [Batch, Classes] or [Batch, 1]
            if orig_preds.shape[-1] > 1:
                orig_y = orig_preds.argmax(-1).item()
            else:
                orig_y = (orig_preds >= 0.5).int().item()
        
        kwargs = {'mask_locations': self.mask_locations, 'L': self.test_L, 'smoothing': self.test_smoothing}
        
        valid_counters = []
        cnt_sample = 0
        cnt_iter = 0
        
        # 2. Aggressive Accumulation Loop (Faithful to original `infer`)
        with torch.no_grad():
            while cnt_sample < self.num_samples and cnt_iter < self.max_iter:
                _, xcf, probs, _ = self.model(x, truth_x=x, **kwargs)
                
                # Inside generate() while loop
                y_preds = self.oracle(xcf)
                if y_preds.shape[-1] > 1:
                    yhat = y_preds.argmax(-1).int() # Explicit cast
                else:
                    yhat = (y_preds >= 0.5).int().flatten()
                
                # Sparsity Filter
                if self.sparsity_threshold < 1.0:
                    sparsity_indices = torch.where(probs.mean(-1) <= self.sparsity_threshold)[0].tolist()
                else:
                    sparsity_indices = list(range(probs.size(0)))
                    
                # Validity Filter (must change class)
                validity_indices = torch.where(yhat != orig_y)[0].tolist()
                
                # Intersection
                indices = list(set(sparsity_indices) & set(validity_indices))
                
                if len(indices) > 0:
                    selected = xcf[indices, :]
                    valid_counters.append(selected)
                    cnt_sample += selected.size(0)
                    
                cnt_iter += 1
                
        if cnt_sample == 0:
            return pd.DataFrame()
            
        # 3. Concatenate and Handle Padding
        xcfs_tensor = torch.cat(valid_counters, dim=0)
        
        if xcfs_tensor.shape[0] < self.num_samples:
            # Original code pads by random sampling if it couldn't find enough unique ones
            cnt = xcfs_tensor.shape[0]
            import random
            selected_idx = random.choices(range(cnt), k = self.num_samples - cnt)
            xcfs_tensor = torch.cat((xcfs_tensor, xcfs_tensor[selected_idx, :]), dim=0)
        else:
            # Truncate to exact number requested
            xcfs_tensor = xcfs_tensor[:self.num_samples, :]
            
        xcfs_np = xcfs_tensor.cpu().numpy()
        
        # 4. Inverse Transform back to Pandas
        cf_dict_list = []
        for row in xcfs_np:
            row_dict = {}
            idx = 0
            if self.continuous:
                num_len = len(self.continuous)
                cont_vals = self.preprocessor.named_transformers_['num'].inverse_transform(row[idx:idx+num_len].reshape(1, -1))
                for i, col in enumerate(self.continuous):
                    row_dict[col] = cont_vals[0, i]
                idx += num_len
            
            if self.categorical:
                cat_encoder = self.preprocessor.named_transformers_['cat']
                cat_len = sum([len(cats) for cats in cat_encoder.categories_])
                cat_vals = cat_encoder.inverse_transform(row[idx:idx+cat_len].reshape(1, -1))
                for i, col in enumerate(self.categorical):
                    row_dict[col] = cat_vals[0, i]
            cf_dict_list.append(row_dict)
            
        cf_df = pd.DataFrame(cf_dict_list)
        
        # 5. Enforce strict immutability post-generation
        for col in self.immutable:
            cf_df[col] = query.iloc[0][col]
            
        return cf_df[query.columns]
