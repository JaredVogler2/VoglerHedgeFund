# ml_trading_models.py

"""
ML Trading System - Ensemble Models with GPU Acceleration
Professional ensemble learning with XGBoost, LightGBM, CatBoost, and Deep Learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit, PurgedGroupTimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATASET CLASSES
# =============================================================================

class TradingDataset(Dataset):
    """PyTorch dataset for trading data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 20):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Return sequence of features and corresponding target
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return X, y

# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class AttentionLSTM(nn.Module):
    """LSTM with multi-head attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_heads: int = 8, 
                 dropout: float = 0.2, output_dim: int = 5):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, 
                           bidirectional=True)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, 
                                              dropout=dropout, batch_first=True)
        
        # Residual connection
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        # Take the last timestep
        out = lstm_out[:, -1, :]
        
        # Output layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Output: [direction_prob, return_1d, return_5d, return_10d, volatility]
        direction = self.sigmoid(out[:, 0:1])
        returns = out[:, 1:4]
        volatility = torch.abs(out[:, 4:5])
        
        return torch.cat([direction, returns, volatility], dim=1)

class CNNLSTM(nn.Module):
    """CNN-LSTM for pattern extraction and sequence modeling"""
    
    def __init__(self, input_dim: int, num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7], lstm_hidden: int = 128,
                 dropout: float = 0.2, output_dim: int = 5):
        super(CNNLSTM, self).__init__()
        
        # Multi-scale CNN layers
        self.convs = nn.ModuleList()
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(input_dim, num_filter, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
        
        # Calculate total CNN output channels
        total_filters = sum(num_filters)
        
        # LSTM layer
        self.lstm = nn.LSTM(total_filters, lstm_hidden, 2, 
                           batch_first=True, dropout=dropout)
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        # Transpose for CNN: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        x = torch.cat(conv_outputs, dim=1)
        
        # Transpose back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last timestep
        out = lstm_out[:, -1, :]
        
        # Output layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Split outputs
        direction = self.sigmoid(out[:, 0:1])
        returns = out[:, 1:4]
        volatility = torch.abs(out[:, 4:5])
        
        return torch.cat([direction, returns, volatility], dim=1)

class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4, 
                 dropout: float = 0.2, output_dim: int = 5):
        super(TransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Split outputs
        direction = self.sigmoid(out[:, 0:1])
        returns = out[:, 1:4]
        volatility = torch.abs(out[:, 4:5])
        
        return torch.cat([direction, returns, volatility], dim=1)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# =============================================================================
# ENSEMBLE MODEL MANAGER
# =============================================================================

class EnsembleModelManager:
    """Manages ensemble of ML models with GPU acceleration"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.feature_importance = {}
        
    def initialize_models(self, input_dim: int):
        """Initialize all models in the ensemble"""
        logger.info("Initializing ensemble models...")
        
        # XGBoost with GPU
        self.models['xgboost'] = {
            'classifier': None,  # Will be created during training
            'regressor': None,
            'quantile_regressors': {}
        }
        
        # LightGBM with GPU
        self.models['lightgbm'] = {
            'classifier': None,
            'regressor': None,
            'quantile_regressors': {}
        }
        
        # CatBoost with GPU
        self.models['catboost'] = {
            'classifier': None,
            'regressor': None,
            'quantile_regressors': {}
        }
        
        # Deep Learning Models
        self.models['attention_lstm'] = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.2
        ).to(self.device)
        
        self.models['cnn_lstm'] = CNNLSTM(
            input_dim=input_dim,
            num_filters=[64, 128, 256],
            kernel_sizes=[3, 5, 7],
            lstm_hidden=128,
            dropout=0.2
        ).to(self.device)
        
        self.models['transformer'] = TransformerModel(
            input_dim=input_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)
        
        # Initialize scalers
        self.scalers['robust'] = RobustScaler()
        self.scalers['standard'] = StandardScaler()
        
        logger.info(f"Initialized {len(self.models)} model types")
        
    def train_xgboost(self, X_train, y_train, X_val, y_val, task='classification'):
        """Train XGBoost models with GPU acceleration"""
        logger.info(f"Training XGBoost for {task}...")
        
        # Prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Base parameters with GPU
        base_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'random_state': 42
        }
        
        if task == 'classification':
            params = {
                **base_params,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
        else:
            params = {
                **base_params,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }
        
        # Train model
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Store feature importance
        importance = model.get_score(importance_type='gain')
        self.feature_importance[f'xgboost_{task}'] = importance
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, task='classification'):
        """Train LightGBM models with GPU acceleration"""
        logger.info(f"Training LightGBM for {task}...")
        
        # Prepare data
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Base parameters with GPU
        base_params = {
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_leaves': 64,
            'max_depth': 8,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'min_child_samples': 20,
            'verbosity': 1,
            'random_state': 42
        }
        
        if task == 'classification':
            params = {
                **base_params,
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': True
            }
        else:
            params = {
                **base_params,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'dart',
                'drop_rate': 0.1
            }
        
        # Train model with callbacks
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Store feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_names = [f'f_{i}' for i in range(len(importance))]
        self.feature_importance[f'lightgbm_{task}'] = dict(zip(feature_names, importance))
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val, y_val, task='classification'):
        """Train CatBoost models with GPU acceleration"""
        logger.info(f"Training CatBoost for {task}...")
        
        # Prepare data
        if task == 'classification':
            model = cb.CatBoostClassifier(
                iterations=1000,
                depth=8,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                task_type='GPU',
                devices='0',
                loss_function='Logloss',
                eval_metric='AUC',
                early_stopping_rounds=50,
                random_state=42,
                verbose=100
            )
        else:
            model = cb.CatBoostRegressor(
                iterations=1000,
                depth=8,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                task_type='GPU',
                devices='0',
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=50,
                random_state=42,
                verbose=100
            )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            plot=False
        )
        
        # Store feature importance
        importance = model.get_feature_importance()
        feature_names = [f'f_{i}' for i in range(len(importance))]
        self.feature_importance[f'catboost_{task}'] = dict(zip(feature_names, importance))
        
        return model
    
    def train_deep_learning_model(self, model_name: str, train_loader: DataLoader,
                                 val_loader: DataLoader, epochs: int = 100):
        """Train deep learning models"""
        logger.info(f"Training {model_name}...")
        
        model = self.models[model_name]
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Loss functions
        direction_loss = nn.BCELoss()
        return_loss = nn.MSELoss()
        volatility_loss = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_features)
                
                # Calculate losses
                loss_dir = direction_loss(outputs[:, 0], batch_targets[:, 0])
                loss_ret = return_loss(outputs[:, 1:4], batch_targets[:, 1:4])
                loss_vol = volatility_loss(outputs[:, 4], batch_targets[:, 4])
                
                # Combined loss with weights
                total_loss = loss_dir + 0.5 * loss_ret + 0.3 * loss_vol
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    
                    # Calculate losses
                    loss_dir = direction_loss(outputs[:, 0], batch_targets[:, 0])
                    loss_ret = return_loss(outputs[:, 1:4], batch_targets[:, 1:4])
                    loss_vol = volatility_loss(outputs[:, 4], batch_targets[:, 4])
                    
                    total_loss = loss_dir + 0.5 * loss_ret + 0.3 * loss_vol
                    val_losses.append(total_loss.item())
                    
                    # Calculate accuracy for direction
                    predictions = (outputs[:, 0] > 0.5).float()
                    accuracy = (predictions == batch_targets[:, 0]).float().mean()
                    val_accuracies.append(accuracy.item())
            
            # Calculate epoch metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_name}.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_name}.pth'))
        
        return model
    
    def train_quantile_models(self, X_train, y_train, X_val, y_val, 
                             quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        """Train quantile regression models"""
        logger.info("Training quantile regression models...")
        
        quantile_models = {}
        
        for q in quantiles:
            logger.info(f"Training quantile {q}...")
            
            # XGBoost quantile
            xgb_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'objective': f'reg:quantileerror',
                'quantile_alpha': q,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'eval')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            
            if 'xgboost' not in quantile_models:
                quantile_models['xgboost'] = {}
            quantile_models['xgboost'][q] = xgb_model
            
            # LightGBM quantile
            lgb_params = {
                'device_type': 'gpu',
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'num_leaves': 32,
                'max_depth': 6,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'verbosity': -1
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            lgb_model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(30)]
            )
            
            if 'lightgbm' not in quantile_models:
                quantile_models['lightgbm'] = {}
            quantile_models['lightgbm'][q] = lgb_model
        
        return quantile_models
    
    def calibrate_probabilities(self, models: Dict, X_cal, y_cal):
        """Calibrate model probabilities using isotonic regression"""
        logger.info("Calibrating model probabilities...")
        
        calibrators = {}
        
        for name, model in models.items():
            if 'classifier' in name or name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                # Get uncalibrated probabilities
                if name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_cal).to(self.device)
                        probs = model(X_tensor)[:, 0].cpu().numpy()
                else:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_cal)[:, 1]
                    else:
                        # For XGBoost/LightGBM
                        if isinstance(model, xgb.Booster):
                            dmatrix = xgb.DMatrix(X_cal)
                            probs = model.predict(dmatrix)
                        else:
                            probs = model.predict(X_cal)
                
                # Fit isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probs, y_cal)
                calibrators[name] = calibrator
                
                logger.info(f"Calibrated {name}")
        
        return calibrators
    
    def create_meta_learner(self, base_predictions: np.ndarray, y_train: np.ndarray):
        """Create meta-learner for stacking"""
        logger.info("Training meta-learner...")
        
        # Use XGBoost as meta-learner
        meta_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc'
        }
        
        dtrain = xgb.DMatrix(base_predictions, label=y_train)
        
        meta_model = xgb.train(
            meta_params,
            dtrain,
            num_boost_round=200
        )
        
        return meta_model
    
    def save_models(self, path: str):
        """Save all models and associated objects"""
        logger.info(f"Saving models to {path}...")
        
        # Save tree-based models
        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            if model_type in self.models:
                model_dict = self.models[model_type]
                for key, model in model_dict.items():
                    if model is not None:
                        if model_type == 'xgboost':
                            model.save_model(f"{path}/{model_type}_{key}.json")
                        elif model_type == 'lightgbm':
                            model.save_model(f"{path}/{model_type}_{key}.txt")
                        elif model_type == 'catboost':
                            model.save_model(f"{path}/{model_type}_{key}.cbm")
        
        # Save deep learning models
        for model_name in ['attention_lstm', 'cnn_lstm', 'transformer']:
            if model_name in self.models:
                torch.save(self.models[model_name].state_dict(), 
                          f"{path}/{model_name}.pth")
        
        # Save scalers and calibrators
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        joblib.dump(self.calibrators, f"{path}/calibrators.pkl")
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
        
        logger.info("Models saved successfully")

# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

class TrainingOrchestrator:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, config, feature_engineer, ensemble_manager):
        self.config = config
        self.feature_engineer = feature_engineer
        self.ensemble_manager = ensemble_manager
        
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            target_horizons: List[int] = [1, 5, 10, 21]) -> Dict:
        """Prepare features and multi-horizon targets"""
        logger.info("Preparing training data...")
        
        # Calculate returns for different horizons
        targets = {}
        
        for horizon in target_horizons:
            # Direction (binary classification)
            targets[f'direction_{horizon}d'] = (
                features_df['return_1d'].shift(-horizon) > 0
            ).astype(int)
            
            # Return magnitude (regression)
            targets[f'return_{horizon}d'] = features_df['return_1d'].shift(-horizon)
            
            # Volatility forecast
            targets[f'volatility_{horizon}d'] = features_df['return_1d'].rolling(
                horizon
            ).std().shift(-horizon)
        
        # Combine targets
        target_df = pd.DataFrame(targets)
        
        # Align features and targets
        valid_idx = target_df.notna().all(axis=1)
        features_clean = features_df[valid_idx].copy()
        targets_clean = target_df[valid_idx].copy()
        
        # Create sample weights (time decay)
        days_ago = (features_clean.index.max() - features_clean.index).days
        sample_weights = np.exp(-days_ago / self.config.SAMPLE_WEIGHT_HALFLIFE)
        
        return {
            'features': features_clean,
            'targets': targets_clean,
            'sample_weights': sample_weights
        }
    
    def create_purged_splits(self, X, y, n_splits=5, purge_days=5):
        """Create purged time series splits to prevent data leakage"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        test_size = n_samples // (n_splits + 1)
        
        for i in range(n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            # Purge gap to prevent leakage
            train_end = test_start - purge_days
            
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            
            yield train_idx, test_idx
    
    def train_ensemble(self, features_df: pd.DataFrame, symbol: str) -> Dict:
        """Train complete ensemble for a symbol"""
        logger.info(f"Training ensemble for {symbol}...")
        
        # Prepare data
        data_dict = self.prepare_training_data(features_df)
        
        X = data_dict['features'].values
        y_direction = data_dict['targets']['direction_5d'].values
        y_return = data_dict['targets']['return_5d'].values
        weights = data_dict['sample_weights']
        
        # Scale features
        X_scaled = self.ensemble_manager.scalers['robust'].fit_transform(X)
        
        # Initialize models
        self.ensemble_manager.initialize_models(X.shape[1])
        
        # Purged cross-validation
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(
            self.create_purged_splits(X_scaled, y_direction, n_splits=5)
        ):
            logger.info(f"Training fold {fold + 1}...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train_dir, y_val_dir = y_direction[train_idx], y_direction[val_idx]
            y_train_ret, y_val_ret = y_return[train_idx], y_return[val_idx]
            w_train = weights[train_idx]
            
            # Train XGBoost
            xgb_clf = self.ensemble_manager.train_xgboost(
                X_train, y_train_dir, X_val, y_val_dir, 'classification'
            )
            xgb_reg = self.ensemble_manager.train_xgboost(
                X_train, y_train_ret, X_val, y_val_ret, 'regression'
            )
            
            # Train LightGBM
            lgb_clf = self.ensemble_manager.train_lightgbm(
                X_train, y_train_dir, X_val, y_val_dir, 'classification'
            )
            lgb_reg = self.ensemble_manager.train_lightgbm(
                X_train, y_train_ret, X_val, y_val_ret, 'regression'
            )
            
            # Train CatBoost
            cb_clf = self.ensemble_manager.train_catboost(
                X_train, y_train_dir, X_val, y_val_dir, 'classification'
            )
            cb_reg = self.ensemble_manager.train_catboost(
                X_train, y_train_ret, X_val, y_val_ret, 'regression'
            )
            
            # Store best models (simplified for this example)
            if fold == 0:
                self.ensemble_manager.models['xgboost']['classifier'] = xgb_clf
                self.ensemble_manager.models['xgboost']['regressor'] = xgb_reg
                self.ensemble_manager.models['lightgbm']['classifier'] = lgb_clf
                self.ensemble_manager.models['lightgbm']['regressor'] = lgb_reg
                self.ensemble_manager.models['catboost']['classifier'] = cb_clf
                self.ensemble_manager.models['catboost']['regressor'] = cb_reg
            
            # Validate
            dval = xgb.DMatrix(X_val)
            xgb_pred = xgb_clf.predict(dval)
            accuracy = accuracy_score(y_val_dir, xgb_pred > 0.5)
            cv_scores.append(accuracy)
            
            logger.info(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
        
        # Train deep learning models (simplified)
        # Create datasets
        sequence_length = 20
        if len(X_scaled) > sequence_length:
            # Prepare sequential data
            train_dataset = TradingDataset(
                X_scaled[:-100], 
                np.column_stack([y_direction[:-100], 
                               y_return[:-100], 
                               y_return[:-100],  # Placeholder for multi-horizon
                               y_return[:-100],  # Placeholder
                               np.abs(y_return[:-100])])  # Volatility proxy
            )
            val_dataset = TradingDataset(
                X_scaled[-100:], 
                np.column_stack([y_direction[-100:], 
                               y_return[-100:],
                               y_return[-100:],
                               y_return[-100:],
                               np.abs(y_return[-100:])])
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train each DL model
            for model_name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                self.ensemble_manager.train_deep_learning_model(
                    model_name, train_loader, val_loader, epochs=50
                )
        
        # Train quantile models
        quantile_models = self.ensemble_manager.train_quantile_models(
            X_scaled[train_idx], y_train_ret, X_scaled[val_idx], y_val_ret
        )
        
        for model_type, q_models in quantile_models.items():
            self.ensemble_manager.models[model_type]['quantile_regressors'] = q_models
        
        # Calibrate probabilities
        cal_size = min(1000, len(X_scaled) // 5)
        X_cal = X_scaled[-cal_size:]
        y_cal = y_direction[-cal_size:]
        
        models_to_calibrate = {
            'xgboost': self.ensemble_manager.models['xgboost']['classifier'],
            'lightgbm': self.ensemble_manager.models['lightgbm']['classifier'],
            'catboost': self.ensemble_manager.models['catboost']['classifier']
        }
        
        self.ensemble_manager.calibrators = self.ensemble_manager.calibrate_probabilities(
            models_to_calibrate, X_cal, y_cal
        )
        
        # Create meta-learner
        # Get base model predictions for meta-learning
        base_preds = []
        
        # XGBoost prediction
        dmatrix = xgb.DMatrix(X_scaled[-cal_size:])
        base_preds.append(self.ensemble_manager.models['xgboost']['classifier'].predict(dmatrix))
        
        # LightGBM prediction
        base_preds.append(self.ensemble_manager.models['lightgbm']['classifier'].predict(X_scaled[-cal_size:]))
        
        # CatBoost prediction
        base_preds.append(self.ensemble_manager.models['catboost']['classifier'].predict_proba(X_scaled[-cal_size:])[:, 1])
        
        base_predictions = np.column_stack(base_preds)
        meta_model = self.ensemble_manager.create_meta_learner(base_predictions, y_cal)
        self.ensemble_manager.models['meta_learner'] = meta_model
        
        logger.info(f"Ensemble training complete. Average CV accuracy: {np.mean(cv_scores):.4f}")
        
        return {
            'cv_scores': cv_scores,
            'feature_importance': self.ensemble_manager.feature_importance,
            'models': self.ensemble_manager.models
        }

# =============================================================================
# PREDICTION ENGINE
# =============================================================================

class PredictionEngine:
    """Generates predictions from the ensemble"""
    
    def __init__(self, ensemble_manager):
        self.ensemble_manager = ensemble_manager
        
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble predictions"""
        # Scale features
        X = features.values
        X_scaled = self.ensemble_manager.scalers['robust'].transform(X)
        
        predictions = {}
        
        # XGBoost predictions
        dmatrix = xgb.DMatrix(X_scaled)
        predictions['xgb_direction'] = self.ensemble_manager.models['xgboost']['classifier'].predict(dmatrix)
        predictions['xgb_return'] = self.ensemble_manager.models['xgboost']['regressor'].predict(dmatrix)
        
        # LightGBM predictions
        predictions['lgb_direction'] = self.ensemble_manager.models['lightgbm']['classifier'].predict(X_scaled)
        predictions['lgb_return'] = self.ensemble_manager.models['lightgbm']['regressor'].predict(X_scaled)
        
        # CatBoost predictions
        predictions['cb_direction'] = self.ensemble_manager.models['catboost']['classifier'].predict_proba(X_scaled)[:, 1]
        predictions['cb_return'] = self.ensemble_manager.models['catboost']['regressor'].predict(X_scaled)
        
        # Deep learning predictions (if models are trained)
        if len(X_scaled) >= 20:
            # Prepare sequences
            sequences = []
            for i in range(len(X_scaled) - 20 + 1):
                sequences.append(X_scaled[i:i+20])
            
            X_seq = torch.FloatTensor(np.array(sequences)).to(self.ensemble_manager.device)
            
            # Get predictions from each model
            for model_name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                model = self.ensemble_manager.models[model_name]
                model.eval()
                with torch.no_grad():
                    outputs = model(X_seq).cpu().numpy()
                    predictions[f'{model_name}_direction'] = outputs[:, 0]
                    predictions[f'{model_name}_return'] = outputs[:, 1]
        
        # Quantile predictions
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            predictions[f'xgb_q{int(q*100)}'] = self.ensemble_manager.models['xgboost']['quantile_regressors'][q].predict(dmatrix)
            predictions[f'lgb_q{int(q*100)}'] = self.ensemble_manager.models['lightgbm']['quantile_regressors'][q].predict(X_scaled)
        
        # Apply calibration
        for name, calibrator in self.ensemble_manager.calibrators.items():
            if f'{name}_direction' in predictions:
                predictions[f'{name}_direction_calibrated'] = calibrator.transform(predictions[f'{name}_direction'])
        
        # Meta-learner ensemble
        base_features = np.column_stack([
            predictions['xgb_direction'],
            predictions['lgb_direction'],
            predictions['cb_direction']
        ])
        meta_dmatrix = xgb.DMatrix(base_features)
        predictions['ensemble_direction'] = self.ensemble_manager.models['meta_learner'].predict(meta_dmatrix)
        
        # Calculate consensus
        direction_preds = [predictions[k] for k in predictions if 'direction' in k and 'calibrated' not in k]
        predictions['consensus_direction'] = np.mean(direction_preds, axis=0)
        predictions['consensus_agreement'] = np.std(direction_preds, axis=0)
        
        return_preds = [predictions[k] for k in predictions if 'return' in k and 'q' not in k]
        predictions['consensus_return'] = np.mean(return_preds, axis=0)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame(predictions, index=features.index[-len(predictions['consensus_direction']):])
        
        return pred_df
        
    def calculate_prediction_confidence(self, predictions: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for predictions"""
        confidence_factors = []
        
        # Model agreement
        direction_cols = [col for col in predictions.columns if 'direction' in col and 'calibrated' in col]
        if direction_cols:
            agreement = predictions[direction_cols].std(axis=1)
            confidence_factors.append(1 - agreement)  # Higher agreement = higher confidence
        
        # Prediction strength
        if 'consensus_direction' in predictions:
            strength = abs(predictions['consensus_direction'] - 0.5) * 2
            confidence_factors.append(strength)
        
        # Quantile spread (tighter spread = higher confidence)
        if 'xgb_q75' in predictions and 'xgb_q25' in predictions:
            iqr = predictions['xgb_q75'] - predictions['xgb_q25']
            normalized_iqr = 1 / (1 + iqr)  # Inverse relationship
            confidence_factors.append(normalized_iqr)
        
        # Combine confidence factors
        if confidence_factors:
            confidence = np.mean(confidence_factors, axis=0)
        else:
            confidence = pd.Series(0.5, index=predictions.index)
        
        return confidence

# Example usage
if __name__ == "__main__":
    # This would be integrated with the main trading system
    logger.info("ML Models module loaded successfully")