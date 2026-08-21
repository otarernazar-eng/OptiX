import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from torch.utils.tensorboard import SummaryWriter

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_max = -np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            print(f'Validation metric increased ({self.val_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_max = val_metric

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, optimizer, scheduler, device='cuda', epoch_callback=None):
        self.config = config
        self.epoch_callback = epoch_callback
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Training params
        self.epochs = config['training'].get('epochs', 50)
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.save_dir = Path(config['model'].get('save_dir', 'models/'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.save_dir / 'best.pth'
        
        # Loss function selection
        loss_type = config['training'].get('loss_type', 'ce').lower()
        if loss_type == 'focal':
            gamma = config['training'].get('focal_gamma', 2.0)
            alpha = config['training'].get('focal_alpha', 0.25)
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha)
        elif loss_type == 'weighted_ce':
            weights = config['training'].get('class_weights', None)
            if weights is not None:
                weights = torch.tensor(weights, dtype=torch.float).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Early Stopping
        patience = config['training'].get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=str(self.best_model_path)
        )
        
        # Logging
        self.tb_writer = None
        if not HAS_MLFLOW:
            self.tb_writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        
        self.csv_log_path = self.save_dir / 'metrics.csv'
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'val_f1', 'val_sens', 'val_spec'])

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        
        num_classes = self.config['model'].get('num_classes', 2)
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            f1 = f1_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Multiclass fallback
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            f1 = f1_score(y_true, y_pred, average='weighted')
            sens = 0.0
            spec = 0.0
            
        return acc, auc, f1, sens, spec

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                running_loss += loss.item() * inputs.size(0)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        acc, auc, f1, sens, spec = self._calculate_metrics(
            np.array(all_targets), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        
        return epoch_loss, acc, auc, f1, sens, spec

    def fit(self):
        best_metrics = {}
        
        if HAS_MLFLOW:
            mlflow.log_params(self.config['training'])
        
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_acc, val_auc, val_f1, val_sens, val_spec = self.validate()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Sens: {val_sens:.4f} | Val Spec: {val_spec:.4f}")
            
            # CSV Logging
            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_acc, val_auc, val_f1, val_sens, val_spec])
                
            # MLflow or TensorBoard Logging
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            if HAS_MLFLOW:
                mlflow.log_metrics(metrics, step=epoch)
            elif self.tb_writer:
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(k, v, epoch)
                    
            if self.epoch_callback:
                self.epoch_callback(epoch, metrics)
            
            # Save periodic checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            
            # Early Stopping based on Validation AUC
            self.early_stopping(val_auc, self.model)
            
            if self.early_stopping.best_score == val_auc:
                best_metrics = metrics
            
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break
                
        if self.tb_writer:
            self.tb_writer.close()
            
        print(f"\nTraining completed. Best Validation AUC: {best_metrics.get('val_auc', 0):.4f}")
        return {
            'best_metrics': best_metrics,
            'model_path': str(self.best_model_path)
        }
