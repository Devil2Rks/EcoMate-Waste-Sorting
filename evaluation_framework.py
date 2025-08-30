"""
Comprehensive Evaluation Framework for Water Body Classification
Includes 5-fold cross-validation, statistical tests, ablation studies

Author: B.Tech Research Team
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    accuracy_score, precision_recall_fscore_support
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from temporal_fusion_model import WaterBodyClassificationModel, create_water_body_model
from baseline_models import create_baseline_models, BaselineTrainer
from training_pipeline import WaterBodyTrainer, create_training_config


class CrossValidationEvaluator:
    """
    5-fold cross-validation evaluator for rigorous model assessment
    """
    
    def __init__(self, config: Dict, save_path: str):
        self.config = config
        self.save_path = save_path
        self.results = {}
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
    def prepare_cv_splits(self, dataset: Any, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Prepare stratified cross-validation splits
        
        Args:
            dataset: Dataset object
            n_splits: Number of CV folds
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        
        # Extract labels for stratification
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample['mask'].numpy()
            
            # Get dominant water body class
            water_pixels = mask > 0
            if water_pixels.any():
                dominant_class = stats.mode(mask[water_pixels], keepdims=False)[0]
            else:
                dominant_class = 0  # Background
            
            labels.append(dominant_class)
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(range(len(dataset)), labels))
        
        print(f"Created {n_splits}-fold cross-validation splits")
        print(f"Dataset size: {len(dataset)}")
        
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return splits
    
    def evaluate_model_cv(self, model_creator_fn, dataset, model_name: str) -> Dict[str, Any]:
        """
        Evaluate model using cross-validation
        
        Args:
            model_creator_fn: Function to create model instance
            dataset: Dataset for evaluation
            model_name: Name of the model
            
        Returns:
            Cross-validation results
        """
        
        print(f"\nEvaluating {model_name} with 5-fold cross-validation...")
        
        # Prepare CV splits
        cv_splits = self.prepare_cv_splits(dataset, n_splits=5)
        
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
            print(f"\nFold {fold_idx + 1}/5")
            print("-" * 30)
            
            # Create model for this fold
            model = model_creator_fn()
            
            # Create data loaders for this fold
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=self.config['batch_size'], 
                shuffle=True, num_workers=2
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=self.config['batch_size'], 
                shuffle=False, num_workers=2
            )
            
            data_loaders = {'train': train_loader, 'val': val_loader}
            
            # Train model for this fold
            if 'temporal_fusion' in model_name.lower():
                trainer = WaterBodyTrainer(model, data_loaders, self.config)
                trainer.train(num_epochs=self.config.get('cv_epochs', 15))
                final_metrics = trainer.validate_epoch(0)  # Get final validation metrics
            else:
                trainer = BaselineTrainer(model, data_loaders, model_name, self.config)
                trainer.train(num_epochs=self.config.get('cv_epochs', 15))
                val_loss, val_accuracy = trainer.validate_epoch(0)
                final_metrics = {
                    'total_loss': val_loss,
                    'segmentation_accuracy': val_accuracy,
                    'classification_accuracy': val_accuracy
                }
            
            fold_results.append({
                'fold': fold_idx + 1,
                'val_loss': final_metrics['total_loss'],
                'seg_accuracy': final_metrics['segmentation_accuracy'],
                'cls_accuracy': final_metrics.get('classification_accuracy', 0.0),
                'train_size': len(train_indices),
                'val_size': len(val_indices)
            })
            
            print(f"Fold {fold_idx + 1} completed - Val Loss: {final_metrics['total_loss']:.4f}")
        
        # Compute cross-validation statistics
        cv_stats = self._compute_cv_statistics(fold_results, model_name)
        
        # Save results
        self.results[model_name] = {
            'fold_results': fold_results,
            'cv_statistics': cv_stats
        }
        
        return cv_stats
    
    def _compute_cv_statistics(self, fold_results: List[Dict], model_name: str) -> Dict[str, float]:
        """Compute cross-validation statistics"""
        
        # Extract metrics
        val_losses = [result['val_loss'] for result in fold_results]
        seg_accuracies = [result['seg_accuracy'] for result in fold_results]
        cls_accuracies = [result['cls_accuracy'] for result in fold_results]
        
        # Compute statistics
        stats_dict = {
            'val_loss_mean': np.mean(val_losses),
            'val_loss_std': np.std(val_losses),
            'val_loss_ci_95': 1.96 * np.std(val_losses) / np.sqrt(len(val_losses)),
            
            'seg_acc_mean': np.mean(seg_accuracies),
            'seg_acc_std': np.std(seg_accuracies),
            'seg_acc_ci_95': 1.96 * np.std(seg_accuracies) / np.sqrt(len(seg_accuracies)),
            
            'cls_acc_mean': np.mean(cls_accuracies),
            'cls_acc_std': np.std(cls_accuracies),
            'cls_acc_ci_95': 1.96 * np.std(cls_accuracies) / np.sqrt(len(cls_accuracies))
        }
        
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"Val Loss: {stats_dict['val_loss_mean']:.4f} ± {stats_dict['val_loss_std']:.4f}")
        print(f"Seg Accuracy: {stats_dict['seg_acc_mean']:.4f} ± {stats_dict['seg_acc_std']:.4f}")
        print(f"Cls Accuracy: {stats_dict['cls_acc_mean']:.4f} ± {stats_dict['cls_acc_std']:.4f}")
        
        return stats_dict


class StatisticalTester:
    """
    Statistical significance testing for model comparisons
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def paired_t_test(self, results1: List[float], results2: List[float], 
                     model1_name: str, model2_name: str) -> Dict[str, Any]:
        """
        Perform paired t-test between two models
        
        Args:
            results1: Results from model 1
            results2: Results from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Statistical test results
        """
        
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(results1, results2)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(results1) - np.mean(results2)
        pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Interpretation
        is_significant = p_value < self.alpha
        effect_size_interpretation = self._interpret_effect_size(abs(cohens_d))
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            'model1_mean': np.mean(results1),
            'model2_mean': np.mean(results2),
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'effect_size': effect_size_interpretation,
            'alpha': self.alpha
        }
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_model_comparison(self, cv_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Perform pairwise comparisons between all models
        
        Args:
            cv_results: Cross-validation results for all models
            
        Returns:
            Comprehensive comparison results
        """
        
        model_names = list(cv_results.keys())
        comparison_results = {}
        
        # Extract accuracy results for each model
        model_accuracies = {}
        for name, results in cv_results.items():
            fold_results = results['fold_results']
            accuracies = [fold['seg_accuracy'] for fold in fold_results]
            model_accuracies[name] = accuracies
        
        # Perform pairwise t-tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{model1}_vs_{model2}"
                    
                    test_result = self.paired_t_test(
                        model_accuracies[model1],
                        model_accuracies[model2],
                        model1, model2
                    )
                    
                    comparison_results[comparison_key] = test_result
        
        return comparison_results
    
    def generate_statistical_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate formatted statistical report"""
        
        report = "Statistical Significance Analysis\n"
        report += "=" * 50 + "\n\n"
        
        for comparison_key, results in comparison_results.items():
            model1, model2 = results['model1'], results['model2']
            
            report += f"Comparison: {model1} vs {model2}\n"
            report += f"Mean Accuracy: {results['model1_mean']:.4f} vs {results['model2_mean']:.4f}\n"
            report += f"t-statistic: {results['t_statistic']:.4f}\n"
            report += f"p-value: {results['p_value']:.6f}\n"
            report += f"Cohen's d: {results['cohens_d']:.4f} ({results['effect_size']})\n"
            
            if results['is_significant']:
                better_model = model1 if results['model1_mean'] > results['model2_mean'] else model2
                report += f"Result: {better_model} is significantly better (p < {results['alpha']})\n"
            else:
                report += f"Result: No significant difference (p >= {results['alpha']})\n"
            
            report += "\n" + "-" * 40 + "\n\n"
        
        return report


class AblationStudy:
    """
    Ablation study to evaluate contribution of different components
    """
    
    def __init__(self, base_config: Dict, save_path: str):
        self.base_config = base_config.copy()
        self.save_path = save_path
        self.ablation_results = {}
        
    def define_ablation_configs(self) -> Dict[str, Dict]:
        """Define different ablation configurations"""
        
        ablations = {
            'full_model': self.base_config.copy(),
            
            'no_temporal': {
                **self.base_config,
                'use_temporal': False,
                'description': 'Remove ConvLSTM temporal fusion'
            },
            
            'no_ndwi': {
                **self.base_config,
                'input_channels': 4,  # Remove NDWI channel
                'description': 'Remove NDWI input channel'
            },
            
            'no_hierarchy': {
                **self.base_config,
                'hierarchy_weight': 0.0,
                'description': 'Remove hierarchical loss component'
            },
            
            'no_attention': {
                **self.base_config,
                'use_cross_attention': False,
                'description': 'Remove cross-scale attention'
            },
            
            'single_scale': {
                **self.base_config,
                'scale_levels': 1,
                'description': 'Use single scale instead of multi-level'
            }
        }
        
        return ablations
    
    def run_ablation_study(self, dataset, num_epochs: int = 10) -> Dict[str, Dict]:
        """
        Run complete ablation study
        
        Args:
            dataset: Dataset for evaluation
            num_epochs: Number of epochs for each ablation
            
        Returns:
            Ablation study results
        """
        
        print("Running Ablation Study")
        print("=" * 40)
        
        ablation_configs = self.define_ablation_configs()
        
        for ablation_name, config in ablation_configs.items():
            print(f"\nEvaluating: {ablation_name}")
            if 'description' in config:
                print(f"Description: {config['description']}")
            
            # Create model with ablation configuration
            try:
                model = create_water_body_model(config)
                
                # Create simple train/val split for ablation (faster than full CV)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size], 
                    generator=torch.Generator().manual_seed(42)
                )
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=config['batch_size'], shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=config['batch_size'], shuffle=False
                )
                
                data_loaders = {'train': train_loader, 'val': val_loader}
                
                # Train model
                trainer = WaterBodyTrainer(model, data_loaders, config)
                trainer.train(num_epochs=num_epochs)
                
                # Evaluate
                final_metrics = trainer.validate_epoch(0)
                
                self.ablation_results[ablation_name] = {
                    'config': config,
                    'final_metrics': final_metrics,
                    'val_loss': final_metrics['total_loss'],
                    'seg_accuracy': final_metrics['segmentation_accuracy'],
                    'cls_accuracy': final_metrics.get('classification_accuracy', 0.0),
                    'parameters': sum(p.numel() for p in model.parameters())
                }
                
                print(f"✓ {ablation_name} completed")
                print(f"  Val Loss: {final_metrics['total_loss']:.4f}")
                print(f"  Seg Acc: {final_metrics['segmentation_accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ {ablation_name} failed: {e}")
                self.ablation_results[ablation_name] = {
                    'error': str(e),
                    'val_loss': float('inf'),
                    'seg_accuracy': 0.0,
                    'cls_accuracy': 0.0
                }
        
        # Save ablation results
        self._save_ablation_results()
        
        # Generate ablation report
        self._generate_ablation_report()
        
        return self.ablation_results
    
    def _save_ablation_results(self):
        """Save ablation results to file"""
        
        # Convert to serializable format
        serializable_results = {}
        for name, results in self.ablation_results.items():
            serializable_results[name] = {
                'val_loss': float(results.get('val_loss', float('inf'))),
                'seg_accuracy': float(results.get('seg_accuracy', 0.0)),
                'cls_accuracy': float(results.get('cls_accuracy', 0.0)),
                'parameters': int(results.get('parameters', 0)),
                'description': results.get('config', {}).get('description', '')
            }
        
        # Save to JSON
        results_path = os.path.join(self.save_path, 'ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Ablation results saved to: {results_path}")
    
    def _generate_ablation_report(self):
        """Generate and save ablation study report"""
        
        # Create comparison plot
        self._plot_ablation_comparison()
        
        # Generate text report
        report = "Ablation Study Results\n"
        report += "=" * 50 + "\n\n"
        
        # Sort by validation accuracy
        sorted_results = sorted(
            self.ablation_results.items(),
            key=lambda x: x[1].get('seg_accuracy', 0.0),
            reverse=True
        )
        
        report += "Ranking by Segmentation Accuracy:\n"
        report += "-" * 40 + "\n"
        
        for rank, (name, results) in enumerate(sorted_results, 1):
            if 'error' not in results:
                report += f"{rank}. {name}\n"
                report += f"   Validation Loss: {results['val_loss']:.4f}\n"
                report += f"   Segmentation Accuracy: {results['seg_accuracy']:.4f}\n"
                report += f"   Classification Accuracy: {results['cls_accuracy']:.4f}\n"
                report += f"   Parameters: {results['parameters']:,}\n"
                
                if 'config' in results and 'description' in results['config']:
                    report += f"   Description: {results['config']['description']}\n"
                
                report += "\n"
        
        # Component importance analysis
        report += "Component Importance Analysis:\n"
        report += "-" * 40 + "\n"
        
        full_model_acc = self.ablation_results.get('full_model', {}).get('seg_accuracy', 0.0)
        
        for name, results in self.ablation_results.items():
            if name != 'full_model' and 'error' not in results:
                accuracy_drop = full_model_acc - results['seg_accuracy']
                report += f"{name}: {accuracy_drop:.4f} accuracy drop\n"
        
        # Save report
        report_path = os.path.join(self.save_path, 'ablation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Ablation report saved to: {report_path}")
    
    def _plot_ablation_comparison(self):
        """Plot ablation study comparison"""
        
        # Extract data for plotting
        names = []
        accuracies = []
        losses = []
        
        for name, results in self.ablation_results.items():
            if 'error' not in results:
                names.append(name.replace('_', ' ').title())
                accuracies.append(results['seg_accuracy'])
                losses.append(results['val_loss'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Ablation Study: Segmentation Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Loss comparison
        bars2 = ax2.bar(names, losses, color='lightcoral', alpha=0.7)
        ax2.set_title('Ablation Study: Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'ablation_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


class RegionalGeneralizabilityEvaluator:
    """
    Evaluates model generalizability across different regions
    """
    
    def __init__(self, trained_model: nn.Module):
        self.model = trained_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def evaluate_on_region(self, region_dataloader, region_name: str) -> Dict[str, float]:
        """
        Evaluate model performance on specific region
        
        Args:
            region_dataloader: DataLoader for specific region
            region_name: Name of the region
            
        Returns:
            Evaluation metrics for the region
        """
        
        print(f"Evaluating on {region_name}...")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_seg_predictions = []
        all_seg_targets = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(region_dataloader, desc=f'Evaluating {region_name}'):
                images = batch['images'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Segmentation predictions
                seg_pred = torch.argmax(outputs['segmentation_logits'], dim=1)
                seg_target = (masks > 0).long()
                
                all_seg_predictions.extend(seg_pred.cpu().numpy().flatten())
                all_seg_targets.extend(seg_target.cpu().numpy().flatten())
                
                # Classification predictions (for water regions)
                for b in range(images.shape[0]):
                    water_mask = masks[b] > 0
                    if water_mask.any():
                        # Get ground truth class
                        gt_classes = masks[b][water_mask]
                        if len(gt_classes) > 0:
                            target_class = stats.mode(gt_classes.cpu().numpy(), keepdims=False)[0]
                            
                            # Get prediction (using global features)
                            features = torch.mean(outputs['features'][b:b+1], dim=(2, 3))
                            if hasattr(self.model, 'hierarchical_classifier'):
                                cls_logits = self.model.hierarchical_classifier.hierarchical_layers['level2'](features)
                                pred_class = torch.argmax(cls_logits, dim=1).cpu().item()
                            else:
                                # For baseline models, use segmentation output
                                pred_class = torch.argmax(outputs['classification_logits'][b]).cpu().item()
                            
                            all_predictions.append(pred_class)
                            all_targets.append(target_class)
                
                num_batches += 1
        
        # Compute metrics
        seg_accuracy = accuracy_score(all_seg_targets, all_seg_predictions)
        
        if all_predictions and all_targets:
            cls_accuracy = accuracy_score(all_targets, all_predictions)
            cls_f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, average=None, zero_division=0
            )
        else:
            cls_accuracy = 0.0
            cls_f1 = 0.0
            precision = recall = f1 = support = np.zeros(6)
        
        metrics = {
            'region': region_name,
            'segmentation_accuracy': seg_accuracy,
            'classification_accuracy': cls_accuracy,
            'classification_f1': cls_f1,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'num_samples': len(all_targets)
        }
        
        return metrics
    
    def evaluate_generalizability(self, region_dataloaders: Dict[str, any]) -> Dict[str, Dict]:
        """
        Evaluate model generalizability across all regions
        
        Args:
            region_dataloaders: Dictionary of region name -> DataLoader
            
        Returns:
            Generalizability evaluation results
        """
        
        results = {}
        
        for region_name, dataloader in region_dataloaders.items():
            region_metrics = self.evaluate_on_region(dataloader, region_name)
            results[region_name] = region_metrics
        
        # Compute generalizability statistics
        self._compute_generalizability_stats(results)
        
        return results
    
    def _compute_generalizability_stats(self, results: Dict[str, Dict]):
        """Compute generalizability statistics across regions"""
        
        # Extract accuracy values
        seg_accuracies = [results[region]['segmentation_accuracy'] for region in results]
        cls_accuracies = [results[region]['classification_accuracy'] for region in results]
        
        # Compute statistics
        stats_dict = {
            'seg_accuracy_mean': np.mean(seg_accuracies),
            'seg_accuracy_std': np.std(seg_accuracies),
            'seg_accuracy_range': max(seg_accuracies) - min(seg_accuracies),
            
            'cls_accuracy_mean': np.mean(cls_accuracies),
            'cls_accuracy_std': np.std(cls_accuracies),
            'cls_accuracy_range': max(cls_accuracies) - min(cls_accuracies)
        }
        
        print("\nGeneralizability Statistics:")
        print(f"Segmentation Accuracy: {stats_dict['seg_accuracy_mean']:.4f} ± {stats_dict['seg_accuracy_std']:.4f}")
        print(f"Classification Accuracy: {stats_dict['cls_accuracy_mean']:.4f} ± {stats_dict['cls_accuracy_std']:.4f}")
        print(f"Performance Range (Seg): {stats_dict['seg_accuracy_range']:.4f}")
        print(f"Performance Range (Cls): {stats_dict['cls_accuracy_range']:.4f}")
        
        # Save statistics
        stats_path = os.path.join(self.save_path, 'generalizability_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)


class ComprehensiveEvaluator:
    """
    Main evaluation class that orchestrates all evaluation components
    """
    
    def __init__(self, config: Dict, save_path: str):
        self.config = config
        self.save_path = save_path
        self.evaluation_results = {}
        
        # Create evaluator components
        self.cv_evaluator = CrossValidationEvaluator(config, save_path)
        self.statistical_tester = StatisticalTester(alpha=0.05)
        self.ablation_study = AblationStudy(config, save_path)
        
        # Ensure save directory
        os.makedirs(save_path, exist_ok=True)
    
    def run_comprehensive_evaluation(self, dataset, region_dataloaders: Dict[str, any]) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            dataset: Main training dataset
            region_dataloaders: Region-specific test dataloaders
            
        Returns:
            Comprehensive evaluation results
        """
        
        print("Starting Comprehensive Evaluation Pipeline")
        print("=" * 60)
        
        # 1. Cross-validation evaluation
        print("\n1. Cross-Validation Evaluation")
        print("-" * 40)
        
        # Define model creators
        model_creators = {
            'temporal_fusion': lambda: create_water_body_model(self.config),
            'unet': lambda: create_baseline_models(self.config)['unet'],
            'deeplabv3': lambda: create_baseline_models(self.config)['deeplabv3']
        }
        
        cv_results = {}
        for model_name, creator_fn in model_creators.items():
            cv_stats = self.cv_evaluator.evaluate_model_cv(creator_fn, dataset, model_name)
            cv_results[model_name] = cv_stats
        
        # 2. Statistical significance testing
        print("\n2. Statistical Significance Testing")
        print("-" * 40)
        
        comparison_results = self.statistical_tester.multiple_model_comparison(
            self.cv_evaluator.results
        )
        
        # Generate statistical report
        stat_report = self.statistical_tester.generate_statistical_report(comparison_results)
        print(stat_report)
        
        # Save statistical report
        stat_report_path = os.path.join(self.save_path, 'statistical_analysis.txt')
        with open(stat_report_path, 'w') as f:
            f.write(stat_report)
        
        # 3. Ablation study
        print("\n3. Ablation Study")
        print("-" * 40)
        
        ablation_results = self.ablation_study.run_ablation_study(dataset, num_epochs=10)
        
        # 4. Regional generalizability
        print("\n4. Regional Generalizability Assessment")
        print("-" * 40)
        
        # Train best model on full dataset
        best_model = create_water_body_model(self.config)
        
        # Create full dataset loader
        full_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config['batch_size'], shuffle=True
        )
        
        # Quick training on full dataset
        trainer = WaterBodyTrainer(best_model, {'train': full_loader, 'val': full_loader}, self.config)
        trainer.train(num_epochs=5)  # Quick training for generalizability test
        
        # Evaluate on different regions
        generalizability_evaluator = RegionalGeneralizabilityEvaluator(best_model)
        generalizability_results = generalizability_evaluator.evaluate_generalizability(region_dataloaders)
        
        # 5. Compile final results
        self.evaluation_results = {
            'cross_validation': cv_results,
            'statistical_tests': comparison_results,
            'ablation_study': ablation_results,
            'generalizability': generalizability_results,
            'evaluation_date': pd.Timestamp.now().isoformat()
        }
        
        # Save comprehensive results
        self._save_comprehensive_results()
        
        # Generate final report
        self._generate_final_report()
        
        return self.evaluation_results
    
    def _save_comprehensive_results(self):
        """Save all evaluation results"""
        
        # Convert to serializable format
        serializable_results = {}
        
        for category, results in self.evaluation_results.items():
            if category == 'cross_validation':
                serializable_results[category] = {
                    model: {
                        'val_loss_mean': float(stats.get('val_loss_mean', 0)),
                        'seg_acc_mean': float(stats.get('seg_acc_mean', 0)),
                        'cls_acc_mean': float(stats.get('cls_acc_mean', 0))
                    }
                    for model, stats in results.items()
                }
            elif category == 'statistical_tests':
                serializable_results[category] = {
                    comparison: {
                        'p_value': float(result.get('p_value', 1.0)),
                        'is_significant': bool(result.get('is_significant', False)),
                        'cohens_d': float(result.get('cohens_d', 0.0))
                    }
                    for comparison, result in results.items()
                }
            else:
                serializable_results[category] = results
        
        # Save to JSON
        results_path = os.path.join(self.save_path, 'comprehensive_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comprehensive results saved to: {results_path}")
    
    def _generate_final_report(self):
        """Generate final evaluation report"""
        
        report = "Water Body Classification - Comprehensive Evaluation Report\n"
        report += "=" * 70 + "\n\n"
        
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 20 + "\n"
        
        # Best performing model
        cv_results = self.evaluation_results['cross_validation']
        best_model = max(cv_results.keys(), 
                        key=lambda x: cv_results[x].get('seg_acc_mean', 0))
        
        report += f"Best Performing Model: {best_model}\n"
        report += f"Segmentation Accuracy: {cv_results[best_model]['seg_acc_mean']:.4f}\n"
        report += f"Classification Accuracy: {cv_results[best_model]['cls_acc_mean']:.4f}\n\n"
        
        # Statistical significance
        stat_tests = self.evaluation_results['statistical_tests']
        significant_improvements = [
            comp for comp, result in stat_tests.items() 
            if result['is_significant']
        ]
        
        report += f"Statistically Significant Improvements: {len(significant_improvements)}\n"
        for comp in significant_improvements:
            result = stat_tests[comp]
            report += f"  {comp}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f}\n"
        
        report += "\n"
        
        # Generalizability assessment
        if 'generalizability' in self.evaluation_results:
            gen_results = self.evaluation_results['generalizability']
            report += "GENERALIZABILITY ASSESSMENT\n"
            report += "-" * 30 + "\n"
            
            for region, metrics in gen_results.items():
                if isinstance(metrics, dict):
                    report += f"{region}: Seg Acc = {metrics.get('segmentation_accuracy', 0):.4f}, "
                    report += f"Cls Acc = {metrics.get('classification_accuracy', 0):.4f}\n"
        
        # Save final report
        report_path = os.path.join(self.save_path, 'final_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Final evaluation report saved to: {report_path}")
        print("\nEvaluation pipeline completed!")


# Example usage
if __name__ == "__main__":
    # Test evaluation framework
    config = create_training_config()
    save_path = "/content/drive/MyDrive/WaterBodyResearch/evaluation"
    
    print("Testing Evaluation Framework")
    print("=" * 40)
    
    # Test statistical tester
    tester = StatisticalTester()
    
    # Simulate results for testing
    model1_results = np.random.normal(0.85, 0.05, 5)  # Better model
    model2_results = np.random.normal(0.80, 0.05, 5)  # Baseline model
    
    test_result = tester.paired_t_test(
        model1_results.tolist(), model2_results.tolist(),
        'temporal_fusion', 'unet_baseline'
    )
    
    print("Statistical Test Result:")
    print(f"p-value: {test_result['p_value']:.6f}")
    print(f"Significant: {test_result['is_significant']}")
    print(f"Effect size: {test_result['effect_size']}")
    
    print("\nEvaluation framework test completed!")