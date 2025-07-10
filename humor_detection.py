import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                           classification_report, roc_auc_score, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
import torch
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
import pingouin as pg
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
import json
from datetime import datetime

RANDOM_SEEDS = [42, 123, 456, 789, 1337]
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
TFIDF_FEATURES = 5000
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 1000

class StatisticalAnalyzer:
    """Class to handle all statistical analysis operations"""
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def bootstrap_metric(self, y_true, y_pred, metric_func, n_bootstrap=1000):
        # Calculate confidence intervals using bootstrap resampling
        """Calculate bootstrap confidence intervals for a metric"""
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = resample(range(n_samples), n_samples=n_samples, random_state=None)
            y_true_boot = [y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]
            
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        
        lower_percentile = (self.alpha/2) * 100
        upper_percentile = (1 - self.alpha/2) * 100
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return np.mean(bootstrap_scores), ci_lower, ci_upper, bootstrap_scores
    
    def cohens_d(self, group1, group2):
        # Calculate Cohen's d effect size between two groups
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    def paired_ttest(self, scores1, scores2):
        """Perform paired t-test"""
        statistic, p_value = ttest_rel(scores1, scores2)
        return statistic, p_value
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2):
        # Perform McNemar's test to compare two classifiers
        """Perform McNemar's test for paired classification results"""
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        both_correct = np.sum(correct1 & correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        both_incorrect = np.sum(~correct1 & ~correct2)
        
        table = np.array([[both_correct, model1_only],
                         [model2_only, both_incorrect]])
        
        try:
            result = mcnemar(table, exact=False, correction=True)
            return result.statistic, result.pvalue
        except:
            b, c = model1_only, model2_only
            if b + c == 0:
                return 0, 1.0
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
            return chi2, p_value

class EnhancedDataset:
    """Enhanced dataset class with proper splitting"""
    
    def __init__(self, file_path, random_seed=42):
        self.df = self.load_and_validate_csv(file_path)
        self.random_seed = random_seed
        self.prepare_official_split()
    
    def load_and_validate_csv(self, file_path):
        """Load CSV with validation and NaN handling"""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['text', 'is_humor', 'humor_rating',
                              'humor_controversy', 'offense_rating']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("CSV missing required columns")
            
            df = df.dropna(subset=['text', 'is_humor'])
            df['humor_rating'] = df['humor_rating'].fillna(0)
            df['offense_rating'] = df['offense_rating'].fillna(0)
            df['humor_controversy'] = df['humor_controversy'].fillna(0)
            return df
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise
    
    def prepare_official_split(self):
        # Create stratified 80:10:10 train/dev/test split
        """Prepare official HaHackathon train/dev/test split (80:10:10)"""
        self.train_df, temp_df = train_test_split(
            self.df, test_size=0.2, random_state=self.random_seed,
            stratify=self.df['is_humor']
        )
        
        self.dev_df, self.test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.random_seed,
            stratify=temp_df['is_humor']
        )
        
        print(f"Dataset split sizes:")
        print(f"Train: {len(self.train_df)} ({len(self.train_df)/len(self.df)*100:.1f}%)")
        print(f"Dev: {len(self.dev_df)} ({len(self.dev_df)/len(self.df)*100:.1f}%)")
        print(f"Test: {len(self.test_df)} ({len(self.test_df)/len(self.df)*100:.1f}%)")

class EnhancedBertClassifier:
    """Enhanced BERT classifier with statistical analysis"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.set_seeds()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
    
    def preprocess_text(self, text):
        """Robust cleaning for social media content"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data(self, dataset):
        # Preprocess text and create BERT-compatible datasets
        """Prepare BERT-compatible datasets"""
        for df_name in ['train_df', 'dev_df', 'test_df']:
            df = getattr(dataset, df_name)
            df['text'] = df['text'].apply(self.preprocess_text)
            df = df[df['text'].str.len() > 0]
            setattr(dataset, df_name, df)
        
        def tokenize(batch):
            return self.tokenizer(
                batch['text'].tolist(),
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            )
        
        def create_dataset(df):
            encodings = tokenize(df)
            return BertDataset(encodings, df['is_humor'].values)
        
        self.train_dataset = create_dataset(dataset.train_df)
        self.dev_dataset = create_dataset(dataset.dev_df)
        self.test_dataset = create_dataset(dataset.test_df)
        
        return dataset.train_df, dataset.dev_df, dataset.test_df
    
    def train(self):
        # Train BERT model with early stopping and evaluation
        """Train BERT model with proper configuration"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        
        training_args = TrainingArguments(
            output_dir=f'./bert_results_{self.random_seed}',
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_dir=f'./bert_logs_{self.random_seed}',
            save_total_limit=3,
            seed=self.random_seed,
            data_seed=self.random_seed
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                'f1': f1_score(labels, preds),
                'accuracy': accuracy_score(labels, preds)
            }
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        self.trainer.train()
    
    def evaluate(self, test_df):
        """Evaluate BERT model"""
        test_results = self.trainer.predict(self.test_dataset)
        preds = np.argmax(test_results.predictions, axis=-1)
        probabilities = torch.softmax(torch.tensor(test_results.predictions), dim=-1)[:, 1].numpy()
        
        return preds, probabilities, test_results

class EnhancedTfidfClassifier:
    """Enhanced TF-IDF classifier with statistical analysis"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=TFIDF_FEATURES,
                ngram_range=(1, 3),
                stop_words='english'
            )),
            ('scaler', MaxAbsScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(512, 128),
                activation='relu',
                early_stopping=True,
                random_state=self.random_seed,
                max_iter=500
            ))
        ])
    
    def preprocess_text(self, text):
        """Text preprocessing for TF-IDF"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, X_train, y_train):
        """Train TF-IDF + MLP model"""
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        self.pipeline.fit(X_train_processed, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate TF-IDF + MLP model"""
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        preds = self.pipeline.predict(X_test_processed)
        probabilities = self.pipeline.predict_proba(X_test_processed)[:, 1]
        return preds, probabilities

class BertDataset(torch.utils.data.Dataset):
    """Dataset class for BERT"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.encodings["input_ids"])

def calculate_comprehensive_metrics(y_true, y_pred, y_prob, analyzer):
    # Calculate all metrics with bootstrap confidence intervals
    """Calculate comprehensive metrics with confidence intervals"""
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    acc_mean, acc_ci_lower, acc_ci_upper, _ = analyzer.bootstrap_metric(
        y_true, y_pred, accuracy_score, N_BOOTSTRAP
    )
    f1_mean, f1_ci_lower, f1_ci_upper, _ = analyzer.bootstrap_metric(
        y_true, y_pred, f1_score, N_BOOTSTRAP
    )
    auc_mean, auc_ci_lower, auc_ci_upper, _ = analyzer.bootstrap_metric(
        y_true, y_prob, roc_auc_score, N_BOOTSTRAP
    )
    
    return {
        'accuracy': accuracy,
        'accuracy_ci': (acc_ci_lower, acc_ci_upper),
        'f1': f1,
        'f1_ci': (f1_ci_lower, f1_ci_upper),
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'auc_ci': (auc_ci_lower, auc_ci_upper)
    }

def run_multiple_seeds_experiment():
    # Main experiment function - runs both models with multiple seeds for statistical analysis
    """Run experiments with multiple random seeds"""
    
    print("Enhanced HaHackathon Humor Detection Pipeline")
    print("=" * 80)
    print(f"Running experiments with {len(RANDOM_SEEDS)} random seeds: {RANDOM_SEEDS}")
    print(f"Confidence Level: {CONFIDENCE_LEVEL*100}%")
    print(f"Bootstrap samples: {N_BOOTSTRAP}")
    print("=" * 80)
    
    analyzer = StatisticalAnalyzer(CONFIDENCE_LEVEL)
    
    bert_results = []
    tfidf_results = []
    
    all_bert_predictions = []
    all_tfidf_predictions = []
    test_labels = None
    
    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n--- Running Experiment {seed_idx + 1}/{len(RANDOM_SEEDS)} (Seed: {seed}) ---")
        
        try:
            dataset = EnhancedDataset('hahackathon_data.csv', random_seed=seed)
        except FileNotFoundError:
            print("Error: hahackathon_data.csv not found. Please ensure the file is in the correct location.")
            return
        
        if test_labels is None:
            test_labels = dataset.test_df['is_humor'].values
        
        print("Training BERT model...")
        bert_model = EnhancedBertClassifier(random_seed=seed)
        train_df, dev_df, test_df = bert_model.prepare_data(dataset)
        bert_model.train()
        
        print("Evaluating BERT model...")
        bert_preds, bert_probs, _ = bert_model.evaluate(test_df)
        bert_metrics = calculate_comprehensive_metrics(
            test_df['is_humor'].values, bert_preds, bert_probs, analyzer
        )
        bert_results.append(bert_metrics)
        all_bert_predictions.append(bert_preds)
        
        print("Training TF-IDF + MLP model...")
        tfidf_model = EnhancedTfidfClassifier(random_seed=seed)
        
        X_train = train_df['text'].values
        y_train = train_df['is_humor'].values
        X_test = test_df['text'].values
        y_test = test_df['is_humor'].values
        
        tfidf_model.train(X_train, y_train)
        
        print("Evaluating TF-IDF + MLP model...")
        tfidf_preds, tfidf_probs = tfidf_model.evaluate(X_test, y_test)
        tfidf_metrics = calculate_comprehensive_metrics(
            y_test, tfidf_preds, tfidf_probs, analyzer
        )
        tfidf_results.append(tfidf_metrics)
        all_tfidf_predictions.append(tfidf_preds)
        
        print(f"\nSeed {seed} Results:")
        print(f"BERT    - Accuracy: {bert_metrics['accuracy']:.4f}, F1: {bert_metrics['f1']:.4f}, AUC: {bert_metrics['auc']:.4f}")
        print(f"TF-IDF  - Accuracy: {tfidf_metrics['accuracy']:.4f}, F1: {tfidf_metrics['f1']:.4f}, AUC: {tfidf_metrics['auc']:.4f}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 80)
    
    bert_accuracies = [r['accuracy'] for r in bert_results]
    bert_f1s = [r['f1'] for r in bert_results]
    bert_aucs = [r['auc'] for r in bert_results]
    
    tfidf_accuracies = [r['accuracy'] for r in tfidf_results]
    tfidf_f1s = [r['f1'] for r in tfidf_results]
    tfidf_aucs = [r['auc'] for r in tfidf_results]
    
    print("\n1. DESCRIPTIVE STATISTICS (across seeds)")
    print("-" * 50)
    
    results_summary = {
        'BERT': {
            'Accuracy': (np.mean(bert_accuracies), np.std(bert_accuracies)),
            'F1-Score': (np.mean(bert_f1s), np.std(bert_f1s)),
            'AUC': (np.mean(bert_aucs), np.std(bert_aucs))
        },
        'TF-IDF+MLP': {
            'Accuracy': (np.mean(tfidf_accuracies), np.std(tfidf_accuracies)),
            'F1-Score': (np.mean(tfidf_f1s), np.std(tfidf_f1s)),
            'AUC': (np.mean(tfidf_aucs), np.std(tfidf_aucs))
        }
    }
    
    for model_name, metrics in results_summary.items():
        print(f"\n{model_name}:")
        for metric_name, (mean_val, std_val) in metrics.items():
            ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(RANDOM_SEEDS))
            ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(RANDOM_SEEDS))
            print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    print("\n2. STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 50)
    
    acc_t_stat, acc_p_val = analyzer.paired_ttest(bert_accuracies, tfidf_accuracies)
    f1_t_stat, f1_p_val = analyzer.paired_ttest(bert_f1s, tfidf_f1s)
    auc_t_stat, auc_p_val = analyzer.paired_ttest(bert_aucs, tfidf_aucs)
    
    print("Paired t-tests (BERT vs TF-IDF+MLP):")
    print(f"  Accuracy: t={acc_t_stat:.4f}, p={acc_p_val:.4f}")
    print(f"  F1-Score: t={f1_t_stat:.4f}, p={f1_p_val:.4f}")
    print(f"  AUC:      t={auc_t_stat:.4f}, p={auc_p_val:.4f}")
    
    p_values = [acc_p_val, f1_p_val, auc_p_val]
    corrected_alpha = 0.05 / len(p_values)
    print(f"\nBonferroni-corrected significance level: {corrected_alpha:.4f}")
    
    significant_tests = []
    for i, (metric, p_val) in enumerate(zip(['Accuracy', 'F1-Score', 'AUC'], p_values)):
        is_significant = p_val < corrected_alpha
        significant_tests.append(is_significant)
        print(f"  {metric}: {'Significant' if is_significant else 'Not significant'} (p={p_val:.4f})")
    
    print("\n3. EFFECT SIZES (Cohen's d)")
    print("-" * 50)
    
    acc_cohens_d = analyzer.cohens_d(bert_accuracies, tfidf_accuracies)
    f1_cohens_d = analyzer.cohens_d(bert_f1s, tfidf_f1s)
    auc_cohens_d = analyzer.cohens_d(bert_aucs, tfidf_aucs)
    
    def interpret_cohens_d(d):
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    print("Cohen's d (BERT vs TF-IDF+MLP):")
    print(f"  Accuracy: d={acc_cohens_d:.4f} ({interpret_cohens_d(acc_cohens_d)} effect)")
    print(f"  F1-Score: d={f1_cohens_d:.4f} ({interpret_cohens_d(f1_cohens_d)} effect)")
    print(f"  AUC:      d={auc_cohens_d:.4f} ({interpret_cohens_d(auc_cohens_d)} effect)")
    
    print("\n4. McNEMAR'S TESTS (Classification Comparison)")
    print("-" * 50)
    
    mcnemar_stats = []
    mcnemar_p_values = []
    
    for seed_idx in range(len(RANDOM_SEEDS)):
        chi2, p_val = analyzer.mcnemar_test(
            test_labels, 
            all_bert_predictions[seed_idx], 
            all_tfidf_predictions[seed_idx]
        )
        mcnemar_stats.append(chi2)
        mcnemar_p_values.append(p_val)
        print(f"  Seed {RANDOM_SEEDS[seed_idx]}: χ²={chi2:.4f}, p={p_val:.4f}")
    
    avg_mcnemar_p = np.mean(mcnemar_p_values)
    print(f"\nAverage McNemar's p-value: {avg_mcnemar_p:.4f}")
    
    print("\n5. PERFORMANCE IMPROVEMENTS")
    print("-" * 50)
    
    acc_improvement = (np.mean(bert_accuracies) - np.mean(tfidf_accuracies)) / np.mean(tfidf_accuracies) * 100
    f1_improvement = (np.mean(bert_f1s) - np.mean(tfidf_f1s)) / np.mean(tfidf_f1s) * 100
    auc_improvement = (np.mean(bert_aucs) - np.mean(tfidf_aucs)) / np.mean(tfidf_aucs) * 100
    
    print(f"BERT improvements over TF-IDF+MLP:")
    print(f"  Accuracy: +{acc_improvement:.2f}%")
    print(f"  F1-Score: +{f1_improvement:.2f}%")
    print(f"  AUC:      +{auc_improvement:.2f}%")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"statistical_analysis_results_{timestamp}.json"
    
    detailed_results = {
        'experiment_config': {
            'random_seeds': RANDOM_SEEDS,
            'confidence_level': CONFIDENCE_LEVEL,
            'n_bootstrap': N_BOOTSTRAP,
            'timestamp': timestamp
        },
        'raw_results': {
            'bert': {
                'accuracies': bert_accuracies,
                'f1_scores': bert_f1s,
                'aucs': bert_aucs
            },
            'tfidf': {
                'accuracies': tfidf_accuracies,
                'f1_scores': tfidf_f1s,
                'aucs': tfidf_aucs
            }
        },
        'statistical_tests': {
            'paired_t_tests': {
                'accuracy': {'t_statistic': acc_t_stat, 'p_value': acc_p_val},
                'f1_score': {'t_statistic': f1_t_stat, 'p_value': f1_p_val},
                'auc': {'t_statistic': auc_t_stat, 'p_value': auc_p_val}
            },
            'effect_sizes': {
                'accuracy_cohens_d': acc_cohens_d,
                'f1_cohens_d': f1_cohens_d,
                'auc_cohens_d': auc_cohens_d
            },
            'mcnemar_tests': {
                'chi2_statistics': mcnemar_stats,
                'p_values': mcnemar_p_values,
                'average_p_value': avg_mcnemar_p
            }
        },
        'summary': results_summary
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n6. RESULTS SAVED")
    print("-" * 50)
    print(f"Detailed results saved to: {results_file}")
    
    create_statistical_visualizations(bert_results, tfidf_results, RANDOM_SEEDS)
    
    return detailed_results

def create_statistical_visualizations(bert_results, tfidf_results, seeds):
    # Generate comprehensive statistical plots and save as separate images
    """Create comprehensive visualizations of statistical analysis as separate images"""
    
    bert_acc = [r['accuracy'] for r in bert_results]
    bert_f1 = [r['f1'] for r in bert_results]
    bert_auc = [r['auc'] for r in bert_results]
    
    tfidf_acc = [r['accuracy'] for r in tfidf_results]
    tfidf_f1 = [r['f1'] for r in tfidf_results]
    tfidf_auc = [r['auc'] for r in tfidf_results]
    
    metrics = ['Accuracy', 'F1-Score', 'AUC']
    bert_data = [bert_acc, bert_f1, bert_auc]
    tfidf_data = [tfidf_acc, tfidf_f1, tfidf_auc]
    
    for i, (metric, bert_vals, tfidf_vals) in enumerate(zip(metrics, bert_data, tfidf_data)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        data_to_plot = [bert_vals, tfidf_vals]
        bp = ax.boxplot(data_to_plot, labels=['BERT', 'TF-IDF+MLP'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title(f'{metric} Distribution Across Random Seeds', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        mean_bert = np.mean(bert_vals)
        mean_tfidf = np.mean(tfidf_vals)
        ax.text(0.02, 0.98, f'BERT Mean: {mean_bert:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.02, 0.88, f'TF-IDF Mean: {mean_tfidf:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        filename = f'boxplot_{metric.lower().replace("-", "_")}_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    for i, (metric, bert_vals, tfidf_vals) in enumerate(zip(metrics, bert_data, tfidf_data)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x = range(len(seeds))
        ax.plot(x, bert_vals, 'o-', label='BERT', color='blue', linewidth=2, markersize=8)
        ax.plot(x, tfidf_vals, 's-', label='TF-IDF+MLP', color='red', linewidth=2, markersize=8)
        ax.set_xlabel('Random Seed', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Performance by Random Seed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        for j, (bert_val, tfidf_val) in enumerate(zip(bert_vals, tfidf_vals)):
            ax.annotate(f'{bert_val:.3f}', (j, bert_val), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9, color='blue')
            ax.annotate(f'{tfidf_val:.3f}', (j, tfidf_val), textcoords="offset points", 
                       xytext=(0,-15), ha='center', fontsize=9, color='red')
        
        plt.tight_layout()
        filename = f'seed_comparison_{metric.lower().replace("-", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    models = ['BERT', 'TF-IDF+MLP']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bert_means = [np.mean(data) for data in bert_data]
    bert_stds = [np.std(data) for data in bert_data]
    tfidf_means = [np.mean(data) for data in tfidf_data]
    tfidf_stds = [np.std(data) for data in tfidf_data]
    
    n_seeds = len(seeds)
    bert_cis = [1.96 * std / np.sqrt(n_seeds) for std in bert_stds]
    tfidf_cis = [1.96 * std / np.sqrt(n_seeds) for std in tfidf_stds]
    
    bars1 = ax.bar(x_pos - width/2, bert_means, width, yerr=bert_cis, 
                   label='BERT', alpha=0.8, capsize=5, color='lightblue')
    bars2 = ax.bar(x_pos + width/2, tfidf_means, width, yerr=tfidf_cis,
                   label='TF-IDF+MLP', alpha=0.8, capsize=5, color='lightcoral')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Performance', fontsize=12)
    ax.set_title('Model Performance Comparison with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = 'confidence_intervals_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    performance_data = np.array([bert_means, tfidf_means])
    
    im = ax.imshow(performance_data, cmap='RdYlBu_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(models)
    
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{performance_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Heatmap: Model Comparison Across Metrics', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', fontsize=12)
    
    plt.tight_layout()
    filename = 'performance_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    from scipy.stats import ttest_rel
    
    effect_sizes = []
    p_values = []
    
    for bert_vals, tfidf_vals in zip(bert_data, tfidf_data):
        pooled_std = np.sqrt((np.var(bert_vals, ddof=1) + np.var(tfidf_vals, ddof=1)) / 2)
        cohens_d = (np.mean(bert_vals) - np.mean(tfidf_vals)) / pooled_std
        effect_sizes.append(cohens_d)
        
        _, p_val = ttest_rel(bert_vals, tfidf_vals)
        p_values.append(p_val)
    
    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, effect_sizes, alpha=0.8, color=['green' if p < 0.05 else 'orange' for p in p_values])
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Effect Sizes (BERT vs TF-IDF+MLP) with Statistical Significance", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
    
    for i, (bar, effect_size, p_val) in enumerate(zip(bars, effect_sizes, p_values)):
        height = bar.get_height()
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.annotate(f'{effect_size:.3f}\n({significance})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5 if height >= 0 else -20),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=10, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    filename = 'effect_sizes_significance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    print(f"\nAll visualization files saved successfully!")
    print("Generated files:")
    print("- boxplot_accuracy_distribution.png")
    print("- boxplot_f1_score_distribution.png") 
    print("- boxplot_auc_distribution.png")
    print("- seed_comparison_accuracy.png")
    print("- seed_comparison_f1_score.png")
    print("- seed_comparison_auc.png")
    print("- confidence_intervals_comparison.png")
    print("- performance_heatmap.png")
    print("- effect_sizes_significance.png")

if __name__ == "__main__":
    try:
        import pingouin
    except ImportError:
        print("Installing pingouin for statistical analysis...")
        import subprocess
        subprocess.check_call(["pip", "install", "pingouin"])
        import pingouin
    
    results = run_multiple_seeds_experiment()
    print("\nStatistical analysis complete!")
