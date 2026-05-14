"""
Latency benchmarking for model inference speed.
Measures inference time per URL in microseconds as specified in the paper.
"""

import json
import os
import time
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator

from .config import *
from .baseline_models import load_data, prepare_features


def benchmark_inference_time(
    model: BaseEstimator, 
    X_test: np.ndarray, 
    n_samples: int = 10000,
    warmup_samples: int = 1000
) -> Dict[str, Any]:
    """
    Benchmark inference time for a scikit-learn compatible model.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        n_samples: Number of samples to benchmark
        warmup_samples: Number of warmup samples (to account for JIT/etc)
        
    Returns:
        Dictionary with latency metrics
    """
    # Ensure we don't exceed test set size
    n_samples = min(n_samples, len(X_test))
    warmup_samples = min(warmup_samples, len(X_test) - n_samples)
    
    # Select samples for benchmarking
    start_idx = warmup_samples
    end_idx = start_idx + n_samples
    X_benchmark = X_test[start_idx:end_idx]
    
    # Warmup runs
    if hasattr(model, 'predict') and warmup_samples > 0:
        _ = model.predict(X_benchmark[:warmup_samples])
    
    # Benchmark runs
    start_time = time.perf_counter()
    predictions = model.predict(X_benchmark)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_sample = total_time / n_samples
    
    # Convert to microseconds
    latency_us = avg_time_per_sample * 1_000_000
    throughput = n_samples / total_time  # samples per second
    
    # Get system info
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        "latency_us": latency_us,
        "latency_ms": latency_us / 1000,
        "throughput_per_sec": throughput,
        "total_time_s": total_time,
        "n_samples": n_samples,
        "memory_mb": memory_mb,
        "model_type": str(type(model).__name__)
    }


def benchmark_distilbert_inference(
    test_texts: List[str],
    n_samples: int = 10000,
    warmup_samples: int = 1000
) -> Dict[str, Any]:
    """
    Benchmark DistilBERT inference time.
    
    Args:
        test_texts: List of URL strings to test
        n_samples: Number of samples to benchmark
        warmup_samples: Number of warmup samples
        
    Returns:
        Dictionary with latency metrics
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    
    # Ensure we don't exceed test set size
    n_samples = min(n_samples, len(test_texts))
    warmup_samples = min(warmup_samples, len(test_texts) - n_samples)
    
    # Select samples for benchmarking
    start_idx = warmup_samples
    end_idx = start_idx + n_samples
    benchmark_texts = test_texts[start_idx:end_idx]
    
    # Load model and tokenizer
    model_path = MODEL_OUTPUT_DIR_DISTILBERT
    if not os.path.exists(model_path):
        model_path = MODEL_OUTPUT_DIR
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Warmup runs
    if warmup_samples > 0:
        warmup_texts = benchmark_texts[:warmup_samples]
        inputs = tokenizer(
            warmup_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        with torch.no_grad():
            _ = model(**inputs)
    
    # Benchmark runs
    start_time = time.perf_counter()
    
    # Process in batches for efficiency
    batch_size = 32
    all_predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_texts = benchmark_texts[i:batch_end]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_predictions.append(predictions)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_sample = total_time / n_samples
    
    # Convert to microseconds
    latency_us = avg_time_per_sample * 1_000_000
    throughput = n_samples / total_time  # samples per second
    
    # Get system info
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        "latency_us": latency_us,
        "latency_ms": latency_us / 1000,
        "throughput_per_sec": throughput,
        "total_time_s": total_time,
        "n_samples": n_samples,
        "memory_mb": memory_mb,
        "model_type": "DistilBERT",
        "batch_size": batch_size
    }


def run_latency_benchmarks() -> Dict[str, Any]:
    """
    Run latency benchmarks for all available models.
    """
    print("Starting latency benchmarks...")
    
    results = {}
    
    # Load data for baseline models
    train_df, test_df = load_data()
    
    # Apply same sampling as in baseline_models.py for consistency
    max_train = 20000
    if len(train_df) > max_train:
        samples_per_class = max_train // 4
        train_df = pd.concat([
            train_df[train_df["label"] == i].sample(
                n=min(samples_per_class, (train_df["label"] == i).sum()),
                random_state=RANDOM_SEED,
            )
            for i in range(4)
        ])
        print(f"Sampled {len(train_df)} training rows (4-class stratified)")
    
    # Prepare features
    train_urls = train_df["url"].tolist()
    test_urls = test_df["url"].tolist()
    
    tfidf, X_train, X_test = prepare_features(train_urls, test_urls)
    
    # Benchmark each baseline model if it exists
    baseline_models_to_test = [
        ("logistic_regression", "logistic_regression.pkl"),
        ("random_forest", "random_forest.pkl"),
        ("xgboost", "xgboost.pkl"),
    ]
    
    for model_name, model_filename in baseline_models_to_test:
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            print(f"Benchmarking {model_name}...")
            
            # Load model
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Run benchmark
            latency_result = benchmark_inference_time(model, X_test)
            results[model_name] = latency_result
            print(f"  {model_name}: {latency_result['latency_us']:.2f} µs/URL")
        else:
            print(f"Model {model_name} not found at {model_path}")
            results[model_name] = {"error": "Model not found"}
    
    # Benchmark DistilBERT if it exists
    distilbert_path = MODEL_OUTPUT_DIR_DISTILBERT
    if not os.path.exists(distilbert_path):
        distilbert_path = MODEL_OUTPUT_DIR
    
    if os.path.exists(distilbert_path):
        print("Benchmarking DistilBERT...")
        # Get test texts for DistilBERT
        test_texts = test_df["url"].tolist()
        
        distilbert_result = benchmark_distilbert_inference(test_texts)
        results["distilbert"] = distilbert_result
        print(f"  DistilBERT: {distilbert_result['latency_us']:.2f} µs/URL")
    else:
        print("DistilBERT model not found")
        results["distilbert"] = {"error": "Model not found"}
    
    print(f"\nLatency benchmarking completed for {len([k for k in results.keys() if 'error' not in results[k]])} models.")
    return results


def save_latency_results(results: Dict[str, Any]):
    """Save latency benchmark results to JSON file."""
    results_dir = os.path.join(LOG_DIR, "latency_benchmarks")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"latency_results_{timestamp}.json")
    
    # Convert numpy types to JSON-serializable types
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_for_json(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    return filepath


if __name__ == "__main__":
    # Test latency benchmarking
    print("Testing latency benchmarks...")
    results = run_latency_benchmarks()
    print(json.dumps(results, indent=2))
