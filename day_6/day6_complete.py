#!/usr/bin/env python3
"""
Day 6: Complete Implementation - ANN Libraries (FAISS and hnswlib)
Task: Build FAISS index and perform top-k nearest neighbor queries
"""

import numpy as np
import pandas as pd
import time
import json
from typing import List, Tuple, Dict, Any
import os

def create_comprehensive_dataset():
    """Create a larger, more realistic dataset for ANN testing"""
    print("\n📊 Creating Comprehensive Dataset...")

    # Create different sized datasets for testing scalability
    datasets = {
        'small': {'n_vectors': 1000, 'dimension': 128},
        'medium': {'n_vectors': 10000, 'dimension': 256},
        'large': {'n_vectors': 50000, 'dimension': 384}
    }

    generated_data = {}
    np.random.seed(42)  # For reproducibility

    for size_name, params in datasets.items():
        print(f"  Creating {size_name} dataset: {params['n_vectors']} vectors × {params['dimension']} dims")

        # Generate clustered data (more realistic than pure random)
        n_clusters = 10
        vectors_per_cluster = params['n_vectors'] // n_clusters
        all_vectors = []

        for cluster_id in range(n_clusters):
            # Create cluster center
            center = np.random.randn(params['dimension']) * 2

            # Generate vectors around the center
            cluster_vectors = np.random.randn(vectors_per_cluster, params['dimension']) * 0.5
            cluster_vectors += center  # Add cluster center offset
            all_vectors.append(cluster_vectors)

        # Combine all clusters
        database_vectors = np.vstack(all_vectors).astype('float32')

        # Generate query vectors (some similar to database, some random)
        n_queries = min(100, params['n_vectors'] // 10)
        similar_queries = database_vectors[:n_queries//2] + np.random.randn(n_queries//2, params['dimension']) * 0.1
        random_queries = np.random.randn(n_queries - n_queries//2, params['dimension'])
        query_vectors = np.vstack([similar_queries, random_queries]).astype('float32')

        generated_data[size_name] = {
            'database': database_vectors,
            'queries': query_vectors,
            'n_vectors': params['n_vectors'],
            'dimension': params['dimension'],
            'n_queries': n_queries
        }

    print("✅ Dataset creation completed")
    return generated_data

def benchmark_faiss_indexes(data: Dict, k: int = 10) -> Dict:
    """Benchmark different FAISS index types"""
    print("\n🔍 Benchmarking FAISS Indexes...")

    try:
        import faiss

        results = {}

        for dataset_name, dataset in data.items():
            print(f"\n  Testing on {dataset_name} dataset ({dataset['n_vectors']} vectors)...")

            database = dataset['database']
            queries = dataset['queries']
            dimension = dataset['dimension']

            dataset_results = {}

            # 1. IndexFlatL2 (Exact search)
            print("    • IndexFlatL2 (Exact)...")
            start_time = time.time()
            index_flat = faiss.IndexFlatL2(dimension)
            index_flat.add(database)
            build_time = time.time() - start_time

            start_time = time.time()
            distances, indices = index_flat.search(queries, k)
            search_time = time.time() - start_time

            dataset_results['IndexFlatL2'] = {
                'build_time': build_time,
                'search_time': search_time,
                'memory_usage': index_flat.ntotal * dimension * 4,  # float32
                'accuracy': 1.0,  # Exact search
                'avg_distance': np.mean(distances[:, 0])
            }

            # 2. IndexIVFFlat (Inverted File)
            if dataset['n_vectors'] >= 1000:  # Only for larger datasets
                print("    • IndexIVFFlat (Approximate)...")
                nlist = min(100, dataset['n_vectors'] // 10)  # Number of clusters

                start_time = time.time()
                quantizer = faiss.IndexFlatL2(dimension)
                index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                index_ivf.train(database)
                index_ivf.add(database)
                build_time = time.time() - start_time

                # Set search parameters
                index_ivf.nprobe = min(10, nlist)  # Number of clusters to search

                start_time = time.time()
                distances_ivf, indices_ivf = index_ivf.search(queries, k)
                search_time = time.time() - start_time

                # Calculate accuracy vs exact search
                accuracy = calculate_recall(indices, indices_ivf, k)

                dataset_results['IndexIVFFlat'] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_usage': index_ivf.ntotal * dimension * 4,
                    'accuracy': accuracy,
                    'avg_distance': np.mean(distances_ivf[:, 0])
                }

            # 3. IndexHNSWFlat (FAISS HNSW implementation)
            if dataset['n_vectors'] >= 500:
                print("    • IndexHNSWFlat (Graph-based)...")

                start_time = time.time()
                index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
                index_hnsw.add(database)
                build_time = time.time() - start_time

                start_time = time.time()
                distances_hnsw, indices_hnsw = index_hnsw.search(queries, k)
                search_time = time.time() - start_time

                accuracy = calculate_recall(indices, indices_hnsw, k)

                dataset_results['IndexHNSWFlat'] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_usage': estimate_hnsw_memory(dataset['n_vectors'], dimension),
                    'accuracy': accuracy,
                    'avg_distance': np.mean(distances_hnsw[:, 0])
                }

            results[dataset_name] = dataset_results

        return results

    except ImportError:
        print("❌ FAISS not available")
        return {}
    except Exception as e:
        print(f"❌ FAISS benchmarking failed: {e}")
        return {}

def benchmark_hnswlib(data: Dict, k: int = 10) -> Dict:
    """Benchmark hnswlib with different parameters"""
    print("\n🕸️ Benchmarking hnswlib...")

    try:
        import hnswlib

        results = {}

        for dataset_name, dataset in data.items():
            print(f"\n  Testing on {dataset_name} dataset ({dataset['n_vectors']} vectors)...")

            database = dataset['database']
            queries = dataset['queries']
            dimension = dataset['dimension']

            dataset_results = {}

            # Test different parameter configurations
            configs = [
                {'M': 16, 'ef_construction': 200, 'ef': 50, 'name': 'Balanced'},
                {'M': 32, 'ef_construction': 400, 'ef': 100, 'name': 'High_Accuracy'},
                {'M': 8, 'ef_construction': 100, 'ef': 25, 'name': 'Fast_Build'}
            ]

            for config in configs:
                print(f"    • {config['name']} (M={config['M']}, ef={config['ef']})...")

                start_time = time.time()
                index = hnswlib.Index(space='l2', dim=dimension)
                index.init_index(
                    max_elements=dataset['n_vectors'],
                    ef_construction=config['ef_construction'],
                    M=config['M']
                )

                # Add vectors
                labels = np.arange(dataset['n_vectors'])
                index.add_items(database, labels)
                build_time = time.time() - start_time

                # Set search parameter
                index.set_ef(config['ef'])

                # Search
                start_time = time.time()
                all_labels = []
                all_distances = []

                for query in queries:
                    labels_found, distances = index.knn_query(query, k=k)
                    all_labels.append(labels_found)
                    all_distances.append(distances)

                search_time = time.time() - start_time

                # Convert results to numpy arrays for consistency
                indices_hnsw = np.array(all_labels)
                distances_hnsw = np.array(all_distances)

                dataset_results[config['name']] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_usage': estimate_hnsw_memory(dataset['n_vectors'], dimension),
                    'avg_distance': np.mean([d[0] for d in all_distances]),
                    'parameters': {
                        'M': config['M'],
                        'ef_construction': config['ef_construction'],
                        'ef': config['ef']
                    }
                }

            results[dataset_name] = dataset_results

        return results

    except ImportError:
        print("❌ hnswlib not available")
        return {}
    except Exception as e:
        print(f"❌ hnswlib benchmarking failed: {e}")
        return {}

def calculate_recall(ground_truth_indices, test_indices, k):
    """Calculate recall@k between ground truth and test results"""
    if ground_truth_indices.shape != test_indices.shape:
        return 0.0

    total_recall = 0
    n_queries = ground_truth_indices.shape[0]

    for i in range(n_queries):
        ground_truth_set = set(ground_truth_indices[i, :k])
        test_set = set(test_indices[i, :k])
        recall = len(ground_truth_set.intersection(test_set)) / k
        total_recall += recall

    return total_recall / n_queries

def estimate_hnsw_memory(n_vectors, dimension):
    """Estimate memory usage for HNSW index"""
    # Rough estimation: vectors + graph structure
    vector_memory = n_vectors * dimension * 4  # float32
    graph_memory = n_vectors * 32 * 4  # Rough estimate for graph connections
    return vector_memory + graph_memory

def create_performance_summary(faiss_results: Dict, hnswlib_results: Dict) -> pd.DataFrame:
    """Create a comprehensive performance summary"""
    print("\n📈 Creating Performance Summary...")

    summary_data = []

    # Process FAISS results
    for dataset_name, methods in faiss_results.items():
        for method_name, metrics in methods.items():
            summary_data.append({
                'Library': 'FAISS',
                'Dataset': dataset_name,
                'Method': method_name,
                'Build_Time': metrics['build_time'],
                'Search_Time': metrics['search_time'],
                'Memory_MB': metrics['memory_usage'] / (1024 * 1024),
                'Accuracy': metrics.get('accuracy', 'N/A'),
                'Avg_Distance': metrics['avg_distance']
            })

    # Process hnswlib results
    for dataset_name, configs in hnswlib_results.items():
        for config_name, metrics in configs.items():
            summary_data.append({
                'Library': 'hnswlib',
                'Dataset': dataset_name,
                'Method': config_name,
                'Build_Time': metrics['build_time'],
                'Search_Time': metrics['search_time'],
                'Memory_MB': metrics['memory_usage'] / (1024 * 1024),
                'Accuracy': 'Approx',
                'Avg_Distance': metrics['avg_distance']
            })

    df = pd.DataFrame(summary_data)
    return df

def find_optimal_configurations(df: pd.DataFrame) -> Dict:
    """Find optimal configurations for different use cases"""
    print("\n🎯 Finding Optimal Configurations...")

    recommendations = {}

    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]

        # Fastest build time
        fastest_build = dataset_df.loc[dataset_df['Build_Time'].idxmin()]

        # Fastest search time
        fastest_search = dataset_df.loc[dataset_df['Search_Time'].idxmin()]

        # Best accuracy (for FAISS methods with numeric accuracy)
        faiss_df = dataset_df[dataset_df['Library'] == 'FAISS']
        if not faiss_df.empty and faiss_df['Accuracy'].dtype != 'object':
            best_accuracy = faiss_df.loc[faiss_df['Accuracy'].idxmax()]
        else:
            best_accuracy = None

        # Most memory efficient
        most_efficient = dataset_df.loc[dataset_df['Memory_MB'].idxmin()]

        recommendations[dataset] = {
            'fastest_build': {
                'library': fastest_build['Library'],
                'method': fastest_build['Method'],
                'time': fastest_build['Build_Time']
            },
            'fastest_search': {
                'library': fastest_search['Library'],
                'method': fastest_search['Method'],
                'time': fastest_search['Search_Time']
            },
            'best_accuracy': {
                'library': best_accuracy['Library'] if best_accuracy is not None else 'N/A',
                'method': best_accuracy['Method'] if best_accuracy is not None else 'N/A',
                'accuracy': best_accuracy['Accuracy'] if best_accuracy is not None else 'N/A'
            },
            'most_memory_efficient': {
                'library': most_efficient['Library'],
                'method': most_efficient['Method'],
                'memory_mb': most_efficient['Memory_MB']
            }
        }

    return recommendations

def save_comprehensive_results(data, faiss_results, hnswlib_results, summary_df, recommendations):
    """Save all results to files"""
    print("\n💾 Saving Results...")

    # Save dataset info
    dataset_info = {}
    for name, dataset in data.items():
        dataset_info[name] = {
            'n_vectors': dataset['n_vectors'],
            'dimension': dataset['dimension'],
            'n_queries': dataset['n_queries']
        }

    with open('day6_dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print("✅ Saved dataset info to day6_dataset_info.json")

    # Save detailed results
    all_results = {
        'faiss_results': faiss_results,
        'hnswlib_results': hnswlib_results,
        'recommendations': recommendations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('day6_detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("✅ Saved detailed results to day6_detailed_results.json")

    # Save performance summary
    summary_df.to_csv('day6_performance_summary.csv', index=False)
    print("✅ Saved performance summary to day6_performance_summary.csv")

    # Save embeddings from largest dataset for potential use in next days
    if 'large' in data:
        np.save('day6_large_embeddings.npy', data['large']['database'])
        np.save('day6_large_queries.npy', data['large']['queries'])
        print("✅ Saved large dataset embeddings for future use")

def main():
    """Main execution function for Day 6"""
    print("=" * 80)
    print("          Day 6: ANN Libraries - FAISS and hnswlib")
    print("          Task: Build FAISS index and perform top-k nearest neighbor queries")
    print("=" * 80)

    try:
        # Check imports
        import faiss
        import hnswlib
        print("✅ Successfully imported FAISS and hnswlib")
        print(f"✅ FAISS version: {faiss.__version__}")

        # Create comprehensive dataset
        data = create_comprehensive_dataset()

        # Benchmark FAISS
        k = 10  # Number of nearest neighbors
        faiss_results = benchmark_faiss_indexes(data, k)

        # Benchmark hnswlib
        hnswlib_results = benchmark_hnswlib(data, k)

        # Create performance summary
        summary_df = create_performance_summary(faiss_results, hnswlib_results)

        print("\n📊 Performance Summary:")
        print(summary_df.to_string(index=False))

        # Find optimal configurations
        recommendations = find_optimal_configurations(summary_df)

        print("\n🏆 Recommendations by Use Case:")
        for dataset, recs in recommendations.items():
            print(f"\n{dataset.upper()} Dataset:")
            print(f"  • Fastest Build:    {recs['fastest_build']['library']} {recs['fastest_build']['method']} ({recs['fastest_build']['time']:.4f}s)")
            print(f"  • Fastest Search:   {recs['fastest_search']['library']} {recs['fastest_search']['method']} ({recs['fastest_search']['time']:.4f}s)")
            print(f"  • Best Accuracy:    {recs['best_accuracy']['library']} {recs['best_accuracy']['method']} ({recs['best_accuracy']['accuracy']})")
            print(f"  • Memory Efficient: {recs['most_memory_efficient']['library']} {recs['most_memory_efficient']['method']} ({recs['most_memory_efficient']['memory_mb']:.1f}MB)")

        # Save all results
        save_comprehensive_results(data, faiss_results, hnswlib_results, summary_df, recommendations)

        print("\n" + "=" * 80)
        print("✅ Day 6 Task Completed Successfully!")
        print("=" * 80)
        print("🎯 What we accomplished:")
        print("✅ Created realistic datasets of varying sizes")
        print("✅ Benchmarked multiple FAISS index types")
        print("✅ Tested hnswlib with different parameter configurations")
        print("✅ Compared performance across build time, search time, memory, accuracy")
        print("✅ Generated recommendations for different use cases")
        print("✅ Saved comprehensive results for analysis")

        print("\n📊 Key Insights:")
        total_vectors = sum(d['n_vectors'] for d in data.values())
        print(f"• Processed {total_vectors:,} total vectors across all datasets")
        print(f"• Tested {len(summary_df)} different configurations")
        print(f"• Generated performance data for {len(data)} dataset sizes")

        print("\n🎓 Skills Learned:")
        print("• Building and configuring FAISS indexes")
        print("• Optimizing hnswlib parameters for different scenarios")
        print("• Understanding speed vs accuracy trade-offs in ANN")
        print("• Benchmarking and comparing vector search libraries")
        print("• Selecting optimal configurations for specific use cases")

        print("\n➡️  Ready for Day 7: Vector Databases (Chroma, Weaviate)")

        # Cleanup option
        cleanup = input("\nKeep generated files? (y/n): ").lower().strip()
        if cleanup != 'y':
            files_to_remove = [
                'day6_dataset_info.json', 'day6_detailed_results.json',
                'day6_performance_summary.csv', 'day6_large_embeddings.npy',
                'day6_large_queries.npy'
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
            print("✅ Temporary files cleaned up")
        else:
            print("✅ Files preserved for further analysis")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 To fix:")
        print("1. Run: python setup_day6.py")
        print("2. Or manually: pip install faiss-cpu hnswlib")
        return False

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔧 Try:")
        print("1. Run: python test_day6.py")
        print("2. Check available memory (large datasets need RAM)")
        print("3. Reduce dataset sizes if running on limited hardware")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
