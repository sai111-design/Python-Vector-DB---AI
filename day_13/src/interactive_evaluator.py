#!/usr/bin/env python3
"""
Day 13 - Interactive Retrieval Evaluator using Streamlit
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from main import RetrievalEvaluator, RetrievalResult, EvaluationMetrics
    from benchmark_datasets import BenchmarkLoader
except ImportError:
    st.error("Could not import required modules. Please ensure main.py and benchmark_datasets.py are in the src folder.")
    st.stop()

def load_sample_data():
    """Load sample evaluation data"""
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = RetrievalEvaluator()

        # Load benchmark data
        loader = BenchmarkLoader()
        try:
            dataset = loader.create_msmarco_sample()

            # Add to evaluator
            for doc in dataset.documents:
                from main import Document
                evaluator_doc = Document(
                    id=doc["id"],
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                st.session_state.evaluator.add_document(evaluator_doc)

            for query in dataset.queries:
                from main import Query
                evaluator_query = Query(
                    id=query["id"],
                    text=query["text"],
                    relevant_doc_ids=set(query["relevant_doc_ids"]),
                    metadata=query["metadata"]
                )
                st.session_state.evaluator.add_query(evaluator_query)

        except Exception as e:
            st.error(f"Error loading data: {e}")

def create_mock_results():
    """Create mock retrieval results for demo"""
    return [
        RetrievalResult(
            query_id="q_geo_1",
            retrieved_docs=[
                ("msmarco_1", 0.95),  # Correct
                ("msmarco_6", 0.73),  # Incorrect
                ("msmarco_4", 0.68),  # Incorrect  
                ("msmarco_2", 0.61),  # Incorrect
                ("msmarco_3", 0.55)   # Incorrect
            ],
            retrieval_time=0.042
        ),
        RetrievalResult(
            query_id="q_tech_1",
            retrieved_docs=[
                ("msmarco_2", 0.92),  # Correct
                ("msmarco_5", 0.78),  # Related
                ("msmarco_8", 0.71),  # Related
                ("msmarco_1", 0.64),  # Incorrect
                ("msmarco_7", 0.52)   # Incorrect
            ],
            retrieval_time=0.038
        ),
        RetrievalResult(
            query_id="q_env_1",
            retrieved_docs=[
                ("msmarco_3", 0.89),  # Correct
                ("msmarco_7", 0.71),  # Related (environment)
                ("msmarco_2", 0.65),  # Incorrect
                ("msmarco_5", 0.61),  # Incorrect
                ("msmarco_1", 0.53)   # Incorrect
            ],
            retrieval_time=0.045
        )
    ]

def main():
    st.set_page_config(
        page_title="Day 13 - Retrieval Evaluator",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Day 13 - Interactive Retrieval Evaluator")
    st.markdown("Evaluate your retrieval system with precision, recall, and advanced IR metrics")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Quick Evaluation", "Detailed Analysis", "Upload Data", "Benchmark Comparison"]
    )

    # Load data
    load_sample_data()

    if page == "Overview":
        show_overview()
    elif page == "Quick Evaluation":
        show_quick_evaluation()
    elif page == "Detailed Analysis":
        show_detailed_analysis()
    elif page == "Upload Data":
        show_upload_data()
    elif page == "Benchmark Comparison":
        show_benchmark_comparison()

def show_overview():
    st.header("📊 Retrieval Evaluation Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What is Retrieval Evaluation?")
        st.write("""
        Retrieval evaluation measures how well a search system finds relevant documents for user queries.

        **Key Metrics:**
        - **Precision@K**: Fraction of retrieved documents that are relevant
        - **Recall@K**: Fraction of relevant documents that are retrieved  
        - **F1 Score**: Harmonic mean of precision and recall
        - **MAP**: Mean Average Precision across all queries
        - **NDCG**: Normalized Discounted Cumulative Gain
        - **MRR**: Mean Reciprocal Rank
        """)

    with col2:
        st.subheader("Sample Data Loaded")
        if 'evaluator' in st.session_state:
            evaluator = st.session_state.evaluator
            st.metric("Documents", len(evaluator.documents))
            st.metric("Queries", len(evaluator.queries))

            # Show sample documents
            if st.checkbox("Show sample documents"):
                docs_data = []
                for doc_id, doc in list(evaluator.documents.items())[:3]:
                    docs_data.append({
                        "ID": doc_id,
                        "Content": doc.content[:100] + "...",
                        "Source": doc.metadata.get("source", "N/A")
                    })
                st.dataframe(pd.DataFrame(docs_data))

def show_quick_evaluation():
    st.header("⚡ Quick Evaluation")

    if 'evaluator' not in st.session_state:
        st.error("No evaluator loaded. Please go to Overview first.")
        return

    evaluator = st.session_state.evaluator

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        k_values = st.multiselect(
            "Select K values",
            [1, 3, 5, 10, 20],
            default=[3, 5, 10]
        )

        if st.button("Run Evaluation", type="primary"):
            # Create mock results
            results = create_mock_results()

            # Update evaluator k values
            evaluator.k_values = k_values

            # Evaluate
            k_results = evaluator.evaluate_at_multiple_k(results)

            # Store in session state
            st.session_state.evaluation_results = k_results
            st.success(f"Evaluation completed for {len(results)} queries!")

    with col2:
        st.subheader("Results")

        if 'evaluation_results' in st.session_state:
            results_data = []
            for k, metrics in st.session_state.evaluation_results.items():
                results_data.append({
                    "K": k,
                    "Precision": f"{metrics.precision:.4f}",
                    "Recall": f"{metrics.recall:.4f}",
                    "F1": f"{metrics.f1:.4f}",
                    "MAP": f"{metrics.map_score:.4f}",
                    "NDCG": f"{metrics.ndcg:.4f}",
                    "Hit Rate": f"{metrics.hit_rate:.4f}"
                })

            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)

            # Visualization
            st.subheader("Metrics Visualization")

            chart_data = pd.DataFrame({
                'K': [r['K'] for r in results_data],
                'Precision': [float(r['Precision']) for r in results_data],
                'Recall': [float(r['Recall']) for r in results_data],
                'F1': [float(r['F1']) for r in results_data]
            })

            st.line_chart(chart_data.set_index('K'))

def show_detailed_analysis():
    st.header("🔬 Detailed Analysis")

    if 'evaluation_results' not in st.session_state:
        st.warning("Please run Quick Evaluation first to see detailed analysis.")
        return

    results = st.session_state.evaluation_results

    # Best performing K
    best_k = max(results.keys(), key=lambda k: results[k].f1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best K (by F1)",
            best_k,
            f"F1: {results[best_k].f1:.4f}"
        )

    with col2:
        st.metric(
            "Highest Precision",
            min(results.keys()),  # Usually K=1 has highest precision
            f"{results[min(results.keys())].precision:.4f}"
        )

    with col3:
        st.metric(
            "Highest Recall", 
            max(results.keys()),  # Usually highest K has highest recall
            f"{results[max(results.keys())].recall:.4f}"
        )

    # Performance analysis
    st.subheader("Performance Analysis")

    analysis_data = []
    for k, metrics in results.items():
        analysis_data.append([
            k,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.map_score,
            metrics.ndcg
        ])

    analysis_df = pd.DataFrame(
        analysis_data,
        columns=['K', 'Precision', 'Recall', 'F1', 'MAP', 'NDCG']
    )

    # Correlation analysis
    st.subheader("Metric Correlations")
    corr_matrix = analysis_df[['Precision', 'Recall', 'F1', 'MAP', 'NDCG']].corr()
    st.dataframe(corr_matrix.style.background_gradient())

    # Insights
    st.subheader("Key Insights")

    precision_trend = "increases" if results[max(results.keys())].precision > results[min(results.keys())].precision else "decreases"
    recall_trend = "increases" if results[max(results.keys())].recall > results[min(results.keys())].recall else "decreases"

    st.write(f"""
    **Trend Analysis:**
    - Precision {precision_trend} as K increases
    - Recall {recall_trend} as K increases
    - Best F1 score achieved at K={best_k}
    - MAP score: {results[best_k].map_score:.4f} indicates {'good' if results[best_k].map_score > 0.5 else 'moderate'} average precision
    """)

def show_upload_data():
    st.header("📤 Upload Your Data")

    st.write("Upload your own ground truth data for evaluation")

    with st.expander("📋 Data Format Guide"):
        st.write("""
        **Required JSON format:**
        ```json
        {
          "documents": [
            {
              "id": "doc_1",
              "content": "Document content...",
              "metadata": {"source": "example"}
            }
          ],
          "queries": [
            {
              "id": "query_1", 
              "text": "Query text",
              "relevant_doc_ids": ["doc_1"],
              "metadata": {"category": "test"}
            }
          ]
        }
        ```
        """)

    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type="json",
        help="Upload ground truth data in the specified format"
    )

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)

            # Validate format
            if "documents" not in data or "queries" not in data:
                st.error("Invalid format: Missing 'documents' or 'queries' keys")
                return

            # Load data
            evaluator = RetrievalEvaluator()

            # Add documents
            for doc_data in data["documents"]:
                from main import Document
                doc = Document(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    metadata=doc_data.get("metadata", {})
                )
                evaluator.add_document(doc)

            # Add queries
            for query_data in data["queries"]:
                from main import Query
                query = Query(
                    id=query_data["id"],
                    text=query_data["text"],
                    relevant_doc_ids=set(query_data["relevant_doc_ids"]),
                    metadata=query_data.get("metadata", {})
                )
                evaluator.add_query(query)

            # Update session state
            st.session_state.evaluator = evaluator

            st.success(f"Successfully loaded {len(data['documents'])} documents and {len(data['queries'])} queries!")

            # Show preview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sample Documents")
                docs_preview = []
                for doc in data["documents"][:3]:
                    docs_preview.append({
                        "ID": doc["id"],
                        "Content": doc["content"][:100] + "..."
                    })
                st.dataframe(pd.DataFrame(docs_preview))

            with col2:
                st.subheader("Sample Queries")
                queries_preview = []
                for query in data["queries"][:3]:
                    queries_preview.append({
                        "ID": query["id"],
                        "Text": query["text"],
                        "Relevant Docs": len(query["relevant_doc_ids"])
                    })
                st.dataframe(pd.DataFrame(queries_preview))

        except Exception as e:
            st.error(f"Error loading file: {e}")

def show_benchmark_comparison():
    st.header("📊 Benchmark Comparison")

    st.write("Compare performance across different benchmark datasets")

    # Create sample benchmark results
    benchmark_results = {
        "MS MARCO Sample": {
            "Precision@5": 0.67,
            "Recall@5": 0.45,
            "F1@5": 0.54,
            "MAP": 0.58,
            "NDCG@5": 0.62
        },
        "TREC Sample": {
            "Precision@5": 0.72,
            "Recall@5": 0.48,
            "F1@5": 0.57,
            "MAP": 0.61,
            "NDCG@5": 0.65
        },
        "Custom Dataset": {
            "Precision@5": 0.64,
            "Recall@5": 0.42,
            "F1@5": 0.51,
            "MAP": 0.55,
            "NDCG@5": 0.59
        }
    }

    # Convert to DataFrame
    df = pd.DataFrame(benchmark_results).T

    st.subheader("Benchmark Results Comparison")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

    # Visualization
    st.subheader("Performance Comparison")

    # Bar chart
    chart_data = df.reset_index()
    chart_data = chart_data.melt(id_vars=['index'], var_name='Metric', value_name='Score')
    chart_data = chart_data.rename(columns={'index': 'Dataset'})

    import altair as alt

    chart = alt.Chart(chart_data).mark_bar().add_selection(
        alt.selection_interval()
    ).encode(
        x=alt.X('Dataset:O', title='Dataset'),
        y=alt.Y('Score:Q', title='Score'),
        color=alt.Color('Metric:N', title='Metric'),
        column=alt.Column('Metric:N', title='Metric')
    ).resolve_scale(
        y='independent'
    )

    st.altair_chart(chart, use_container_width=True)

    # Best performer analysis
    st.subheader("Best Performers")

    for metric in df.columns:
        best_dataset = df[metric].idxmax()
        best_score = df[metric].max()
        st.write(f"**{metric}**: {best_dataset} ({best_score:.3f})")

if __name__ == "__main__":
    main()
