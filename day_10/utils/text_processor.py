import re
import string
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text preprocessing and cleaning utilities"""

    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_special_chars: bool = False,
                 preserve_structure: bool = True):
        """
        Initialize text processor

        Args:
            remove_extra_whitespace: Remove extra whitespace
            normalize_unicode: Normalize Unicode characters
            remove_special_chars: Remove special characters
            preserve_structure: Preserve document structure (headers, lists, etc.)
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.preserve_structure = preserve_structure

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        cleaned = text

        # Unicode normalization
        if self.normalize_unicode:
            import unicodedata
            cleaned = unicodedata.normalize('NFKC', cleaned)

        # Remove or replace problematic characters
        cleaned = self._fix_encoding_issues(cleaned)

        # Handle whitespace
        if self.remove_extra_whitespace:
            cleaned = self._normalize_whitespace(cleaned)

        # Remove special characters if requested
        if self.remove_special_chars:
            cleaned = self._remove_special_characters(cleaned)

        # Fix common text issues
        cleaned = self._fix_common_issues(cleaned)

        return cleaned.strip()

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        # Common encoding replacements
        replacements = {
            'â€™': "'",  # Right single quotation mark
            'â€œ': '"',  # Left double quotation mark
            'â€\x9d': '"',  # Right double quotation mark
            'â€"': '–',  # En dash
            'â€"': '—',  # Em dash
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã\xa0': ' ',  # Non-breaking space
            '\ufeff': '',  # Byte order mark
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure"""
        if self.preserve_structure:
            # Preserve paragraph breaks
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r' *\n *', '\n', text)  # Remove spaces around line breaks
        else:
            # Aggressive whitespace normalization
            text = re.sub(r'\s+', ' ', text)

        return text

    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving structure"""
        if self.preserve_structure:
            # Keep structural characters
            allowed_chars = string.ascii_letters + string.digits + string.whitespace + '.,!?;:-()[]{}"\'/\n\t'
            cleaned = ''.join(char for char in text if char in allowed_chars)
        else:
            # More aggressive removal
            cleaned = ''.join(char for char in text if char.isalnum() or char.isspace())

        return cleaned

    def _fix_common_issues(self, text: str) -> str:
        """Fix common text processing issues"""
        # Fix multiple periods
        text = re.sub(r'\.{3,}', '...', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentence end

        # Fix common contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text"""
        metadata = {}

        # Character and word counts
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['line_count'] = len(text.split('\n'))

        # Detect language (simplified)
        metadata['language'] = self._detect_language_simple(text)

        # Structure analysis
        metadata['has_headers'] = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
        metadata['has_lists'] = bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE))
        metadata['has_code'] = bool(re.search(r'```|`[^`]+`', text))
        metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])

        return metadata

    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on common words"""
        # Very basic language detection
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        words = set(word.lower() for word in re.findall(r'\b\w+\b', text[:1000]))  # Check first 1000 chars
        english_score = len(words.intersection(english_words))

        if english_score >= 3:
            return 'en'
        else:
            return 'unknown'

class DocumentLoader:
    """Load documents from various file formats"""

    def __init__(self):
        self.supported_formats = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.xml', '.json', '.csv'}

        # Try to import optional dependencies
        self.docx_available = self._check_docx()
        self.pdf_available = self._check_pdf()

        if self.docx_available:
            self.supported_formats.add('.docx')
        if self.pdf_available:
            self.supported_formats.add('.pdf')

    def _check_docx(self) -> bool:
        """Check if python-docx is available"""
        try:
            import docx
            return True
        except ImportError:
            return False

    def _check_pdf(self) -> bool:
        """Check if PyPDF2 is available"""
        try:
            import PyPDF2
            return True
        except ImportError:
            return False

    def load_document(self, file_path: str) -> str:
        """
        Load document from file

        Args:
            file_path: Path to document file

        Returns:
            Document text content
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")

        try:
            if file_ext in {'.txt', '.md', '.rst', '.py', '.js', '.html', '.xml'}:
                return self._load_text_file(file_path)
            elif file_ext == '.json':
                return self._load_json_file(file_path)
            elif file_ext == '.csv':
                return self._load_csv_file(file_path)
            elif file_ext == '.docx' and self.docx_available:
                return self._load_docx_file(file_path)
            elif file_ext == '.pdf' and self.pdf_available:
                return self._load_pdf_file(file_path)
            else:
                # Fallback to text loading
                return self._load_text_file(file_path)

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise

    def _load_text_file(self, file_path: Path) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _load_json_file(self, file_path: Path) -> str:
        """Load JSON file and convert to readable text"""
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to readable text
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _load_csv_file(self, file_path: Path) -> str:
        """Load CSV file and convert to readable text"""
        import csv

        text_content = []

        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if headers:
                text_content.append("Headers: " + ", ".join(headers))
                text_content.append("")

            for i, row in enumerate(reader):
                if i < 100:  # Limit to first 100 rows
                    text_content.append(" | ".join(str(cell) for cell in row))
                else:
                    text_content.append(f"... and {i} more rows")
                    break

        return "\n".join(text_content)

    def _load_docx_file(self, file_path: Path) -> str:
        """Load Word document"""
        import docx

        doc = docx.Document(file_path)
        text_content = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())

        return "\n".join(text_content)

    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF document"""
        import PyPDF2

        text_content = []

        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"=== Page {page_num + 1} ===")
                        text_content.append(page_text.strip())
                        text_content.append("")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

        return "\n".join(text_content)

    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a document"""
        file_path = Path(file_path)

        info = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "supported": file_path.suffix.lower() in self.supported_formats,
            "exists": file_path.exists()
        }

        if file_path.exists():
            import os
            stat = file_path.stat()
            info.update({
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "readable": os.access(file_path, os.R_OK)
            })

        return info

    def batch_load_documents(self, directory: str, 
                           pattern: str = "*",
                           recursive: bool = False) -> Dict[str, str]:
        """Load multiple documents from a directory"""
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        # Filter supported files
        supported_files = [f for f in files if f.suffix.lower() in self.supported_formats]

        documents = {}

        for file_path in supported_files:
            try:
                content = self.load_document(str(file_path))
                documents[str(file_path)] = content
                logger.info(f"Loaded document: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                documents[str(file_path)] = f"ERROR: {str(e)}"

        return documents

def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)

    samples = {
    "ai_overview.txt": """
# Artificial Intelligence Overview

Artificial Intelligence (AI) represents one of the most transformative technologies of our time. It encompasses various techniques and approaches that enable machines to simulate human intelligence.

## Machine Learning
Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. Key approaches include:

- Supervised Learning: Learning from labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data  
- Reinforcement Learning: Learning through interaction with an environment

## Natural Language Processing
Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Applications include:

- Sentiment analysis
- Language translation
- Text summarization
- Question answering systems

## Computer Vision
Computer Vision allows machines to interpret and understand visual information from the world. This includes:

- Image recognition and classification
- Object detection and tracking
- Facial recognition
- Medical image analysis

## Applications
AI has found applications across numerous industries:

### Healthcare
- Medical diagnosis assistance
- Drug discovery and development
- Personalized treatment plans
- Medical imaging analysis

### Finance
- Algorithmic trading
- Fraud detection
- Credit risk assessment
- Robo-advisors for investment

### Transportation
- Autonomous vehicles
- Traffic optimization
- Route planning
- Predictive maintenance

## Ethical Considerations
As AI becomes more prevalent, important ethical questions arise:

- Bias in algorithms and decision-making
- Privacy and data protection
- Job displacement and economic impact
- Transparency and explainability of AI systems
- Safety and reliability of autonomous systems

The future of AI depends not only on technological advances but also on how we address these ethical challenges and ensure AI benefits all of humanity.
    """,
    "programming_guide.md": """
# Python Programming Best Practices

## Code Organization

### Project Structure
A well-organized project structure is crucial for maintainability:

```
project/
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── modules/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

### Naming Conventions

## Error Handling

Always use proper exception handling:

```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    handle_error(e)
finally:
    cleanup_resources()
```

## Documentation

Write clear docstrings for all functions and classes:

```python
```

## Testing

Write comprehensive tests for your code:


## Performance Optimization

        """,
        "data_science.txt": """
Data Science Fundamentals

Data science combines statistics, programming, and domain expertise to extract insights from data. The typical data science workflow includes:

1. Data Collection
   - Gathering data from various sources
   - APIs, databases, web scraping
   - Ensuring data quality and completeness

2. Data Cleaning and Preprocessing
   - Handling missing values
   - Removing outliers and duplicates
   - Data type conversions
   - Feature engineering

3. Exploratory Data Analysis (EDA)
   - Understanding data distributions
   - Identifying patterns and relationships
   - Statistical summaries and visualizations
   - Correlation analysis

4. Model Building and Selection
   - Choosing appropriate algorithms
   - Training and validation
   - Hyperparameter tuning
   - Cross-validation techniques

5. Model Evaluation
   - Performance metrics selection
   - Confusion matrices and ROC curves
   - Precision, recall, and F1-scores
   - Business impact assessment

6. Deployment and Monitoring
   - Model deployment strategies
   - A/B testing frameworks
   - Performance monitoring
   - Model maintenance and updates

Key Tools and Technologies:
- Python: pandas, scikit-learn, matplotlib
- R: dplyr, ggplot2, caret
- SQL: Data querying and manipulation
- Big Data: Spark, Hadoop
- Visualization: Tableau, Power BI
- Cloud Platforms: AWS, GCP, Azure

Statistical Concepts:
- Descriptive vs Inferential Statistics
- Probability distributions
- Hypothesis testing
- Regression analysis
- Time series analysis
- Bayesian methods

Machine Learning Types:
- Supervised Learning: Classification and regression
- Unsupervised Learning: Clustering and dimensionality reduction
- Semi-supervised Learning: Combining labeled and unlabeled data
- Reinforcement Learning: Learning through rewards and penalties
        """
    }

    for filename, content in samples.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created sample document: {file_path}")

    return sample_dir

if __name__ == "__main__":
    # Test the utilities
    print("Creating sample documents...")
    sample_dir = create_sample_documents()

    print("\nTesting document loader...")
    loader = DocumentLoader()
    processor = TextProcessor()

    for file_path in sample_dir.glob("*.txt"):
        print(f"\nProcessing: {file_path.name}")
        try:
            content = loader.load_document(str(file_path))
            cleaned = processor.clean_text(content)
            metadata = processor.extract_metadata(cleaned)

            print(f"  Original length: {len(content)}")
            print(f"  Cleaned length: {len(cleaned)}")
            print(f"  Word count: {metadata['word_count']}")
            print(f"  Has headers: {metadata['has_headers']}")
            print(f"  Language: {metadata['language']}")

        except Exception as e:
            print(f"  Error: {e}")
