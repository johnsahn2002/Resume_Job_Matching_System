# 🎯 Advanced Resume-Job Matching System

An AI-powered web application that helps job seekers optimize their resumes and find the best job matches using advanced NLP techniques, machine learning algorithms, and web scraping capabilities.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 📄 **Resume Analysis**
- **Multi-format support**: PDF, DOCX, TXT files
- **Keyword extraction** using TF-IDF algorithms
- **Skills categorization** (technical, soft skills, programming languages)
- **Word cloud visualization** of key terms
- **Text preprocessing** and cleaning

### 🔍 **Job Matching**
- **Multiple similarity algorithms**: TF-IDF + Cosine Similarity
- **Semantic matching** using Sentence Transformers
- **Combined scoring system** for comprehensive results
- **Interactive visualizations** of match scores
- **Batch job comparison** capabilities

### 🌐 **Web Scraping**
- **Automated job search** with keyword targeting
- **Location-based filtering** 
- **Real-time job matching** against your resume
- **Structured data extraction** from job postings
- **Match score calculation** for scraped positions

### 🧪 **A/B Testing**
- **Resume variant generation** with different strategies
- **Keyword emphasis** testing
- **Skills highlighting** experiments
- **Performance comparison** across versions
- **Statistical improvement analysis**

### 📊 **Gap Analysis & Recommendations**
- **Skills gap identification** by category
- **Priority-based learning recommendations**
- **Action plan generation**
- **Learning resource suggestions**
- **Progress tracking** and metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resume-job-matcher.git
cd resume-job-matcher
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download spaCy language model** (optional but recommended)
```bash
python -m spacy download en_core_web_sm
```

4. **Run the application**
```bash
streamlit run resume_matcher.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 💻 Usage Guide

### Step 1: Resume Analysis
1. Navigate to the **Resume Analysis** page
2. Upload your resume (PDF, DOCX, or TXT format)
3. View extracted keywords and skills analysis
4. Review the generated word cloud

### Step 2: Job Matching
1. Go to the **Job Matching** page
2. Add job descriptions via:
   - Manual text entry
   - File upload
   - Sample job descriptions
3. View match scores and rankings
4. Analyze similarity metrics

### Step 3: Web Scraping (Optional)
1. Visit the **Web Scraping** page
2. Enter relevant job keywords
3. Specify location (optional)
4. View scraped jobs with match scores

### Step 4: A/B Testing
1. Access the **A/B Testing** page
2. Configure resume variants:
   - Keyword emphasis
   - Skills highlighting
3. Compare performance across versions
4. Identify the best-performing variant

### Step 5: Gap Analysis
1. Navigate to **Gap Analysis** page
2. Review skills gaps by category
3. View prioritized learning recommendations
4. Access suggested learning resources

## 🛠 Technical Architecture

### Core Components

```
├── DataIngestion          # File processing and text extraction
├── NLPProcessor          # Text cleaning, tokenization, keyword extraction
├── MatchingEngine        # TF-IDF, cosine similarity, embeddings
├── WebScraper           # Job posting extraction and processing
├── RecommendationEngine # Gap analysis and skill prioritization
├── ABTester            # Resume variant testing and comparison
└── StreamlitUI         # Web interface and visualization
```

### Key Algorithms
- **TF-IDF (Term Frequency-Inverse Document Frequency)** for keyword importance
- **Cosine Similarity** for document comparison
- **Sentence Transformers** for semantic understanding
- **Named Entity Recognition** for skill extraction
- **Statistical Analysis** for A/B testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
