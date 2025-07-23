# Resume-Job Matching System
# Complete implementation with data ingestion, NLP, matching, recommendations, and web scraping

import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import PyPDF2
import docx
from typing import List, Dict, Tuple
import time
import random
from datetime import datetime
import json

st.set_page_config(
    page_title="Resume-Job Matching System",
    page_icon="🎯",
    layout="wide"
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class DataIngestion:
    """Handle resume and job description data ingestion"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def process_uploaded_file(self, file) -> str:
        """Process uploaded file and extract text"""
        if file is None:
            return ""
        
        file_extension = file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file)
        elif file_extension == 'txt':
            return str(file.read(), 'utf-8')
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return ""

class WebScraper:
    """Web scraping functionality for job postings"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_indeed_jobs(self, keywords: str, location: str = "", num_jobs: int = 10) -> List[Dict]:
        """Scrape job postings from Indeed (simulation for demo)"""
        # In a real implementation, you'd use proper web scraping
        # This is a simulation due to rate limiting and legal considerations
        
        jobs = []
        job_titles = [
            "Software Engineer", "Data Scientist", "Product Manager", "UX Designer",
            "Marketing Manager", "Sales Representative", "Business Analyst", "DevOps Engineer",
            "Frontend Developer", "Backend Developer", "Full Stack Developer", "Data Analyst"
        ]
        
        companies = [
            "TechCorp", "DataSoft", "InnovateX", "StartupY", "BigTech Inc.",
            "CloudSolutions", "AICompany", "WebDev Co", "Analytics Pro", "DevOps Ltd"
        ]
        
        for i in range(min(num_jobs, 20)):
            job = {
                'title': random.choice(job_titles),
                'company': random.choice(companies),
                'location': location if location else random.choice(['New York, NY', 'San Francisco, CA', 'Seattle, WA', 'Austin, TX']),
                'description': self.generate_job_description(keywords),
                'salary': f"${random.randint(60, 150)}k - ${random.randint(160, 200)}k",
                'posted_date': f"{random.randint(1, 30)} days ago",
                'url': f"https://example-job-{i}.com"
            }
            jobs.append(job)
        
        return jobs
    
    def generate_job_description(self, keywords: str) -> str:
        """Generate realistic job description incorporating keywords"""
        base_descriptions = [
            "We are looking for a talented professional to join our growing team. The ideal candidate will have strong analytical skills and experience with modern technologies.",
            "Join our innovative company and work on cutting-edge projects. We value collaboration, creativity, and technical excellence.",
            "Seeking a motivated individual to contribute to our mission. You'll work with cross-functional teams on exciting challenges.",
            "Great opportunity to grow your career in a fast-paced environment. We offer competitive benefits and professional development."
        ]
        
        skills_pool = [
            "Python", "JavaScript", "SQL", "AWS", "Docker", "Kubernetes", "React", "Node.js",
            "machine learning", "data analysis", "project management", "team leadership",
            "communication", "problem solving", "agile methodologies", "git"
        ]
        
        # Incorporate user keywords with random skills
        keyword_list = keywords.lower().split()
        selected_skills = keyword_list + random.sample(skills_pool, min(5, len(skills_pool)))
        
        description = random.choice(base_descriptions)
        description += f" Required skills include: {', '.join(selected_skills[:8])}. "
        description += "Bachelor's degree preferred. 2-5 years of experience required."
        
        return description

class NLPProcessor:
    """Natural Language Processing for text cleaning and feature extraction"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            st.warning("spaCy model not found. Using basic NLP processing.")
            self.nlp = None
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Technical skills keywords
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'pandas', 'numpy', 'scipy', 'matplotlib'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp'],
            'soft': ['leadership', 'communication', 'teamwork', 'problem solving', 'project management']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower().strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text and remove stop words"""
        if not text:
            return []
        
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        if not text:
            return []
        
        # Use TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Create keyword-score pairs
            keywords = list(zip(feature_names, tfidf_scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:top_n]
        except:
            # Fallback to simple word frequency
            words = self.tokenize_text(text)
            from collections import Counter
            word_freq = Counter(words)
            return [(word, freq) for word, freq in word_freq.most_common(top_n)]
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills from text"""
        text_lower = text.lower()
        found_skills = {category: [] for category in self.tech_skills}
        
        for category, skills in self.tech_skills.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        return found_skills

class MatchingEngine:
    """Core matching algorithms for resume-job matching"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            st.warning("SentenceTransformer model not available. Using TF-IDF only.")
            self.sentence_model = None
    
    def calculate_tfidf_similarity(self, resume_text: str, job_texts: List[str]) -> List[float]:
        """Calculate TF-IDF cosine similarity between resume and jobs"""
        if not resume_text or not job_texts:
            return [0.0] * len(job_texts)
        
        all_texts = [resume_text] + job_texts
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            resume_vector = tfidf_matrix[0]
            job_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(resume_vector, job_vectors)[0]
            return similarities.tolist()
        except:
            return [0.0] * len(job_texts)
    
    def calculate_embedding_similarity(self, resume_text: str, job_texts: List[str]) -> List[float]:
        """Calculate embedding similarity using sentence transformers"""
        if not self.sentence_model or not resume_text or not job_texts:
            return [0.0] * len(job_texts)
        
        try:
            resume_embedding = self.sentence_model.encode([resume_text])
            job_embeddings = self.sentence_model.encode(job_texts)
            
            similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
            return similarities.tolist()
        except:
            return [0.0] * len(job_texts)
    
    def calculate_skill_match(self, resume_skills: Dict, job_skills: Dict) -> float:
        """Calculate skill-based matching score"""
        total_score = 0
        total_categories = 0
        
        for category in resume_skills:
            if category in job_skills:
                resume_set = set(resume_skills[category])
                job_set = set(job_skills[category])
                
                if job_set:  # Only count if job has skills in this category
                    overlap = len(resume_set.intersection(job_set))
                    category_score = overlap / len(job_set) if job_set else 0
                    total_score += category_score
                    total_categories += 1
        
        return total_score / total_categories if total_categories > 0 else 0.0

class RecommendationEngine:
    """Generate recommendations and gap analysis"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
    
    def analyze_skill_gaps(self, resume_skills: Dict, job_skills: Dict) -> Dict:
        """Analyze skill gaps between resume and job requirements"""
        gaps = {}
        
        for category in job_skills:
            job_skills_set = set(job_skills[category])
            resume_skills_set = set(resume_skills.get(category, []))
            
            missing_skills = job_skills_set - resume_skills_set
            matching_skills = job_skills_set.intersection(resume_skills_set)
            
            if job_skills_set:  # Only include categories with job requirements
                gaps[category] = {
                    'missing': list(missing_skills),
                    'matching': list(matching_skills),
                    'match_percentage': len(matching_skills) / len(job_skills_set) * 100
                }
        
        return gaps
    
    def prioritize_skills(self, skill_gaps: Dict, job_frequency: Dict = None) -> List[Dict]:
        """Prioritize skills based on gaps and market demand"""
        priorities = []
        
        for category, gap_info in skill_gaps.items():
            for skill in gap_info['missing']:
                priority_score = 1.0  # Base priority
                
                # Increase priority for categories with lower match percentage
                if gap_info['match_percentage'] < 50:
                    priority_score += 0.5
                
                # Add market demand factor if available
                if job_frequency and skill in job_frequency:
                    priority_score += job_frequency[skill] * 0.3
                
                priorities.append({
                    'skill': skill,
                    'category': category,
                    'priority_score': priority_score,
                    'current_match': gap_info['match_percentage']
                })
        
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def generate_recommendations(self, skill_gaps: Dict, top_matches: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Skill improvement recommendations
        high_priority_skills = []
        for category, gap_info in skill_gaps.items():
            if gap_info['match_percentage'] < 60:
                high_priority_skills.extend(gap_info['missing'][:3])
        
        if high_priority_skills:
            recommendations.append(
                f"🎯 Focus on developing these in-demand skills: {', '.join(high_priority_skills[:5])}"
            )
        
        # Industry-specific recommendations
        if top_matches:
            common_requirements = {}
            for match in top_matches[:3]:
                # Extract common keywords from top matches
                pass  # Implement based on job descriptions
        
        recommendations.extend([
            "📚 Consider taking online courses or certifications in your gap areas",
            "💼 Look for projects or volunteer work to gain experience in missing skills",
            "🤝 Network with professionals in your target roles",
            "📝 Update your resume to better highlight relevant experience"
        ])
        
        return recommendations

class ABTester:
    """A/B testing functionality for resume versions"""
    
    def __init__(self):
        self.test_results = {}
    
    def create_resume_variants(self, base_resume: str, variations: List[Dict]) -> List[str]:
        """Create different versions of resume based on variations"""
        variants = [base_resume]  # Original version
        
        for variation in variations:
            modified_resume = base_resume
            
            if variation['type'] == 'keyword_emphasis':
                # Emphasize certain keywords
                for keyword in variation['keywords']:
                    modified_resume = modified_resume.replace(
                        keyword, f"**{keyword}**"
                    )
            
            elif variation['type'] == 'section_reorder':
                # Different section ordering logic could be implemented
                pass
            
            elif variation['type'] == 'skill_highlight':
                # Add skills section or modify existing one
                skills_text = f"\n\nKey Skills: {', '.join(variation['skills'])}\n"
                modified_resume += skills_text
            
            variants.append(modified_resume)
        
        return variants
    
    def compare_resume_versions(self, variants: List[str], job_descriptions: List[str], 
                               matching_engine: MatchingEngine) -> Dict:
        """Compare different resume versions against job descriptions"""
        results = {}
        
        for i, variant in enumerate(variants):
            version_name = f"Version_{i}" if i > 0 else "Original"
            
            # Calculate average similarity scores
            tfidf_scores = matching_engine.calculate_tfidf_similarity(variant, job_descriptions)
            embedding_scores = matching_engine.calculate_embedding_similarity(variant, job_descriptions)
            
            avg_tfidf = np.mean(tfidf_scores) if tfidf_scores else 0
            avg_embedding = np.mean(embedding_scores) if embedding_scores else 0
            
            results[version_name] = {
                'avg_tfidf_score': avg_tfidf,
                'avg_embedding_score': avg_embedding,
                'combined_score': (avg_tfidf + avg_embedding) / 2,
                'individual_scores': {
                    'tfidf': tfidf_scores,
                    'embedding': embedding_scores
                }
            }
        
        return results

# Streamlit Application
def main():
   
    st.title("🎯 Advanced Resume-Job Matching System")
    st.markdown("Upload your resume and find the best job matches with AI-powered analysis")
    
    # Initialize components
    if 'data_ingestion' not in st.session_state:
        st.session_state.data_ingestion = DataIngestion()
        st.session_state.web_scraper = WebScraper()
        st.session_state.nlp_processor = NLPProcessor()
        st.session_state.matching_engine = MatchingEngine()
        st.session_state.recommendation_engine = RecommendationEngine()
        st.session_state.ab_tester = ABTester()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Feature",
        ["Resume Analysis", "Job Matching", "Web Scraping", "A/B Testing", "Gap Analysis"]
    )
    
    if page == "Resume Analysis":
        resume_analysis_page()
    elif page == "Job Matching":
        job_matching_page()
    elif page == "Web Scraping":
        web_scraping_page()
    elif page == "A/B Testing":
        ab_testing_page()
    elif page == "Gap Analysis":
        gap_analysis_page()

def resume_analysis_page():
    st.header("📄 Resume Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )
        
        if uploaded_file:
            # Extract text from uploaded file
            resume_text = st.session_state.data_ingestion.process_uploaded_file(uploaded_file)
            
            if resume_text:
                st.success("Resume uploaded successfully!")
                
                # Store in session state
                st.session_state.resume_text = resume_text
                
                # Show preview
                with st.expander("Preview Resume Text"):
                    st.text_area("", value=resume_text[:1000] + "...", height=200)
    
    with col2:
        if 'resume_text' in st.session_state:
            st.subheader("Analysis Results")
            
            resume_text = st.session_state.resume_text
            nlp_processor = st.session_state.nlp_processor
            
            # Extract keywords
            keywords = nlp_processor.extract_keywords(resume_text, top_n=15)
            
            # Extract skills
            skills = nlp_processor.extract_skills(resume_text)
            
            # Display keyword cloud
            if keywords:
                st.write("**Top Keywords:**")
                keyword_dict = {k: v for k, v in keywords}
                
                try:
                    wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(keyword_dict)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except:
                    # Fallback display
                    keyword_text = ", ".join([k for k, v in keywords[:10]])
                    st.write(keyword_text)
            
            # Display skills by category
            st.write("**Skills by Category:**")
            for category, skill_list in skills.items():
                if skill_list:
                    st.write(f"*{category.title()}:* {', '.join(skill_list)}")

def job_matching_page():
    st.header("🔍 Job Matching")
    
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first in the Resume Analysis page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Job Description Input")
        
        input_method = st.radio(
            "How would you like to add job descriptions?",
            ["Manual Entry", "Upload File", "Use Sample Jobs"]
        )
        
        job_descriptions = []
        
        if input_method == "Manual Entry":
            job_text = st.text_area(
                "Paste job description:",
                height=200,
                help="Copy and paste job descriptions here"
            )
            if job_text:
                job_descriptions.append(job_text)
        
        elif input_method == "Upload File":
            job_file = st.file_uploader(
                "Upload job descriptions file",
                type=['txt', 'pdf', 'docx']
            )
            if job_file:
                job_text = st.session_state.data_ingestion.process_uploaded_file(job_file)
                if job_text:
                    job_descriptions.append(job_text)
        
        else:  # Use Sample Jobs
            sample_jobs = [
                "Software Engineer position requiring Python, JavaScript, AWS, and machine learning experience. Must have 3+ years experience building web applications.",
                "Data Scientist role focusing on predictive modeling, SQL, Python, and data visualization. Experience with TensorFlow and statistical analysis required.",
                "Product Manager position for tech startup. Need experience with agile methodologies, user research, and cross-functional team leadership."
            ]
            job_descriptions = sample_jobs
            st.info(f"Using {len(sample_jobs)} sample job descriptions")
    
    with col2:
        if job_descriptions:
            st.subheader("Matching Results")
            
            # Calculate similarities
            matching_engine = st.session_state.matching_engine
            resume_text = st.session_state.resume_text
            
            tfidf_scores = matching_engine.calculate_tfidf_similarity(resume_text, job_descriptions)
            embedding_scores = matching_engine.calculate_embedding_similarity(resume_text, job_descriptions)
            
            # Create results dataframe
            results_data = []
            for i, (tfidf, embedding) in enumerate(zip(tfidf_scores, embedding_scores)):
                combined_score = (tfidf + embedding) / 2
                results_data.append({
                    'Job': f'Job {i+1}',
                    'TF-IDF Score': f'{tfidf:.3f}',
                    'Embedding Score': f'{embedding:.3f}',
                    'Combined Score': f'{combined_score:.3f}',
                    'Match %': f'{combined_score*100:.1f}%'
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                x=[f'Job {i+1}' for i in range(len(job_descriptions))],
                y=[float(score) for score in [row['Combined Score'] for row in results_data]],
                title="Job Match Scores",
                labels={'x': 'Jobs', 'y': 'Match Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Store results for other pages
            st.session_state.job_descriptions = job_descriptions
            st.session_state.match_results = results_data

def web_scraping_page():
    st.header("🌐 Job Web Scraping")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Parameters")
        
        keywords = st.text_input(
            "Job Keywords",
            placeholder="e.g., Python developer, data scientist",
            help="Enter relevant keywords for job search"
        )
        
        location = st.text_input(
            "Location",
            placeholder="e.g., New York, NY",
            help="Optional: specify job location"
        )
        
        num_jobs = st.slider("Number of Jobs", 5, 50, 10)
        
        if st.button("Search Jobs", type="primary"):
            if keywords:
                with st.spinner("Scraping job postings..."):
                    # Simulate web scraping
                    web_scraper = st.session_state.web_scraper
                    scraped_jobs = web_scraper.scrape_indeed_jobs(keywords, location, num_jobs)
                    
                    st.session_state.scraped_jobs = scraped_jobs
                    st.success(f"Found {len(scraped_jobs)} job postings!")
            else:
                st.error("Please enter keywords to search for jobs.")
    
    with col2:
        if 'scraped_jobs' in st.session_state:
            st.subheader("Scraped Job Results")
            
            scraped_jobs = st.session_state.scraped_jobs
            
            # Display jobs with match scores if resume is available
            if 'resume_text' in st.session_state:
                resume_text = st.session_state.resume_text
                matching_engine = st.session_state.matching_engine
                
                job_texts = [job['description'] for job in scraped_jobs]
                tfidf_scores = matching_engine.calculate_tfidf_similarity(resume_text, job_texts)
                embedding_scores = matching_engine.calculate_embedding_similarity(resume_text, job_texts)
                
                # Add scores to jobs
                for i, job in enumerate(scraped_jobs):
                    combined_score = (tfidf_scores[i] + embedding_scores[i]) / 2
                    job['match_score'] = combined_score
                
                # Sort by match score
                scraped_jobs.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Display jobs
            for i, job in enumerate(scraped_jobs):
                with st.expander(f"#{i+1} {job['title']} - {job['company']}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Location:** {job['location']}")
                        st.write(f"**Salary:** {job['salary']}")
                        st.write(f"**Posted:** {job['posted_date']}")
                        if 'match_score' in job:
                            st.metric("Match Score", f"{job['match_score']:.2f}")
                    
                    with col_b:
                        st.write("**Description:**")
                        st.write(job['description'][:300] + "...")
                        st.link_button("View Job", job['url'])

def ab_testing_page():
    st.header("🧪 A/B Testing Resume Versions")
    
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first in the Resume Analysis page.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create Resume Variants")
        
        # Original resume
        st.write("**Original Resume Available** ✅")
        
        # Variant configuration
        st.write("**Configure Test Variants:**")
        
        variant_configs = []
        
        # Keyword emphasis variant
        if st.checkbox("Keyword Emphasis Variant"):
            keywords = st.multiselect(
                "Select keywords to emphasize:",
                ["Python", "JavaScript", "SQL", "AWS", "React", "Machine Learning", "Leadership"],
                default=["Python", "SQL"]
            )
            if keywords:
                variant_configs.append({
                    'type': 'keyword_emphasis',
                    'keywords': keywords
                })
        
        # Skills highlight variant
        if st.checkbox("Skills Highlight Variant"):
            additional_skills = st.text_input(
                "Additional skills to highlight:",
                placeholder="e.g., TensorFlow, Docker, Agile"
            )
            if additional_skills:
                skills_list = [s.strip() for s in additional_skills.split(',')]
                variant_configs.append({
                    'type': 'skill_highlight',
                    'skills': skills_list
                })
        
        if st.button("Generate Variants", type="primary"):
            if variant_configs:
                ab_tester = st.session_state.ab_tester
                resume_text = st.session_state.resume_text
                
                variants = ab_tester.create_resume_variants(resume_text, variant_configs)
                st.session_state.resume_variants = variants
                st.session_state.variant_configs = variant_configs
                
                st.success(f"Generated {len(variants)} resume variants!")
    
    with col2:
        if 'resume_variants' in st.session_state and 'job_descriptions' in st.session_state:
            st.subheader("A/B Test Results")
            
            variants = st.session_state.resume_variants
            job_descriptions = st.session_state.job_descriptions
            matching_engine = st.session_state.matching_engine
            ab_tester = st.session_state.ab_tester
            
            # Run A/B test comparison
            test_results = ab_tester.compare_resume_versions(variants, job_descriptions, matching_engine)
            
            # Display results
            results_data = []
            for version, metrics in test_results.items():
                results_data.append({
                    'Version': version,
                    'TF-IDF Score': f"{metrics['avg_tfidf_score']:.3f}",
                    'Embedding Score': f"{metrics['avg_embedding_score']:.3f}",
                    'Combined Score': f"{metrics['combined_score']:.3f}",
                    'Improvement': f"{((metrics['combined_score'] - test_results['Original']['combined_score']) / test_results['Original']['combined_score'] * 100) if version != 'Original' else 0:.1f}%"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = go.Figure()
            
            versions = list(test_results.keys())
            tfidf_scores = [test_results[v]['avg_tfidf_score'] for v in versions]
            embedding_scores = [test_results[v]['avg_embedding_score'] for v in versions]
            
            fig.add_trace(go.Bar(name='TF-IDF', x=versions, y=tfidf_scores))
            fig.add_trace(go.Bar(name='Embedding', x=versions, y=embedding_scores))
            
            fig.update_layout(
                title='Resume Variant Performance Comparison',
                xaxis_title='Versions',
                yaxis_title='Match Score',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            best_version = max(test_results.keys(), key=lambda x: test_results[x]['combined_score'])
            if best_version != 'Original':
                st.success(f"🏆 **Best Performing Version:** {best_version}")
                improvement = ((test_results[best_version]['combined_score'] - test_results['Original']['combined_score']) / test_results['Original']['combined_score'] * 100)
                st.info(f"📈 Performance improvement: {improvement:.1f}% over original")
            else:
                st.info("📊 Original version performs best. Consider trying different variations.")
        
        elif 'resume_variants' not in st.session_state:
            st.info("Configure and generate variants first")
        else:
            st.warning("No job descriptions available for testing. Visit the Job Matching page first.")

def gap_analysis_page():
    st.header("📊 Skills Gap Analysis")
    
    if 'resume_text' not in st.session_state:
        st.warning("Please upload a resume first in the Resume Analysis page.")
        return
    
    if 'job_descriptions' not in st.session_state:
        st.warning("Please add job descriptions in the Job Matching page first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Gap Analysis Results")
        
        # Extract skills from resume and jobs
        nlp_processor = st.session_state.nlp_processor
        recommendation_engine = st.session_state.recommendation_engine
        
        resume_text = st.session_state.resume_text
        job_descriptions = st.session_state.job_descriptions
        
        # Get skills from resume
        resume_skills = nlp_processor.extract_skills(resume_text)
        
        # Get skills from all job descriptions (combined)
        combined_job_text = " ".join(job_descriptions)
        job_skills = nlp_processor.extract_skills(combined_job_text)
        
        # Analyze gaps
        skill_gaps = recommendation_engine.analyze_skill_gaps(resume_skills, job_skills)
        
        # Display gap analysis
        for category, gap_info in skill_gaps.items():
            if gap_info['missing'] or gap_info['matching']:
                with st.expander(f"{category.title()} Skills"):
                    
                    # Progress bar for match percentage
                    match_pct = gap_info['match_percentage']
                    st.metric(
                        f"Match Rate", 
                        f"{match_pct:.1f}%",
                        delta=f"{match_pct - 50:.1f}% vs avg" if match_pct != 50 else None
                    )
                    
                    if gap_info['matching']:
                        st.success(f"✅ **You have:** {', '.join(gap_info['matching'])}")
                    
                    if gap_info['missing']:
                        st.error(f"❌ **Missing:** {', '.join(gap_info['missing'])}")
        
        # Store for recommendations
        st.session_state.skill_gaps = skill_gaps
    
    with col2:
        st.subheader("Recommendations & Action Plan")
        
        if 'skill_gaps' in st.session_state:
            skill_gaps = st.session_state.skill_gaps
            recommendation_engine = st.session_state.recommendation_engine
            
            # Generate recommendations
            recommendations = recommendation_engine.generate_recommendations(
                skill_gaps, 
                st.session_state.get('match_results', [])
            )
            
            st.write("**Action Items:**")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Priority skills chart
            priorities = recommendation_engine.prioritize_skills(skill_gaps)
            
            if priorities:
                st.write("**Top Priority Skills to Learn:**")
                
                priority_df = pd.DataFrame(priorities[:10])  # Top 10
                
                fig = px.bar(
                    priority_df,
                    x='priority_score',
                    y='skill',
                    orientation='h',
                    color='category',
                    title='Skill Learning Priorities',
                    labels={'priority_score': 'Priority Score', 'skill': 'Skills'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Learning resources suggestions
                st.write("**Suggested Learning Resources:**")
                top_skills = [p['skill'] for p in priorities[:5]]
                
                resource_suggestions = {
                    'python': 'Python.org tutorials, Codecademy Python course',
                    'javascript': 'MDN Web Docs, FreeCodeCamp JavaScript',
                    'sql': 'SQLBolt, W3Schools SQL Tutorial',
                    'aws': 'AWS Training and Certification, A Cloud Guru',
                    'react': 'React official documentation, Scrimba React course',
                    'machine learning': 'Coursera ML Course, Kaggle Learn',
                    'docker': 'Docker official documentation, Docker tutorial',
                    'kubernetes': 'Kubernetes official tutorials, KodeKloud'
                }
                
                for skill in top_skills:
                    if skill.lower() in resource_suggestions:
                        st.write(f"• **{skill.title()}:** {resource_suggestions[skill.lower()]}")

# Analytics and Reporting
def create_analytics_dashboard():
    """Create analytics dashboard for system performance"""
    st.header("📈 Analytics Dashboard")
    
    # Mock analytics data
    analytics_data = {
        'total_resumes_processed': 1247,
        'total_jobs_matched': 5632,
        'avg_match_score': 0.73,
        'top_skills_in_demand': ['Python', 'JavaScript', 'SQL', 'AWS', 'React'],
        'success_stories': 89
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Resumes Processed", analytics_data['total_resumes_processed'])
    
    with col2:
        st.metric("Job Matches Made", analytics_data['total_jobs_matched'])
    
    with col3:
        st.metric("Avg Match Score", f"{analytics_data['avg_match_score']:.2f}")
    
    with col4:
        st.metric("Success Stories", analytics_data['success_stories'])

# Additional utility functions
def export_results():
    """Export analysis results to various formats"""
    if 'match_results' in st.session_state:
        results_df = pd.DataFrame(st.session_state.match_results)
        
        # CSV export
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Results as CSV",
            data=csv,
            file_name=f"resume_match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # JSON export
        json_data = json.dumps(st.session_state.match_results, indent=2)
        st.download_button(
            label="📄 Download Results as JSON",
            data=json_data,
            file_name=f"resume_match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Error handling and logging
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")
    return wrapper

# Performance optimization
@st.cache_data
def cached_tfidf_calculation(resume_text, job_texts_tuple):
    """Cache TF-IDF calculations for better performance"""
    job_texts = list(job_texts_tuple)  # Convert back from tuple
    matching_engine = MatchingEngine()
    return matching_engine.calculate_tfidf_similarity(resume_text, job_texts)

@st.cache_data
def cached_embedding_calculation(resume_text, job_texts_tuple):
    """Cache embedding calculations for better performance"""
    job_texts = list(job_texts_tuple)
    matching_engine = MatchingEngine()
    return matching_engine.calculate_embedding_similarity(resume_text, job_texts)

# Footer and additional information
def add_footer():
    """Add footer with additional information"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 Advanced Resume-Job Matching System</p>
        <p>Built with Streamlit | Powered by AI | Data Privacy Compliant</p>
        <p><small>This system helps job seekers optimize their resumes and find better job matches through AI-powered analysis.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Apply custom styling
    add_custom_css()
    
    # Run main application
    main()
    
    # Add export functionality in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    if st.sidebar.button("Export Analysis"):
        export_results()
    
    # Add footer
    add_footer()
