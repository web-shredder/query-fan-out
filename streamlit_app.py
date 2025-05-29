import streamlit as st
import json
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import os
from io import BytesIO, StringIO
import zipfile

# Import provider libraries conditionally
GEMINI_AVAILABLE = False
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
REPORTLAB_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    pass

# Query type definitions with enhanced metadata
QUERY_TYPES = {
    "reformulation": {
        "name": "🔄 Reformulations",
        "description": "Alternative ways to phrase the query",
        "icon": "🔄",
        "color": "#FF6B6B"
    },
    "related": {
        "name": "🔗 Related Queries", 
        "description": "Related topics and adjacent searches",
        "icon": "🔗",
        "color": "#4ECDC4"
    },
    "implicit": {
        "name": "💭 Implicit Queries",
        "description": "Hidden questions within the original query",
        "icon": "💭", 
        "color": "#45B7D1"
    },
    "comparative": {
        "name": "⚖️ Comparative",
        "description": "Comparison and versus queries",
        "icon": "⚖️",
        "color": "#96CEB4"
    },
    "entity_focus": {
        "name": "🎯 Entity Focus",
        "description": "Queries focused on specific entities",
        "icon": "🎯",
        "color": "#FECA57"
    },
    "personalized": {
        "name": "👤 Personalized",
        "description": "User-specific variations",
        "icon": "👤",
        "color": "#FF9FF3"
    },
    "technical": {
        "name": "🔧 Technical",
        "description": "Technical or specialized queries",
        "icon": "🔧",
        "color": "#54A0FF"
    },
    "location": {
        "name": "📍 Location-based",
        "description": "Geographic or location-specific queries",
        "icon": "📍",
        "color": "#48DBFB"
    },
    "temporal": {
        "name": "🕒 Temporal",
        "description": "Time-based or seasonal queries",
        "icon": "🕒",
        "color": "#FF6B9D"
    },
    "transactional": {
        "name": "💳 Transactional",
        "description": "Purchase or action-oriented queries",
        "icon": "💳",
        "color": "#C44569"
    },
    "navigational": {
        "name": "🧭 Navigational",
        "description": "Brand or website-specific queries",
        "icon": "🧭",
        "color": "#F8B500"
    },
    "user_intent": {
        "name": "🎯 Intent",
        "description": "Specific user intent variations",
        "icon": "🎯",
        "color": "#6C5CE7"
    }
}

# Model options with categories
MODEL_OPTIONS = {
    "Google Gemini": [
        {"name": "Gemini 2.0 Flash (Latest)", "id": "gemini-2.0-flash-exp", "category": "Latest", "description": "Newest, fastest Gemini model"},
        {"name": "Gemini 1.5 Flash (8B)", "id": "gemini-1.5-flash-8b-latest", "category": "Fast", "description": "Lightweight, fast responses"},
        {"name": "Gemini 1.5 Flash", "id": "gemini-1.5-flash-latest", "category": "Fast", "description": "Fast, efficient model"},
        {"name": "Gemini 1.5 Pro", "id": "gemini-1.5-pro-latest", "category": "Advanced", "description": "Most capable Gemini model"},
        {"name": "Gemini 1.0 Pro", "id": "gemini-1.0-pro-latest", "category": "Legacy", "description": "Stable, proven model"}
    ],
    "OpenAI": [
        {"name": "GPT-4 Turbo", "id": "gpt-4-turbo-preview", "category": "Advanced", "description": "Most capable GPT-4"},
        {"name": "GPT-4", "id": "gpt-4", "category": "Advanced", "description": "Advanced reasoning"},
        {"name": "GPT-4o", "id": "gpt-4o", "category": "Optimized", "description": "Optimized GPT-4"},
        {"name": "GPT-4o mini", "id": "gpt-4o-mini", "category": "Fast", "description": "Fast, affordable GPT-4"},
        {"name": "GPT-3.5 Turbo", "id": "gpt-3.5-turbo", "category": "Fast", "description": "Fast, cost-effective"},
        {"name": "o1", "id": "o1", "category": "Reasoning", "description": "Complex reasoning"},
        {"name": "o1-mini", "id": "o1-mini", "category": "Reasoning", "description": "Efficient reasoning"},
        {"name": "o3-mini", "id": "o3-mini", "category": "Latest", "description": "Newest compact model"}
    ],
    "Anthropic Claude": [
        {"name": "Claude 3.5 Sonnet", "id": "claude-3-5-sonnet-latest", "category": "Latest", "description": "Most intelligent Claude"},
        {"name": "Claude 3.5 Haiku", "id": "claude-3-5-haiku-latest", "category": "Fast", "description": "Fast and efficient"},
        {"name": "Claude 3 Opus", "id": "claude-3-opus-latest", "category": "Advanced", "description": "Powerful reasoning"},
        {"name": "Claude 3 Sonnet", "id": "claude-3-sonnet-20240229", "category": "Balanced", "description": "Balanced performance"},
        {"name": "Claude 3 Haiku", "id": "claude-3-haiku-20240307", "category": "Fast", "description": "Instant responses"}
    ]
}

# Header
st.set_page_config(
    page_title="heLLiuM - LLM Query Fan Out Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔮 heLLiuM</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">Multiplied intelligence. LLM Query Fan Out Simulator.</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">Transform one search into 50+ intelligent variations using Google's query expansion methodology</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'content_analysis' not in st.session_state:
    st.session_state.content_analysis = None

# Initialize personalization data
personalization_data = {}

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "AI Provider",
        ["Google Gemini", "OpenAI", "Anthropic Claude"]
    )
    
    # API Key input
    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
        if not api_key and 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ['GEMINI_API_KEY']
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        if not api_key and 'OPENAI_API_KEY' in os.environ:
            api_key = os.environ['OPENAI_API_KEY']
    else:  # Anthropic Claude
        api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        if not api_key and 'ANTHROPIC_API_KEY' in os.environ:
            api_key = os.environ['ANTHROPIC_API_KEY']
    
    # Model selection based on provider
    model_options = MODEL_OPTIONS.get(provider, [])
    if model_options:
        # Separate model selector with categories
        st.markdown("#### Model Selection")
        
        # Category filter
        all_categories = list(set(m['category'] for m in model_options))
        selected_category = st.selectbox("Category", ["All"] + all_categories)
        
        # Filter models by category
        if selected_category == "All":
            filtered_models = model_options
        else:
            filtered_models = [m for m in model_options if m['category'] == selected_category]
        
        # Model selector
        model_names = [m['name'] for m in filtered_models]
        selected_model_name = st.selectbox("Model", model_names)
        
        # Get the selected model ID
        selected_model = next(m for m in filtered_models if m['name'] == selected_model_name)
        model = selected_model['id']
        
        # Show model description
        st.caption(f"*{selected_model['description']}*")
    else:
        model = st.text_input("Model Name")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)
    
    # Personalization settings
    st.markdown("### 🎯 Personalization (Optional)")
    with st.expander("👤 User Context", expanded=False):
        st.markdown("*Add context to generate more targeted queries*")
        
        # Demographics
        st.markdown("**Demographics**")
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Age/Age Range", placeholder="e.g., 25-34", key="age_input")
            gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Non-binary", "Other"], key="gender_input")
            location = st.text_input("Location", placeholder="e.g., New York, USA", key="location_input")
        
        with col2:
            income_range = st.text_input("Income Range", placeholder="e.g., $50k-$75k", key="income_input")
            occupation = st.text_input("Job/Industry", placeholder="e.g., Marketing Manager", key="occupation_input")
            education = st.selectbox("Education", ["Not specified", "High School", "Bachelor's", "Master's", "PhD", "Trade School"], key="education_input")
        
        # Interests & Preferences
        st.markdown("**Interests & Preferences**")
        interests = st.text_area("Interests/Hobbies", placeholder="e.g., hiking, cooking, tech gadgets", height=60, key="interests_input")
        favorite_brands = st.text_input("Favorite Brands", placeholder="e.g., Apple, Nike, Tesla", key="brands_input")
        
        # Behavioral Context
        st.markdown("**Behavioral Context**")
        col3, col4 = st.columns(2)
        with col3:
            budget = st.text_input("Budget Range", placeholder="e.g., $100-$500", key="budget_input")
            buying_stage = st.selectbox("Buying Stage", ["Not specified", "Research", "Comparison", "Ready to buy", "Just browsing"], key="buying_stage_input")
        
        with col4:
            device_type = st.selectbox("Primary Device", ["Not specified", "Mobile", "Desktop", "Tablet", "Smart TV"], key="device_input")
            time_of_day = st.selectbox("Typical Search Time", ["Not specified", "Morning", "Afternoon", "Evening", "Late night"], key="time_input")
        
        # Custom Context
        st.markdown("**Additional Context**")
        custom_context = st.text_area("Other Context", placeholder="Any other relevant information about the user or use case", height=60, key="custom_input")
        
        # Compile personalization data
        personalization_data = {
            'age': age if age else None,
            'gender': gender if gender != "Not specified" else None,
            'location': location if location else None,
            'income': income_range if income_range else None,
            'occupation': occupation if occupation else None,
            'education': education if education != "Not specified" else None,
            'interests': interests if interests else None,
            'brands': favorite_brands if favorite_brands else None,
            'budget': budget if budget else None,
            'buying_stage': buying_stage if buying_stage != "Not specified" else None,
            'device': device_type if device_type != "Not specified" else None,
            'time': time_of_day if time_of_day != "Not specified" else None,
            'custom': custom_context if custom_context else None
        }
        
        # Remove None values
        personalization_data = {k: v for k, v in personalization_data.items() if v is not None}

# Title (after sidebar)
st.title("🚀 heLLiuM: Multi-Provider AI Query Intelligence System")
st.caption("Powered by Gemini, OpenAI, and Claude with Advanced Analytics")

# Function definitions
def call_ai_api(prompt, api_key, model, temperature, max_tokens, provider):
    """Unified function to call different AI APIs"""
    try:
        if provider == "Google Gemini" and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            return response.text
            
        elif provider == "OpenAI" and OPENAI_AVAILABLE:
            # Use the newer OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an advanced search query expansion system based on Google's 'Generating Diverse Queries for Search Autocomplete' and 'Search with Stateful Chat' patents. You specialize in query fan-out methodology, generating reformulations, related queries, implicit queries, comparative queries, entity-focused variations, and personalized contextual queries. Your goal is to expand a single query into a comprehensive set of search variations that capture different user intents, contexts, and information needs - similar to how modern search engines internally process queries to better understand and serve user intent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
            
        elif provider == "Anthropic Claude" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
            
        else:
            st.error(f"Provider {provider} is not available. Please install the required library.")
            return None
            
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def generate_queries(user_query, api_key, model="gemini-1.5-flash", temperature=0.7, 
                    max_tokens=2000, num_queries=25, search_mode="ai_overview", 
                    include_types=None, provider="Google Gemini", personalization=None):
    """Generate search query variations using selected AI provider with personalization"""
    
    # Build personalization context
    personalization_context = ""
    if personalization:
        contexts = []
        if personalization.get('age'):
            contexts.append(f"Age: {personalization['age']}")
        if personalization.get('gender'):
            contexts.append(f"Gender: {personalization['gender']}")
        if personalization.get('location'):
            contexts.append(f"Location: {personalization['location']}")
        if personalization.get('income'):
            contexts.append(f"Income range: {personalization['income']}")
        if personalization.get('occupation'):
            contexts.append(f"Occupation: {personalization['occupation']}")
        if personalization.get('education'):
            contexts.append(f"Education: {personalization['education']}")
        if personalization.get('interests'):
            contexts.append(f"Interests: {personalization['interests']}")
        if personalization.get('brands'):
            contexts.append(f"Favorite brands: {personalization['brands']}")
        if personalization.get('budget'):
            contexts.append(f"Budget: {personalization['budget']}")
        if personalization.get('buying_stage'):
            contexts.append(f"Buying stage: {personalization['buying_stage']}")
        if personalization.get('device'):
            contexts.append(f"Device: {personalization['device']}")
        if personalization.get('time'):
            contexts.append(f"Search time: {personalization['time']}")
        if personalization.get('custom'):
            contexts.append(f"Additional context: {personalization['custom']}")
        
        if contexts:
            personalization_context = f"\n\nUser Context for Personalization:\n{chr(10).join(contexts)}\n\nUse this context to generate personalized query variations that would be relevant to this specific user profile."
    
    search_mode_prompts = {
        "ai_overview": f"Generate {num_queries} search query variations for AI Overview optimization. Focus on question-based queries, comparisons, and comprehensive informational searches.",
        "ai_mode_complex": f"Generate {num_queries} complex search query variations that would trigger AI-powered responses. Include multi-faceted questions, analysis requests, and queries requiring synthesis.",
        "research_mode": f"Generate {num_queries} research-focused query variations. Include academic angles, data requests, study-related queries, and expert-seeking searches.",
        "comparative": f"Generate {num_queries} comparative query variations. Focus on 'vs', 'compared to', 'difference between', 'better than', and alternative-seeking queries.",
        "multi_perspective": f"Generate {num_queries} query variations from different perspectives: beginner vs expert, buyer vs researcher, local vs global, personal vs professional."
    }
    
    mode_instruction = search_mode_prompts.get(search_mode, search_mode_prompts["ai_overview"])
    
    # Type restrictions
    type_instruction = ""
    if include_types:
        type_instruction = f"\nFocus primarily on these query types: {', '.join(include_types)}"
    
    prompt = f"""Based on the original search query: "{user_query}"

{mode_instruction}{type_instruction}{personalization_context}

Generate diverse query variations following Google's query expansion methodology. For each query, assign:
- type: one of [reformulation, related, implicit, comparative, entity_focus, personalized, technical, location, temporal, transactional, navigational, user_intent]
- confidence: float between 0 and 1 indicating likelihood of user interest
- priority: "high", "medium", or "low" based on relevance

Return ONLY a valid JSON array with no other text:
[
  {{
    "id": "q1",
    "query": "example query here",
    "type": "reformulation",
    "confidence": 0.95,
    "priority": "high"
  }}
]
"""
    
    response_text = call_ai_api(prompt, api_key, model, temperature, max_tokens, provider)
    
    if not response_text:
        return []
    
    try:
        # Clean the response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        queries = json.loads(response_text.strip())
        
        # Ensure all queries have required fields
        for i, query in enumerate(queries):
            if 'id' not in query:
                query['id'] = f"q{i+1}"
            if 'confidence' not in query:
                query['confidence'] = 0.8
            if 'priority' not in query:
                query['priority'] = 'medium'
            if 'type' not in query:
                query['type'] = 'related'
        
        return queries
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response: {e}")
        st.text("Raw response:")
        st.code(response_text)
        return []

def analyze_content(content, queries):
    """Analyze content for query matches with proper stop word filtering"""
    import re
    from collections import Counter
    
    # Comprehensive stop words list
    STOP_WORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
        'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
        'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 
        'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 
        'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 
        'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', 
        "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', 
        "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 
        'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 
        'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 
        'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', 
        "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
        'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 
        'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", 
        "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 
        'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 
        "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 
        'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', 
        "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 
        'yourself', 'yourselves', 'can', 'will', 'just', 'should', 'us'
    }
    
    def extract_keywords(text):
        """Extract meaningful keywords from text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = []
        for word in words:
            if word not in STOP_WORDS and len(word) > 2:
                keywords.append(word)
        
        # Also extract:
        # 1. Acronyms (all caps)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        keywords.extend([a.lower() for a in acronyms])
        
        # 2. Compound terms
        compounds = re.findall(r'\b\w+(?:-\w+)+\b', text.lower())
        keywords.extend(compounds)
        
        # 3. Numbers that might be years or important
        numbers = re.findall(r'\b\d{4}\b', text)
        keywords.extend(numbers)
        
        return keywords
    
    # Extract keywords from content
    content_keywords = extract_keywords(content)
    content_keyword_set = set(content_keywords)
    
    analysis_results = []
    
    for query in queries:
        # Extract keywords from query
        query_keywords = extract_keywords(query['query'])
        query_keyword_set = set(query_keywords)
        
        # Find matching keywords (excluding stop words)
        matched_keywords = []
        for keyword in query_keyword_set:
            if keyword in content_keyword_set:
                matched_keywords.append(keyword)
        
        # Calculate match score based on meaningful keywords only
        if query_keyword_set:
            match_score = len(matched_keywords) / len(query_keyword_set)
        else:
            match_score = 0
        
        # Check for exact phrase matches (bonus points)
        exact_match_bonus = 0
        query_lower = query['query'].lower()
        content_lower = content.lower()
        
        # Check for important phrases
        important_phrases = []
        if len(query_keywords) >= 2:
            # Create 2-word phrases from keywords
            for i in range(len(query_keywords)-1):
                phrase = f"{query_keywords[i]} {query_keywords[i+1]}"
                if phrase in content_lower:
                    exact_match_bonus = 0.1
                    important_phrases.append(phrase)
        
        # Final score with bonus
        final_score = min(match_score + exact_match_bonus, 1.0)
        
        analysis_results.append({
            'query': query['query'],
            'type': query['type'],
            'match_score': final_score,
            'matched_words': matched_keywords,
            'query_keywords': list(query_keyword_set),
            'confidence': query.get('confidence', 0),
            'priority': query.get('priority', 'medium')
        })
    
    # Sort by match score
    analysis_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return analysis_results

def analyze_content_match(content, queries):
    """Analyze how well content matches the generated queries with improved keyword extraction"""
    import re
    from collections import Counter
    
    # Common words to ignore (expanded list)
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
        'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new',
        'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do',
        'first', 'any', 'my', 'now', 'way', 'over', 'even', 'most', 'me',
        'i', 'you', 'your', 'we', 'our', 'us', 'am', 'been', 'being', 'were'
    }
    
    def extract_important_words(text):
        """Extract important words from text, prioritizing nouns, verbs, and key terms"""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and very short words
        important_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Identify potential key terms (capitalized in original, compounds, technical terms)
        original_words = re.findall(r'\b[A-Za-z]+\b', text)
        capitalized = [w.lower() for w in original_words if w[0].isupper() and w.lower() not in stop_words]
        
        # Identify compound terms (connected by hyphens)
        compounds = re.findall(r'\b[a-z]+(?:-[a-z]+)+\b', text.lower())
        
        # Identify acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        acronyms_lower = [a.lower() for a in acronyms]
        
        # Combine all important terms
        all_important = important_words + capitalized + compounds + acronyms_lower
        
        # Weight certain terms higher
        weighted_words = []
        for word in all_important:
            # Give more weight to: capitalized terms, compounds, acronyms, and longer words
            if word in capitalized or word in compounds or word in acronyms_lower:
                weighted_words.extend([word] * 3)  # Triple weight
            elif len(word) > 6:
                weighted_words.extend([word] * 2)  # Double weight
            else:
                weighted_words.append(word)
        
        return weighted_words
    
    # Extract important words from content
    content_words = extract_important_words(content)
    content_word_freq = Counter(content_words)
    
    results = []
    
    for query in queries:
        # Extract important words from query
        query_words = extract_important_words(query['query'])
        query_word_set = set(query_words)
        
        # Calculate matches with important words only
        matches = []
        matched_words = []
        
        for word in query_word_set:
            if word in content_word_freq:
                matches.append(word)
                matched_words.append(word)
        
        # Calculate match score based on important words
        if query_word_set:
            match_score = len(matches) / len(query_word_set)
        else:
            match_score = 0
        
        # Boost score if high-priority query words are matched
        priority_boost = 1.0
        if query.get('priority') == 'high' and match_score > 0:
            priority_boost = 1.2
        
        final_score = min(match_score * priority_boost, 1.0)
        
        results.append({
            'query': query['query'],
            'type': query.get('type', 'unknown'),
            'match_score': final_score,
            'matched_words': matched_words,
            'query_words': list(query_word_set),
            'priority': query.get('priority', 'medium')
        })
    
    # Sort by match score
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return results

def create_hierarchical_visualization(queries, original_query, content_analysis=None):
    """Create a hierarchical visualization of queries using treemap"""
    import plotly.express as px
    
    # Build data for treemap
    data = []
    
    # Add root
    data.append({
        'names': original_query,
        'parents': '',
        'values': 1,
        'text': original_query,
        'confidence': 1.0,
        'match_score': 1.0 if content_analysis else None
    })
    
    # Group queries by type
    type_counts = {}
    for query in queries:
        qtype = query.get('type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    # Add type categories
    for qtype, count in type_counts.items():
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓"})
        data.append({
            'names': f"{type_info['icon']} {type_info['name']}",
            'parents': original_query,
            'values': count,
            'text': type_info['name'],
            'confidence': 0.9,
            'match_score': None
        })
    
    # Add individual queries
    for query in queries:
        qtype = query.get('type', 'unknown')
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓"})
        parent_name = f"{type_info['icon']} {type_info['name']}"
        
        # Get match score if content analysis exists
        match_score = None
        if content_analysis:
            for analysis in content_analysis:
                if analysis['query'] == query['query']:
                    match_score = analysis['match_score']
                    break
        
        # Create display text with confidence and match score
        display_text = query['query']
        if match_score is not None:
            display_text += f" [{match_score:.0%} match]"
        
        data.append({
            'names': display_text,
            'parents': parent_name,
            'values': 1,
            'text': query['query'],
            'confidence': query.get('confidence', 0),
            'match_score': match_score
        })
    
    # Create treemap
    df = pd.DataFrame(data)
    
    # Create custom hover text
    df['hover_text'] = df.apply(lambda row: 
        f"<b>{row['text']}</b><br>" +
        f"Confidence: {row['confidence']:.0%}<br>" +
        (f"Content Match: {row['match_score']:.0%}" if row['match_score'] is not None else ""),
        axis=1
    )
    
    # Color based on confidence or match score
    color_column = 'match_score' if content_analysis else 'confidence'
    
    fig = px.treemap(
        df,
        names='names',
        parents='parents',
        values='values',
        color=color_column,
        hover_data={'hover_text': True, 'names': False, 'parents': False, 'values': False},
        color_continuous_scale='RdYlGn',
        range_color=[0, 1]
    )
    
    fig.update_traces(
        textinfo="label",
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker=dict(cornerradius=5)
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        height=700,
        font=dict(size=12)
    )
    
    return fig

def create_sunburst_visualization(queries, original_query, content_analysis=None):
    """Create a sunburst visualization of queries"""
    import plotly.express as px
    
    # Build data for sunburst
    data = []
    
    # Group queries by type
    for query in queries:
        qtype = query.get('type', 'unknown')
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓", "color": "#999"})
        
        # Get match score if content analysis exists
        match_score = None
        if content_analysis:
            for analysis in content_analysis:
                if analysis['query'] == query['query']:
                    match_score = analysis['match_score']
                    break
        
        # Create display text
        display_text = query['query']
        if match_score is not None:
            display_text += f" ({match_score:.0%})"
        
        data.append({
            'labels': display_text,
            'parents': f"{type_info['icon']} {type_info['name']}",
            'values': 1,
            'confidence': query.get('confidence', 0),
            'match_score': match_score,
            'priority': query.get('priority', 'medium'),
            'full_query': query['query']
        })
    
    # Add type categories
    type_summaries = {}
    for query in queries:
        qtype = query.get('type', 'unknown')
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓"})
        type_name = f"{type_info['icon']} {type_info['name']}"
        
        if type_name not in type_summaries:
            type_summaries[type_name] = {
                'count': 0,
                'total_confidence': 0,
                'total_match': 0,
                'has_match': False
            }
        
        type_summaries[type_name]['count'] += 1
        type_summaries[type_name]['total_confidence'] += query.get('confidence', 0)
        
        if content_analysis:
            for analysis in content_analysis:
                if analysis['query'] == query['query']:
                    type_summaries[type_name]['total_match'] += analysis['match_score']
                    type_summaries[type_name]['has_match'] = True
                    break
    
    # Add type nodes
    for type_name, summary in type_summaries.items():
        avg_confidence = summary['total_confidence'] / summary['count']
        avg_match = summary['total_match'] / summary['count'] if summary['has_match'] else None
        
        data.append({
            'labels': type_name,
            'parents': 'Query Analysis',
            'values': summary['count'],
            'confidence': avg_confidence,
            'match_score': avg_match,
            'priority': 'high',
            'full_query': f"{summary['count']} queries"
        })
    
    # Add root
    data.append({
        'labels': 'Query Analysis',
        'parents': '',
        'values': len(queries),
        'confidence': 1.0,
        'match_score': None,
        'priority': 'high',
        'full_query': original_query
    })
    
    df = pd.DataFrame(data)
    
    # Color based on match score or confidence
    color_column = 'match_score' if content_analysis and any(d['match_score'] is not None for d in data) else 'confidence'
    
    fig = px.sunburst(
        df,
        names='labels',
        parents='parents',
        values='values',
        color=color_column,
        hover_data={'full_query': True, 'confidence': ':.0%', 'match_score': ':.0%'},
        color_continuous_scale='RdYlGn',
        range_color=[0, 1]
    )
    
    fig.update_traces(
        textinfo="label",
        hovertemplate='<b>%{label}</b><br>Query: %{customdata[0]}<br>Confidence: %{customdata[1]}<br>Match: %{customdata[2]}<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        height=700
    )
    
    return fig

def export_to_csv(queries):
    """Export queries to CSV format"""
    df = pd.DataFrame(queries)
    return df.to_csv(index=False)

def export_to_json(queries, metadata=None):
    """Export queries to JSON with metadata"""
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "total_queries": len(queries),
        "queries": queries
    }
    if metadata:
        export_data["metadata"] = metadata
    return json.dumps(export_data, indent=2)

def export_to_pdf(queries, original_query, content_analysis=None, personalization=None):
    """Export queries and analysis to PDF with personalization info"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Title
    story.append(Paragraph("heLLiuM Query Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Original Query
    story.append(Paragraph(f"<b>Original Query:</b> {original_query}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Queries:</b> {len(queries)}", styles['Normal']))
    
    # Personalization Context
    if personalization:
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Personalization Context:</b>", styles['Heading2']))
        for key, value in personalization.items():
            story.append(Paragraph(f"• <b>{key.title()}:</b> {value}", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Query Summary by Type
    story.append(Paragraph("Query Distribution by Type", styles['Heading2']))
    type_counts = Counter(q.get('type', 'unknown') for q in queries)
    
    type_data = [['Query Type', 'Count', 'Percentage']]
    for qtype, count in type_counts.most_common():
        type_info = QUERY_TYPES.get(qtype, {"name": qtype})
        percentage = f"{(count/len(queries)*100):.1f}%"
        type_data.append([type_info['name'], str(count), percentage])
    
    type_table = Table(type_data)
    type_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(type_table)
    story.append(Spacer(1, 20))
    
    # All Queries
    story.append(Paragraph("Generated Queries", styles['Heading2']))
    
    # Group queries by type
    queries_by_type = defaultdict(list)
    for query in queries:
        queries_by_type[query.get('type', 'unknown')].append(query)
    
    for qtype, type_queries in queries_by_type.items():
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓"})
        story.append(Paragraph(f"{type_info['icon']} {type_info['name']}", styles['Heading3']))
        
        for query in sorted(type_queries, key=lambda x: x.get('confidence', 0), reverse=True):
            confidence = query.get('confidence', 0)
            priority = query.get('priority', 'medium')
            
            # Color code by priority
            if priority == 'high':
                color = colors.darkgreen
            elif priority == 'low':
                color = colors.grey
            else:
                color = colors.black
            
            query_style = ParagraphStyle(
                'QueryStyle',
                parent=styles['Normal'],
                textColor=color,
                leftIndent=20
            )
            
            story.append(Paragraph(
                f"• {query['query']} (Confidence: {confidence:.0%})",
                query_style
            ))
        
        story.append(Spacer(1, 10))
    
    # Content Analysis Results
    if content_analysis:
        story.append(PageBreak())
        story.append(Paragraph("Content Analysis Results", styles['Heading2']))
        story.append(Paragraph(
            "Top 10 queries by content match score:",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))
        
        analysis_data = [['Query', 'Type', 'Match Score', 'Matched Keywords']]
        for match in content_analysis[:10]:
            type_info = QUERY_TYPES.get(match['type'], {"name": match['type']})
            analysis_data.append([
                match['query'][:50] + '...' if len(match['query']) > 50 else match['query'],
                type_info['name'],
                f"{match['match_score']:.0%}",
                ', '.join(match.get('matched_words', [])[:5])
            ])
        
        analysis_table = Table(analysis_data, colWidths=[3*inch, 1.5*inch, 1*inch, 1.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(analysis_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_pdf_report(queries, original_query, personalization=None):
    """Generate a PDF report of the queries"""
    if not REPORTLAB_AVAILABLE:
        st.error("PDF export requires reportlab. Install with: pip install reportlab")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1
    )
    elements.append(Paragraph("heLLiuM Query Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Metadata
    elements.append(Paragraph(f"<b>Original Query:</b> {original_query}", styles['Normal']))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Paragraph(f"<b>Total Queries:</b> {len(queries)}", styles['Normal']))
    
    # Add personalization context if provided
    if personalization:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Personalization Context", styles['Heading2']))
        for key, value in personalization.items():
            elements.append(Paragraph(f"• <b>{key.title()}:</b> {value}", styles['Normal']))
    
    elements.append(Spacer(1, 20))
    
    # Query type distribution
    elements.append(Paragraph("Query Type Distribution", styles['Heading2']))
    type_counts = {}
    for query in queries:
        qtype = query.get('type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    # Create table data
    table_data = [['Query Type', 'Count', 'Percentage']]
    for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        type_info = QUERY_TYPES.get(qtype, {"name": qtype})
        percentage = f"{(count/len(queries)*100):.1f}%"
        table_data.append([type_info['name'], str(count), percentage])
    
    # Create and style table
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # Queries by priority
    elements.append(Paragraph("High Priority Queries", styles['Heading2']))
    high_priority = [q for q in queries if q.get('priority') == 'high']
    for query in high_priority[:10]:  # Top 10 high priority
        elements.append(Paragraph(f"• {query['query']}", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main UI
tabs = st.tabs(["🚀 Generate", "📊 Results", "🔍 Content Analysis", "🗺️ Visualization", "📚 History", "📖 Resources"])

with tabs[0]:  # Generate Tab
    st.markdown("### 🎯 Query Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., best electric SUV for families",
            help="Enter the main search query you want to expand"
        )
    
    with col2:
        num_queries = st.number_input(
            "Number of queries:",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="How many query variations to generate"
        )
    
    # Search mode selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["ai_overview", "ai_mode_complex", "research_mode", "comparative", "multi_perspective"],
            format_func=lambda x: {
                "ai_overview": "AI Overview (Simple)",
                "ai_mode_complex": "AI Mode (Complex)",
                "research_mode": "Research Mode",
                "comparative": "Comparative Analysis",
                "multi_perspective": "Multi-Perspective"
            }[x],
            help="Choose the type of query expansion strategy"
        )
    
    with col2:
        include_confidence = st.checkbox("Show confidence scores", value=True)
        include_priority = st.checkbox("Show priority levels", value=True)
    
    with col3:
        include_mindmap = st.checkbox("Generate visualization", value=True)
        export_format = st.multiselect(
            "Export formats",
            ["CSV", "JSON", "PDF Report"],
            default=["CSV"]
        )
    
    # Query type filter
    with st.expander("🎛️ Advanced Options"):
        st.markdown("**Filter by Query Types**")
        filter_types = st.checkbox("Enable type filtering")
        if filter_types:
            selected_types = st.multiselect(
                "Select query types to include:",
                list(QUERY_TYPES.keys()),
                format_func=lambda x: QUERY_TYPES[x]['name'],
                default=list(QUERY_TYPES.keys())[:5]
            )
        else:
            selected_types = None
    
    # Generate button
    if st.button("🚀 Generate Queries", type="primary", disabled=not api_key):
        if user_query:
            with st.spinner(f"Generating {num_queries} queries using {provider}..."):
                try:
                    results = generate_queries(
                        user_query, 
                        api_key, 
                        model, 
                        temperature, 
                        max_tokens, 
                        num_queries,
                        search_mode,
                        selected_types if filter_types else None,
                        provider,
                        personalization_data
                    )
                    st.session_state.last_results = results
                    st.session_state.last_query = user_query
                    st.session_state.last_provider = provider
                    st.session_state.last_model = model
                    
                    # Update history
                    st.session_state.query_history.append({
                        'timestamp': datetime.now(),
                        'query': user_query,
                        'results': results,
                        'provider': provider,
                        'model': model,
                        'mode': search_mode,
                        'personalization': personalization_data
                    })
                    
                    st.success(f"✅ Generated {len(results)} queries using {provider}!")
                    
                    # Show personalization indicator
                    if personalization_data:
                        with st.expander("Applied Personalization Context"):
                            for key, value in personalization_data.items():
                                st.write(f"**{key.title()}**: {value}")
                    
                    # Quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Queries", len(results))
                    with col2:
                        high_conf = len([q for q in results if q.get('confidence', 0) > 0.8])
                        st.metric("High Confidence", high_conf)
                    with col3:
                        types = len(set(q.get('type', 'unknown') for q in results))
                        st.metric("Query Types", types)
                    with col4:
                        avg_conf = sum(q.get('confidence', 0) for q in results) / len(results)
                        st.metric("Avg Confidence", f"{avg_conf:.0%}")
                        
                except Exception as e:
                    st.error(f"Error generating queries: {str(e)}")
        else:
            st.warning("Please enter a search query.")

with tabs[1]:  # Results Tab
    if st.session_state.last_results:
        st.markdown(f"### 📊 Results for: '{st.session_state.last_query}'")
        st.caption(f"Provider: {st.session_state.last_provider} | Model: {st.session_state.last_model}")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "CSV" in export_format:
                csv_data = export_to_csv(st.session_state.last_results)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if "PDF Report" in export_format and REPORTLAB_AVAILABLE:
                pdf_buffer = generate_pdf_report(
                    st.session_state.last_results, 
                    user_query,
                    personalization_data
                )
                if pdf_buffer:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"qforia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        with col3:
            if "JSON" in export_format:
                json_data = export_to_json(
                    st.session_state.last_results,
                    {
                        "original_query": st.session_state.last_query,
                        "provider": st.session_state.last_provider,
                        "model": st.session_state.last_model,
                        "search_mode": search_mode
                    }
                )
                st.download_button(
                    "📋 Download JSON",
                    data=json_data,
                    file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Filter and sort options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Filter by type",
                ["All"] + [QUERY_TYPES[t]['name'] for t in QUERY_TYPES.keys()],
                key="result_filter"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Confidence", "Priority", "Type", "Query Length"],
                key="result_sort"
            )
        
        with col3:
            min_confidence = st.slider(
                "Min confidence",
                0.0, 1.0, 0.0, 0.1,
                key="confidence_filter"
            )
        
        # Filter results
        filtered_results = st.session_state.last_results
        
        if filter_type != "All":
            type_key = next(k for k, v in QUERY_TYPES.items() if v['name'] == filter_type)
            filtered_results = [q for q in filtered_results if q.get('type') == type_key]
        
        filtered_results = [q for q in filtered_results if q.get('confidence', 0) >= min_confidence]
        
        # Sort results
        if sort_by == "Confidence":
            filtered_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        elif sort_by == "Priority":
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            filtered_results.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2), reverse=True)
        elif sort_by == "Type":
            filtered_results.sort(key=lambda x: x.get('type', 'unknown'))
        elif sort_by == "Query Length":
        if sort_by == "Confidence":
            filtered_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        elif sort_by == "Priority":
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            filtered_results.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2), reverse=True)
        elif sort_by == "Type":
            filtered_results.sort(key=lambda x: x.get('type', 'unknown'))
        elif sort_by == "Query Length":
            filtered_results.sort(key=lambda x: len(x.get('query', '')), reverse=True)
        
        # Display filtered results
        st.markdown(f"### 📊 Showing {len(filtered_results)} queries")
        
        if not filtered_results:
            st.warning("No queries match your filters. Try adjusting the criteria.")
        else:
            # Display queries with enhanced formatting
            for idx, query in enumerate(filtered_results, 1):
                with st.container():
                    col1, col2, col3 = st.columns([6, 2, 2])
                    
                    with col1:
                        # Query text with priority indicator
                        priority_colors = {
                            'high': '🔴',
                            'medium': '🟡', 
                            'low': '🟢'
                        }
                        priority_icon = priority_colors.get(query.get('priority', 'medium'), '⚪')
                        
                        st.markdown(f"**{idx}. {query.get('query', 'Unknown Query')}**")
                        
                        # Show additional details if available
                        details = []
                        if 'intent' in query:
                            details.append(f"💡 {query['intent']}")
                        if 'reasoning' in query:
                            details.append(f"📝 {query['reasoning']}")
                        
                        if details:
                            st.caption(" | ".join(details))
                    
                    with col2:
                        # Type and confidence
                        type_info = QUERY_TYPES.get(query.get('type', 'unknown'), {
                            'name': query.get('type', 'Unknown'),
                            'icon': '❓'
                        })
                        st.markdown(f"{type_info['icon']} **{type_info['name']}**")
                        st.markdown(f"**{query.get('confidence', 0):.0%}** confidence")
                    
                    with col3:
                        # Priority and actions
                        priority = query.get('priority', 'medium')
                        st.markdown(f"{priority_icon} **{priority.upper()}**")
                        
                        # Query length
                        query_length = len(query.get('query', ''))
                        st.caption(f"Length: {query_length} chars")
                    
                    st.divider()
    
    else:
        st.info("👆 Generate queries first to see results here.")

with tabs[2]:  # Content Analysis Tab
    if st.session_state.last_results:
        st.markdown("### 🔍 Content Analysis")
        st.markdown("Analyze how well your content matches the generated queries")
        
        content_input = st.text_area(
            "Paste your content here",
            height=200,
            placeholder="Paste the content you want to analyze against the generated queries...",
            help="Enter the text content you want to analyze for keyword matches with your generated queries"
        )
        
        if st.button("🔬 Analyze Content", type="primary"):
            if content_input.strip():
                with st.spinner("Analyzing content matches..."):
                    analysis = analyze_content(content_input, st.session_state.last_results)
                    st.session_state.content_analysis = analysis
                    
                    # Display results
                    st.success(f"✅ Analysis complete! Found matches for {len(analysis)} queries.")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    high_matches = [a for a in analysis if a['match_score'] > 0.7]
                    medium_matches = [a for a in analysis if 0.3 < a['match_score'] <= 0.7]
                    low_matches = [a for a in analysis if a['match_score'] <= 0.3]
                    zero_matches = [a for a in analysis if a['match_score'] == 0]
                    
                    with col1:
                        st.metric("🟢 High Matches", len(high_matches), help="Queries with >70% keyword match")
                    with col2:
                        st.metric("🟡 Medium Matches", len(medium_matches), help="Queries with 30-70% keyword match")
                    with col3:
                        st.metric("🔴 Low Matches", len(low_matches), help="Queries with <30% keyword match")
                    with col4:
                        st.metric("⚫ No Matches", len(zero_matches), help="Queries with no keyword matches")
                    
                    # Detailed results
                    st.markdown("---")
                    st.markdown("#### 🎯 Query Match Analysis")
                    
                    # Filter options for analysis results
                    col1, col2 = st.columns(2)
                    with col1:
                        score_filter = st.selectbox(
                            "Filter by match score",
                            ["All Scores", "High (>70%)", "Medium (30-70%)", "Low (<30%)", "No Match (0%)"],
                            key="analysis_filter"
                        )
                    with col2:
                        sort_analysis = st.selectbox(
                            "Sort by",
                            ["Match Score (High to Low)", "Match Score (Low to High)", "Query Type", "Alphabetical"],
                            key="analysis_sort"
                        )
                    
                    # Apply filters
                    filtered_analysis = analysis.copy()
                    
                    if score_filter == "High (>70%)":
                        filtered_analysis = [a for a in filtered_analysis if a['match_score'] > 0.7]
                    elif score_filter == "Medium (30-70%)":
                        filtered_analysis = [a for a in filtered_analysis if 0.3 < a['match_score'] <= 0.7]
                    elif score_filter == "Low (<30%)":
                        filtered_analysis = [a for a in filtered_analysis if 0 < a['match_score'] <= 0.3]
                    elif score_filter == "No Match (0%)":
                        filtered_analysis = [a for a in filtered_analysis if a['match_score'] == 0]
                    
                    # Apply sorting
                    if sort_analysis == "Match Score (High to Low)":
                        filtered_analysis.sort(key=lambda x: x['match_score'], reverse=True)
                    elif sort_analysis == "Match Score (Low to High)":
                        filtered_analysis.sort(key=lambda x: x['match_score'])
                    elif sort_analysis == "Query Type":
                        filtered_analysis.sort(key=lambda x: x['type'])
                    elif sort_analysis == "Alphabetical":
                        filtered_analysis.sort(key=lambda x: x['query'].lower())
                    
                    # Display filtered results
                    if not filtered_analysis:
                        st.warning("No queries match your filter criteria.")
                    else:
                        st.markdown(f"**Showing {len(filtered_analysis)} of {len(analysis)} queries**")
                        
                        for idx, match in enumerate(filtered_analysis[:50], 1):  # Limit to 50 for performance
                            score = match['match_score']
                            
                            # Determine score styling
                            if score > 0.7:
                                score_color = "🟢"
                                score_bg = "background-color: #d4edda; border-left: 4px solid #28a745;"
                            elif score > 0.3:
                                score_color = "🟡"
                                score_bg = "background-color: #fff3cd; border-left: 4px solid #ffc107;"
                            elif score > 0:
                                score_color = "🔴"
                                score_bg = "background-color: #f8d7da; border-left: 4px solid #dc3545;"
                            else:
                                score_color = "⚫"
                                score_bg = "background-color: #f8f9fa; border-left: 4px solid #6c757d;"
                            
                            # Create expandable container
                            with st.container():
                                st.markdown(f"""
                                <div style='padding: 15px; margin: 10px 0; border-radius: 5px; {score_bg}'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div style='flex: 1;'>
                                            <strong>{idx}. {match['query']}</strong>
                                        </div>
                                        <div style='text-align: right; margin-left: 20px;'>
                                            <span style='font-size: 1.2em;'>{score_color} <strong>{score:.0%}</strong></span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show details in expandable section
                                with st.expander(f"View details for query {idx}", expanded=False):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        type_info = QUERY_TYPES.get(match['type'], {'name': match['type'], 'icon': '❓'})
                                        st.markdown(f"**Type:** {type_info['icon']} {type_info['name']}")
                                        st.markdown(f"**Priority:** {match.get('priority', 'N/A')}")
                                        st.markdown(f"**Confidence:** {match.get('confidence', 0):.0%}")
                                    
                                    with col2:
                                        if match.get('matched_words'):
                                            st.markdown(f"**Matched Keywords:** {', '.join(match['matched_words'])}")
                                        else:
                                            st.markdown("**Matched Keywords:** None")
                                        
                                        if match.get('query_keywords'):
                                            st.markdown(f"**All Query Keywords:** {', '.join(match['query_keywords'])}")
                    
                    # Export analysis
                    st.markdown("---")
                    st.markdown("#### 📤 Export Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 Export Analysis as CSV"):
                            analysis_df = pd.DataFrame(analysis)
                            # Add type names for readability
                            analysis_df['type_name'] = analysis_df['type'].apply(
                                lambda x: QUERY_TYPES.get(x, {'name': x}).get('name', x)
                            )
                            csv = analysis_df.to_csv(index=False)
                            st.download_button(
                                "Download Analysis CSV",
                                data=csv,
                                file_name=f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        if st.button("📋 Export Top Matches JSON"):
                            top_matches = [a for a in analysis if a['match_score'] > 0.3]
                            json_data = json.dumps(top_matches, indent=2)
                            st.download_button(
                                "Download Top Matches JSON",
                                data=json_data,
                                file_name=f"top_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with col3:
                        if st.button("🎯 Export Content Gaps"):
                            gaps = [a for a in analysis if a['match_score'] < 0.3]
                            gaps_df = pd.DataFrame(gaps)
                            if not gaps_df.empty:
                                gaps_df['type_name'] = gaps_df['type'].apply(
                                    lambda x: QUERY_TYPES.get(x, {'name': x}).get('name', x)
                                )
                                csv = gaps_df.to_csv(index=False)
                                st.download_button(
                                    "Download Content Gaps CSV",
                                    data=csv,
                                    file_name=f"content_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No content gaps found - all queries have good matches!")
            else:
                st.warning("⚠️ Please enter content to analyze")
    else:
        st.info("👆 Generate queries first to enable content analysis.")

with tabs[3]:  # Visualization Tab
    if st.session_state.last_results and include_mindmap:
        st.markdown("### 🗺️ Query Visualization")
        st.markdown("Interactive hierarchical view of query relationships")
        
        # Visualization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_type = st.selectbox(
                "Visualization Type",
                ["Treemap", "Sunburst"],
                help="Choose your preferred visualization style"
            )
        
        with col2:
            include_match_scores = st.checkbox(
                "Show Content Match Scores",
                value=bool(st.session_state.content_analysis),
                disabled=not bool(st.session_state.content_analysis),
                help="Run content analysis first to see match scores"
            )
        
        with col3:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["RdYlGn", "Viridis", "Blues", "Reds", "Plasma"],
                help="Choose color scheme for the visualization"
            )
        
        # Create visualization
        try:
            if viz_type == "Treemap":
                viz_fig = create_hierarchical_visualization(
                    st.session_state.last_results, 
                    st.session_state.last_query,
                    st.session_state.content_analysis if include_match_scores else None
                )
            else:  # Sunburst
                viz_fig = create_sunburst_visualization(
                    st.session_state.last_results,
                    st.session_state.last_query,
                    st.session_state.content_analysis if include_match_scores else None
                )
            
            # Update color scheme
            viz_fig.update_traces(
                marker=dict(colorscale=color_scheme)
            )
            
            # Display visualization
            st.plotly_chart(viz_fig, use_container_width=True)
            
            # Legend and interpretation
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Color Legend")
                if include_match_scores and st.session_state.content_analysis:
                    st.success("🟢 High Content Match (>70%)")
                    st.warning("🟡 Medium Content Match (30-70%)")
                    st.error("🔴 Low Content Match (<30%)")
                else:
                    st.info("💙 Colors represent confidence scores")
                    st.caption("Higher confidence = More relevant to original query")
            
            with col2:
                st.markdown("#### 🎯 Interpretation Guide")
                if viz_type == "Treemap":
                    st.markdown("""
                    - **Size** = Number of queries in each category
                    - **Color** = Confidence or match score
                    - **Sections** = Different query types
                    - Larger, greener sections = High-value areas
                    """)
                else:
                    st.markdown("""
                    - **Inner ring** = Query categories
                    - **Outer ring** = Individual queries  
                    - **Color** = Confidence or match score
                    - Click sections to focus/zoom
                    """)
            
            # Export visualization options
            st.markdown("---")
            st.markdown("#### 📤 Export Visualization")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📸 Save as PNG"):
                    img_bytes = viz_fig.to_image(format="png", width=1600, height=1200, scale=2)
                    st.download_button(
                        "Download High-Res PNG",
                        data=img_bytes,
                        file_name=f"query_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
            
            with col2:
                if st.button("📊 Save as Interactive HTML"):
                    html_str = viz_fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        "Download Interactive HTML",
                        data=html_str,
                        file_name=f"query_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            
            with col3:
                if st.button("🎨 Save as SVG"):
                    svg_bytes = viz_fig.to_image(format="svg")
                    st.download_button(
                        "Download SVG",
                        data=svg_bytes,
                        file_name=f"query_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                        mime="image/svg+xml"
                    )
            
            with col4:
                if st.button("📈 Save as PDF"):
                    pdf_bytes = viz_fig.to_image(format="pdf", width=1600, height=1200)
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name=f"query_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("💡 Try generating queries first, or check if all required data is available.")
    
    else:
        st.info("👆 Generate queries with visualization option enabled to see the hierarchical view.")

with tabs[4]:  # History Tab
    st.markdown("### 📚 Query Generation History")
    
    if st.session_state.query_history:
        # History controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            history_limit = st.selectbox(
                "Show entries",
                [5, 10, 20, 50, "All"],
                index=1,
                help="Number of history entries to display"
            )
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
        
        with col3:
            if st.button("📊 Export History"):
                history_df = pd.DataFrame(st.session_state.query_history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    "Download History CSV",
                    data=csv,
                    file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Display history
        entries_to_show = st.session_state.query_history
        if history_limit != "All":
            entries_to_show = entries_to_show[-history_limit:]
        
        for idx, entry in enumerate(reversed(entries_to_show), 1):
            with st.expander(
                f"{idx}. {entry.get('timestamp', 'Unknown time')} - {entry.get('query', 'Unknown query')[:50]}... "
                f"({entry.get('results_count', 0)} queries)",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Query:** {entry.get('query', 'N/A')}")
                    st.markdown(f"**Mode:** {entry.get('mode', 'N/A')}")
                    st.markdown(f"**Results:** {entry.get('results_count', 0)} queries")
                
                with col2:
                    st.markdown(f"**Provider:** {entry.get('provider', 'N/A')}")
                    st.markdown(f"**Model:** {entry.get('model', 'N/A')}")
                    if 'personalization' in entry and entry['personalization']:
                        st.markdown("**Personalization:** Applied")
                    else:
                        st.markdown("**Personalization:** None")
    else:
        st.info("📝 No history yet. Start generating queries to build your history!")
        st.markdown("""
        **Your history will include:**
        - 📅 Timestamp of each generation
        - 🔍 Original queries used  
        - 🤖 AI provider and model
        - ⚙️ Generation settings
        - 📊 Results summary
        """)

with tabs[5]:  # Resources Tab
    st.markdown("### 📖 Resources & Documentation")
    
    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started with heLLiuM
        
        **1. Set Up Your API Key**
        - Choose your AI provider (Gemini, OpenAI, or Claude)
        - Enter your API key in the sidebar
        - Select your preferred model
        
        **2. Configure Your Query**
        - Enter your main search query
        - Choose a search mode based on your needs:
          - **AI Overview**: 12-20 queries for quick exploration
          - **AI Mode (Complex)**: 20-35 queries for comprehensive coverage
          - **Research Mode**: 35-50 queries for deep research
          - **Comparative Analysis**: Focus on comparisons
          - **Multi-Perspective**: Various viewpoints
        
        **3. Generate Queries**
        - Click "Generate Queries" to create variations
        - Review results in the Results tab
        - Filter by type, priority, or confidence
        
        **4. Analyze Content (Optional)**
        - Paste your content in the Content Analysis tab
        - See how well your content matches the queries
        - Export analysis results
        
        **5. Visualize & Export**
        - View hierarchical visualizations
        - Export as CSV, PDF, or JSON
        - Save visualizations as images
        """)
    
    # About the Technology
    with st.expander("🧠 Understanding the Technology"):
        st.markdown("""
        ### The Google Patent Explained
        
        heLLiuM is inspired by Google's "Search with Stateful Chat" patent, which describes how modern search engines generate multiple query variations to better understand user intent.
        
        **Key Concepts:**
        
        📍 **Query Fan-Out**: Instead of processing just one query, the system generates many related queries to explore different aspects of your search intent.
        
        🔄 **Query Types**:
        - **Reformulations**: Different ways to phrase the same question
        - **Related Queries**: Adjacent topics you might be interested in
        - **Implicit Queries**: Hidden questions within your main query
        - **Comparative Queries**: Comparisons you might want to make
        - **Entity Expansions**: Deep dives on specific things mentioned
        - **Personalized Queries**: Variations based on context
        
        🎯 **Why This Matters**:
        - Better content coverage for SEO
        - Understanding user search behavior
        - Creating comprehensive content strategies
        - Identifying content gaps
        - Improving search relevance
        
        💡 **Real-World Application**: When you search for "best electric SUV", Google might internally generate queries like:
        - "electric SUV comparison 2024"
        - "Tesla Model Y vs competitors"
        - "electric SUV range anxiety"
        - "cost of owning electric SUV"
        - And dozens more...
        
        heLLiuM gives you access to that same powerful capability!
        """)
    
    # Advanced Features
    with st.expander("🔬 Advanced Features & Tips"):
        st.markdown("""
        ### Pro Tips for Power Users
        
        **🎨 Visualization Features**
        - Use Treemap view for hierarchical analysis
        - Switch to Sunburst for radial exploration
        - Colors indicate confidence or match scores
        - Click sections to zoom in (Sunburst)
        
        **📊 Content Analysis**
        - Match scores show keyword overlap
        - High matches (>70%) indicate well-covered topics
        - Low matches (<30%) reveal content gaps
        - Use this for content optimization
        
        **🤖 Multi-Provider Strategy**
        - Gemini: Fast and cost-effective
        - GPT-4: Advanced reasoning
        - Claude: Nuanced understanding
        - Try different providers for different perspectives
        
        **📤 Export Strategies**
        - CSV: For spreadsheet analysis
        - PDF: For client reports
        - JSON: For programmatic use
        - HTML visualizations: For interactive sharing
        
        **⚡ Performance Tips**
        - Lower temperature for consistent results
        - Higher temperature for creative variations
        - Adjust max tokens based on query complexity
        - Use categories to filter model selection
        """)
    
    # Credits and Attribution
    with st.expander("👥 Credits & Attribution", expanded=True):
        st.markdown("""
        ### 🏆 heLLiuM Team
        
        #### 🚀 **Version 3.0 Developer**
        **Tyler Einberger** - Enhanced and expanded Qforia into heLLiuM with advanced features
        
        Connect with Tyler:
        - 💼 [LinkedIn](https://www.linkedin.com/in/tyler-einberger)
        - 🌐 [Personal Website](https://www.tylereinberger.com/)
        - 🏢 [Momentic Marketing](https://momenticmarketing.com/team/tyler-einberger)
        - 🏙️ [MKE DMC](https://www.mkedmc.org/people/tyler-einberger)
        
        #### 🎨 **Original Creator**
        **Mike King** - Created Qforia 1.0, the foundation for this tool
        - 🔗 [Original Qforia](https://qforia.streamlit.app/)
        
        Special thanks to Mike for creating the original concept and making it open source!
        
        ---
        
        ### 📜 Version History
        - **v3.0** (Current - heLLiuM): Multi-provider support, visualizations, content analysis, PDF reports
        - **v2.0**: Enhanced UI, confidence scoring, session history
        - **v1.0**: Original by Mike King - Core query generation concept
        """)
    
    # API Resources - SEPARATE EXPANDER (not nested)
    with st.expander("🔗 API Documentation & Resources"):
        st.markdown("""
        ### Getting API Keys
        
        **Google Gemini**
        - 🔗 [Get API Key](https://makersuite.google.com/app/apikey)
        - 📚 [Documentation](https://ai.google.dev/docs)
        - 💰 Free tier available
        
        **OpenAI**
        - 🔗 [Get API Key](https://platform.openai.com/api-keys)
        - 📚 [Documentation](https://platform.openai.com/docs)
        - 💰 Pay-as-you-go pricing
        
        **Anthropic Claude**
        - 🔗 [Get API Key](https://console.anthropic.com/)
        - 📚 [Documentation](https://docs.anthropic.com/)
        - 💰 Usage-based pricing
        
        ### Model Recommendations
        
        **For Speed & Cost**: 
        - Gemini 1.5 Flash
        - GPT-4o mini
        - Claude Haiku
        
        **For Quality**:
        - Gemini 2.5 Pro
        - GPT-4
        - Claude Opus/Sonnet
        
        **For Balance**:
        - Gemini 2.0 Flash
        - GPT-4o
        - Claude Sonnet 3.5
        """)
    
    # Use Cases
    with st.expander("💡 Use Cases & Examples"):
        st.markdown("""
        ### How Different Teams Use heLLiuM
        
        **SEO Professionals**
        - Keyword research and expansion
        - Content gap analysis
        - Search intent understanding
        - Competitor content planning
        
        **Content Marketers**
        - Blog topic ideation
        - Content cluster planning
        - FAQ generation
        - User question discovery
        
        **Product Teams**
        - Feature request analysis
        - User need exploration
        - Documentation planning
        - Support content creation
        
        **Researchers**
        - Literature review queries
        - Research question refinement
        - Topic exploration
        - Grant proposal keywords
        
        **Example Workflow**:
        1. Start with: "sustainable packaging solutions"
        2. Generate 35 queries in Research Mode
        3. Analyze existing content against queries
        4. Identify gaps where match scores are low
        5. Create content plan based on gaps
        6. Export visualization for team presentation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h4>🔮 heLLiuM - LLM Query Fan Out Simulator</h4>
        <p>Built with ❤️ by Tyler Einberger, based on original work by Mike King</p>
        <p>Powered by Google Gemini, OpenAI, and Anthropic Claude</p>
        <br>
        <p><em>"Multiplied intelligence for search understanding."</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>heLLiuM v3.0 | LLM Query Fan Out Simulator</p>
    <p>Created by <a href='https://www.linkedin.com/in/tyler-einberger' target='_blank'>Tyler Einberger</a> | 
    Based on <a href='https://qforia.streamlit.app/' target='_blank'>original</a> by Mike King</p>
</div>
""", unsafe_allow_html=True)
