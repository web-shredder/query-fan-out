import streamlit as st
import pandas as pd
import json
import re
import time
from datetime import datetime
import hashlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from io import BytesIO
import base64

# Try to import all providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# App config
st.set_page_config(page_title="heLLiuM 3.0", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 10px 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .query-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 5px 0;
    }
    .match-score {
        padding: 5px 10px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
    }
    .high-match { background-color: #28a745; }
    .medium-match { background-color: #ffc107; }
    .low-match { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

st.title("heLLiuM 3.0")
st.markdown("*Powered by Gemini, OpenAI, and Claude with Advanced Analytics*")

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'generation_sessions' not in st.session_state:
    st.session_state.generation_sessions = []
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'content_analysis' not in st.session_state:
    st.session_state.content_analysis = None

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Provider Selection
with st.sidebar.expander("🤖 AI Provider Configuration", expanded=True):
    available_providers = []
    if GEMINI_AVAILABLE:
        available_providers.append("Google Gemini")
    if OPENAI_AVAILABLE:
        available_providers.append("OpenAI")
    if ANTHROPIC_AVAILABLE:
        available_providers.append("Anthropic Claude")
    
    if not available_providers:
        st.error("No AI providers available. Please install at least one: google-generativeai, openai, or anthropic")
        st.stop()
    
    selected_provider = st.selectbox("Select AI Provider", available_providers)
    
    # Provider-specific configuration
    if selected_provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
        
        # Group models by category
        gemini_models = {
            "Latest & Fast": [
                ("gemini-2.5-flash-preview-05-20", "2.5 Flash - Adaptive thinking, cost efficient"),
                ("gemini-2.0-flash", "2.0 Flash - Next gen features, speed, thinking"),
                ("gemini-2.0-flash-lite", "2.0 Flash-Lite - Cost efficient, low latency"),
                ("gemini-1.5-flash", "1.5 Flash - Fast & versatile"),
                ("gemini-1.5-flash-8b", "1.5 Flash-8B - High volume, lower intelligence")
            ],
            "Advanced & Pro": [
                ("gemini-2.5-pro-preview-05-06", "2.5 Pro - Enhanced reasoning, multimodal"),
                ("gemini-1.5-pro", "1.5 Pro - Complex reasoning tasks")
            ],
            "Specialized": [
                ("gemini-2.5-flash-preview-native-audio-dialog", "2.5 Flash Audio - Natural conversations"),
                ("gemini-2.0-flash-preview-image-generation", "2.0 Flash - Image generation"),
                ("gemini-embedding-exp", "Embedding - Text relatedness")
            ]
        }
        
        model_category = st.selectbox(
            "Model Category",
            list(gemini_models.keys()),
            help="Choose a category based on your needs"
        )
        
        model_options = gemini_models[model_category]
        selected_model = st.selectbox(
            "Select Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: next(m[1] for m in model_options if m[0] == x)
        )
        
    elif selected_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        
        # Group models by category
        openai_models = {
            "Reasoning Models": [
                ("o4-mini", "o4-mini - Faster, affordable reasoning"),
                ("o3", "o3 - Most powerful reasoning model"),
                ("o3-mini", "o3-mini - Small alternative to o3"),
                ("o1", "o1 - Previous full reasoning model"),
                ("o1-pro", "o1-pro - More compute for better responses")
            ],
            "Flagship Chat Models": [
                ("gpt-4.1", "GPT-4.1 - Flagship for complex tasks"),
                ("gpt-4o", "GPT-4o - Fast, intelligent, flexible"),
                ("gpt-4o-audio", "GPT-4o Audio - Audio inputs/outputs"),
                ("chatgpt-4o", "ChatGPT-4o - Used in ChatGPT")
            ],
            "Cost-Optimized Models": [
                ("gpt-4.1-mini", "GPT-4.1 mini - Balanced intelligence/cost"),
                ("gpt-4.1-nano", "GPT-4.1 nano - Fastest, most cost-effective"),
                ("gpt-4o-mini", "GPT-4o mini - Small model for focused tasks"),
                ("gpt-4o-mini-audio", "GPT-4o mini Audio - Small with audio")
            ]
        }
        
        model_category = st.selectbox(
            "Model Category",
            list(openai_models.keys()),
            help="Choose based on reasoning needs vs cost"
        )
        
        model_options = openai_models[model_category]
        selected_model = st.selectbox(
            "Select Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: next(m[1] for m in model_options if m[0] == x)
        )
        
    elif selected_provider == "Anthropic Claude":
        api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        
        # Group models by generation
        anthropic_models = {
            "Claude 4 (Latest)": [
                ("claude-opus-4-20250514", "Opus 4 - Most capable"),
                ("claude-sonnet-4-20250514", "Sonnet 4 - Balanced performance")
            ],
            "Claude 3.5/3.7": [
                ("claude-3-7-sonnet-latest", "Sonnet 3.7 - Latest 3.7"),
                ("claude-3-5-sonnet-latest", "Sonnet 3.5 v2 - Latest 3.5"),
                ("claude-3-5-sonnet-20240620", "Sonnet 3.5 - Previous version"),
                ("claude-3-5-haiku-latest", "Haiku 3.5 - Fast & efficient")
            ],
            "Claude 3": [
                ("claude-3-opus-latest", "Opus 3 - Most capable v3"),
                ("claude-3-sonnet-20240229", "Sonnet 3 - Balanced v3"),
                ("claude-3-haiku-20240307", "Haiku 3 - Fast v3")
            ]
        }
        
        model_category = st.selectbox(
            "Model Generation",
            list(anthropic_models.keys()),
            help="Newer generations have better capabilities"
        )
        
        model_options = anthropic_models[model_category]
        selected_model = st.selectbox(
            "Select Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: next(m[1] for m in model_options if m[0] == x)
        )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)

# Model availability note - separate sidebar expander
with st.sidebar.expander("ℹ️ Model Availability Notes"):
    st.markdown("""
    **Important Notes:**
    - Some models may require specific API access or waitlist approval
    - Audio/Image generation models won't work for text query generation
    - Embedding models are for similarity, not generation
    - Newer models (o3, o4, Claude 4) may have limited availability
    - Check your API tier for model access
    """)
    
    if 'selected_provider' in locals():
        if selected_provider == "OpenAI":
            st.warning("Note: o3, o4, and GPT-4.1 models may require special access")
        elif selected_provider == "Google Gemini":
            st.info("Audio/TTS models are specialized - use text models for query generation")

# Query Configuration
with st.sidebar.expander("🎯 Query Settings", expanded=True):
    user_query = st.text_area(
        "Primary Query", 
        "What's the best electric SUV for driving up mt rainier?",
        height=100
    )
    
    mode = st.radio(
        "Search Mode",
        [
            "AI Overview (Simple)",
            "AI Mode (Complex)",
            "Research Mode (Deep)",
            "Comparative Analysis",
            "Multi-Perspective"
        ]
    )
    
    # Advanced options
    include_mindmap = st.checkbox("Generate Mind Map", value=True)
    include_analysis = st.checkbox("Enable Content Analysis", value=True)

# Export Settings
with st.sidebar.expander("📤 Export Settings"):
    export_format = st.multiselect(
        "Export Formats",
        ["CSV", "PDF Report", "JSON"],
        default=["CSV"]
    )
    include_visualizations = st.checkbox("Include Visualizations in PDF", value=True)
    pdf_size = st.radio("PDF Page Size", ["Letter", "A4"], index=0)

# Configure AI Provider
if api_key:
    if selected_provider == "Google Gemini" and GEMINI_AVAILABLE:
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel(selected_model)
    elif selected_provider == "OpenAI" and OPENAI_AVAILABLE:
        openai.api_key = api_key
        ai_model = selected_model
    elif selected_provider == "Anthropic Claude" and ANTHROPIC_AVAILABLE:
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        ai_model = selected_model
else:
    st.error(f"⚠️ Please enter your {selected_provider} API Key to proceed.")
    st.stop()

# Query Types
QUERY_TYPES = {
    "reformulation": {"name": "Reformulations", "icon": "🔄", "color": "#FF6B6B"},
    "related": {"name": "Related Queries", "icon": "🔗", "color": "#4ECDC4"},
    "implicit": {"name": "Implicit Queries", "icon": "💭", "color": "#45B7D1"},
    "comparative": {"name": "Comparative", "icon": "⚖️", "color": "#FFA07A"},
    "entity_expansion": {"name": "Entity Focus", "icon": "🎯", "color": "#98D8C8"},
    "personalized": {"name": "Personalized", "icon": "👤", "color": "#F7DC6F"},
    "temporal": {"name": "Temporal", "icon": "⏰", "color": "#BB8FCE"},
    "location": {"name": "Location-based", "icon": "📍", "color": "#85C1E2"},
    "technical": {"name": "Technical", "icon": "🔧", "color": "#F8C471"},
    "user_intent": {"name": "Intent", "icon": "🎯", "color": "#82E0AA"}
}

def generate_query_id(query):
    """Generate unique ID for each query"""
    return hashlib.md5(query.encode()).hexdigest()[:8]

def calculate_confidence_score(query, original_query, query_type):
    """Calculate confidence score for generated query"""
    base_score = 0.7
    common_words = set(original_query.lower().split()) & set(query.lower().split())
    similarity_bonus = len(common_words) / len(set(original_query.lower().split())) * 0.2
    type_scores = {
        "reformulation": 0.05,
        "related": 0.03,
        "implicit": 0.08,
        "comparative": 0.04,
        "entity_expansion": 0.06
    }
    type_bonus = type_scores.get(query_type, 0.02)
    return min(base_score + similarity_bonus + type_bonus, 0.99)

def get_ai_response(prompt, provider, model):
    """Unified function to get AI response from any provider"""
    try:
        if provider == "Google Gemini" and GEMINI_AVAILABLE:
            response = ai_model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
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
            response = anthropic_client.messages.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
    
    except Exception as e:
        st.error(f"Error with {provider}: {str(e)}")
        return None

def create_query_prompt(query, mode):
    """Create prompt for query generation"""
    mode_configs = {
        "AI Overview (Simple)": {"min": 12, "max": 20},
        "AI Mode (Complex)": {"min": 20, "max": 35},
        "Research Mode (Deep)": {"min": 35, "max": 50},
        "Comparative Analysis": {"min": 15, "max": 25},
        "Multi-Perspective": {"min": 25, "max": 40}
    }
    
    config = mode_configs.get(mode, mode_configs["AI Mode (Complex)"])
    
    return f"""Generate diverse search queries based on this input query: "{query}"

Create {config['min']} to {config['max']} queries using these types:
- Reformulations: Alternative phrasings
- Related: Adjacent topics  
- Implicit: Hidden questions
- Comparative: Comparisons
- Entity Expansions: Entity focus
- Personalized: Context-aware
- Temporal: Time-based
- Location: Geographic
- Technical: Specifications
- Intent: User goals

Return ONLY a JSON array:
[
  {{
    "query": "generated query text",
    "type": "reformulation|related|implicit|comparative|entity_expansion|personalized|temporal|location|technical|user_intent",
    "intent": "what the user wants",
    "reasoning": "why useful",
    "priority": "high|medium|low"
  }}
]"""

def clean_json_response(text):
    """Clean and extract JSON from response"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    # Find JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        return array_match.group(0)
    
    # Find JSON object and wrap in array
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        return f"[{obj_match.group(0)}]"
    
    return text.strip()

def generate_queries(query, mode, provider, model):
    """Generate queries using selected AI provider"""
    prompt = create_query_prompt(query, mode)
    
    with st.spinner(f"🧠 Generating queries with {provider}..."):
        response_text = get_ai_response(prompt, provider, model)
        
        if not response_text:
            return []
        
        # Clean response
        json_text = clean_json_response(response_text)
        
        try:
            queries = json.loads(json_text)
            
            # Ensure it's a list
            if isinstance(queries, dict):
                queries = [queries]
            
            # Add IDs and confidence
            for q in queries:
                q["id"] = generate_query_id(q.get("query", ""))
                q["confidence"] = calculate_confidence_score(
                    q.get("query", ""), 
                    query, 
                    q.get("type", "related")
                )
            
            return queries
            
        except json.JSONDecodeError:
            st.error("Failed to parse AI response")
            return []

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

def generate_pdf_report(queries, original_query, analysis_results=None):
    """Generate a PDF report of the query analysis"""
    if not REPORTLAB_AVAILABLE:
        st.error("PDF generation requires reportlab. Install with: pip install reportlab")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter if pdf_size == "Letter" else A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#FF6B6B'),
        alignment=TA_CENTER
    )
    story.append(Paragraph("Qforia Pro Query Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Original Query
    story.append(Paragraph(f"<b>Original Query:</b> {original_query}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Queries Generated:</b> {len(queries)}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Query Distribution
    type_counts = {}
    for q in queries:
        qtype = q.get('type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    # Create distribution table
    dist_data = [['Query Type', 'Count', 'Percentage']]
    for qtype, count in type_counts.items():
        type_info = QUERY_TYPES.get(qtype, {"name": qtype})
        percentage = f"{(count/len(queries)*100):.1f}%"
        dist_data.append([type_info['name'], str(count), percentage])
    
    dist_table = Table(dist_data)
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(Paragraph("<b>Query Distribution</b>", styles['Heading2']))
    story.append(dist_table)
    story.append(Spacer(1, 20))
    
    # Generated Queries
    story.append(Paragraph("<b>Generated Queries</b>", styles['Heading2']))
    
    # Group by type
    grouped = {}
    for q in queries:
        qtype = q.get('type', 'unknown')
        if qtype not in grouped:
            grouped[qtype] = []
        grouped[qtype].append(q)
    
    for qtype, type_queries in grouped.items():
        type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓"})
        story.append(Paragraph(f"{type_info['icon']} <b>{type_info['name']}</b>", styles['Heading3']))
        
        for q in type_queries:
            story.append(Paragraph(f"• {q['query']}", styles['Normal']))
            story.append(Paragraph(f"  <i>Intent: {q.get('intent', 'N/A')}</i>", styles['Normal']))
            story.append(Paragraph(f"  <i>Confidence: {q.get('confidence', 0):.0%}</i>", styles['Normal']))
            story.append(Spacer(1, 10))
    
    # Content Analysis Results
    if analysis_results:
        story.append(PageBreak())
        story.append(Paragraph("<b>Content Analysis Results</b>", styles['Heading2']))
        
        # Top matches
        top_matches = [r for r in analysis_results if r['match_score'] > 0.3][:10]
        
        if top_matches:
            match_data = [['Query', 'Match Score', 'Type']]
            for match in top_matches:
                score_str = f"{match['match_score']:.0%}"
                match_data.append([
                    match['query'][:50] + "...",
                    score_str,
                    QUERY_TYPES.get(match['type'], {}).get('name', match['type'])
                ])
            
            match_table = Table(match_data)
            match_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(match_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

def export_to_csv(queries):
    """Export queries to CSV format"""
    df = pd.DataFrame(queries)
    
    # Reorder columns for better readability
    column_order = ['query', 'type', 'intent', 'reasoning', 'priority', 'confidence']
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Add type names
    df['type_name'] = df['type'].apply(lambda x: QUERY_TYPES.get(x, {}).get('name', x))
    
    return df.to_csv(index=False)

# Main UI
tabs = st.tabs(["🚀 Generate", "📊 Results", "🔍 Content Analysis", "🗺️ Visualization", "📚 History", "📖 Resources"])

with tabs[0]:  # Generate Tab
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Query Configuration")
        st.info(f"**Current Query:** {user_query}")
        st.info(f"**Mode:** {mode}")
        st.info(f"**Provider:** {selected_provider} ({selected_model})")
    
    with col2:
        st.markdown("### 🎮 Actions")
        
        if st.button("🚀 Generate Queries", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("⚠️ Please enter a query.")
            else:
                # Generate queries
                results = generate_queries(user_query, mode, selected_provider, selected_model)
                
                if results:
                    st.session_state.last_results = results
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": user_query,
                        "mode": mode,
                        "provider": selected_provider,
                        "model": selected_model,
                        "results_count": len(results)
                    })
                    
                    st.success(f"✅ Generated {len(results)} queries!")
                    st.balloons()

with tabs[1]:  # Results Tab
    if st.session_state.last_results:
        st.markdown("### 📋 Generated Queries")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(st.session_state.last_results))
        with col2:
            unique_types = len(set(q.get('type', 'unknown') for q in st.session_state.last_results))
            st.metric("Query Types", unique_types)
        with col3:
            avg_confidence = sum(q.get('confidence', 0) for q in st.session_state.last_results) / len(st.session_state.last_results)
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        with col4:
            high_priority = len([q for q in st.session_state.last_results if q.get('priority') == 'high'])
            st.metric("High Priority", high_priority)
        
        # Display queries grouped by type
        grouped_queries = {}
        for q in st.session_state.last_results:
            qtype = q.get('type', 'unknown')
            if qtype not in grouped_queries:
                grouped_queries[qtype] = []
            grouped_queries[qtype].append(q)
        
        for qtype, queries in grouped_queries.items():
            type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓", "color": "#999"})
            
            with st.expander(f"{type_info['icon']} {type_info['name']} ({len(queries)} queries)", expanded=True):
                for idx, q in enumerate(queries):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{q['query']}**")
                        st.caption(f"💡 {q.get('intent', 'N/A')} | 📝 {q.get('reasoning', 'N/A')}")
                    
                    with col2:
                        priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        st.markdown(f"{priority_colors.get(q.get('priority', 'medium'), '⚪')} **{q.get('priority', 'medium').upper()}**")
                        st.markdown(f"**{q.get('confidence', 0):.0%}** confidence")
        
        # Export section
        st.markdown("---")
        st.markdown("### 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "CSV" in export_format:
                csv_data = export_to_csv(st.session_state.last_results)
                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=f"qforia_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if "PDF Report" in export_format and REPORTLAB_AVAILABLE:
                pdf_buffer = generate_pdf_report(st.session_state.last_results, user_query)
                if pdf_buffer:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"qforia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        with col3:
            if "JSON" in export_format:
                json_data = json.dumps(st.session_state.last_results, indent=2)
                st.download_button(
                    "📋 Download JSON",
                    data=json_data,
                    file_name=f"qforia_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        st.info("👆 Generate queries first to see results here.")

with tabs[2]:  # Content Analysis Tab
    if st.session_state.last_results:
        st.markdown("### 🔍 Content Analysis")
        st.markdown("Analyze how well your content matches the generated queries")
        
        content_input = st.text_area(
            "Paste your content here",
            height=200,
            placeholder="Paste the content you want to analyze against the generated queries..."
        )
        
        if st.button("🔬 Analyze Content"):
            if content_input:
                analysis = analyze_content(content_input, st.session_state.last_results)
                st.session_state.content_analysis = analysis
                
                # Display results
                st.markdown("#### 📊 Analysis Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                high_matches = [a for a in analysis if a['match_score'] > 0.7]
                medium_matches = [a for a in analysis if 0.3 < a['match_score'] <= 0.7]
                low_matches = [a for a in analysis if a['match_score'] <= 0.3]
                
                with col1:
                    st.metric("High Matches (>70%)", len(high_matches))
                with col2:
                    st.metric("Medium Matches (30-70%)", len(medium_matches))
                with col3:
                    st.metric("Low Matches (<30%)", len(low_matches))
                
                # Detailed results
                st.markdown("#### 🎯 Top Matching Queries")
                
                for match in analysis[:20]:  # Show top 20
                    score = match['match_score']
                    
                    if score > 0.7:
                        score_class = "high-match"
                    elif score > 0.3:
                        score_class = "medium-match"
                    else:
                        score_class = "low-match"
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{match['query']}**")
                        if match['matched_words']:
                            st.caption(f"Keywords: **{', '.join(match['matched_words'])}**")
                        else:
                            st.caption("No keyword matches")
                        
                        # Show query keywords for debugging
                        with st.expander("Debug Info", expanded=False):
                            st.caption(f"Query keywords: {', '.join(match.get('query_keywords', []))}")
                            st.caption(f"Matched: {match['matched_words']}")
                    
                    with col2:
                        st.markdown(f"<span class='match-score {score_class}'>{score:.0%}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col3:
                        type_info = QUERY_TYPES.get(match['type'], {})
                        st.markdown(f"{type_info.get('icon', '❓')} {type_info.get('name', match['type'])}")
                
                # Export analysis
                if st.button("📊 Export Analysis as CSV"):
                    analysis_df = pd.DataFrame(analysis)
                    csv = analysis_df.to_csv(index=False)
                    st.download_button(
                        "Download Analysis",
                        data=csv,
                        file_name=f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Please enter content to analyze")
    else:
        st.info("👆 Generate queries first to enable content analysis.")

with tabs[3]:  # Mind Map Tab
    if st.session_state.last_results and include_mindmap:
        st.markdown("### 🗺️ Query Visualization")
        st.markdown("Interactive hierarchical view of query relationships")
        
        # Visualization type selector
        viz_type = st.radio(
            "Visualization Type",
            ["Treemap", "Sunburst"],
            horizontal=True,
            help="Choose your preferred visualization style"
        )
        
        # Check if content analysis has been run
        include_match_scores = st.checkbox(
            "Show Content Match Scores",
            value=bool(st.session_state.content_analysis),
            disabled=not bool(st.session_state.content_analysis),
            help="Run content analysis first to see match scores"
        )
        
        # Create visualization
        if viz_type == "Treemap":
            viz_fig = create_hierarchical_visualization(
                st.session_state.last_results, 
                user_query,
                st.session_state.content_analysis if include_match_scores else None
            )
        else:  # Sunburst
            viz_fig = create_sunburst_visualization(
                st.session_state.last_results,
                user_query,
                st.session_state.content_analysis if include_match_scores else None
            )
        
        # Display visualization
        st.plotly_chart(viz_fig, use_container_width=True)
        
        # Legend
        if include_match_scores and st.session_state.content_analysis:
            st.markdown("#### 📊 Color Scale")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("🟢 High Match (>70%)")
            with col2:
                st.warning("🟡 Medium Match (30-70%)")
            with col3:
                st.error("🔴 Low Match (<30%)")
        else:
            st.info("💡 Colors represent confidence scores. Enable content match scores after running content analysis.")
        
        # Export options
        st.markdown("#### 📤 Export Visualization")
        col1, col2, col3 = st.columns(3)
        
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
    else:
        st.info("👆 Generate queries with visualization option enabled to see the hierarchical view.")

with tabs[4]:  # History Tab
    st.markdown("### 📚 Query Generation History")
    
    if st.session_state.query_history:
        # Display history in reverse chronological order
        for idx, entry in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
            with st.expander(f"{entry['timestamp']} - {entry['query'][:50]}... ({entry['results_count']} queries)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Query:** {entry['query']}")
                    st.markdown(f"**Mode:** {entry['mode']}")
                with col2:
                    st.markdown(f"**Provider:** {entry.get('provider', 'N/A')}")
                    st.markdown(f"**Model:** {entry.get('model', 'N/A')}")
    else:
        st.info("No history yet. Start generating queries!")

with tabs[5]:  # Resources Tab
    st.markdown("### 📖 Resources & Documentation")
    
    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started with HeLLiuM
        
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
        
        HeLLiuM is inspired by Google's "Search with Stateful Chat" patent, which describes how modern search engines generate multiple query variations to better understand user intent.
        
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
        
        HeLLiuM gives you access to that same powerful capability!
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
        ### 🏆 HeLLiuM Team
        
        #### 🚀 **Version 3.0 Developer**
        **Tyler Einberger** - Enhanced and expanded Qforia into HeLLiuM with advanced features
        
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
        - **v3.0** (Current - HeLLiuM): Multi-provider support, visualizations, content analysis, PDF reports
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
        ### How Different Teams Use Qforia Pro
        
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
        <h4>🔮 HeLLiuM - LLM Query Fan Out Simulator</h4>
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
    <p>HeLLiuM v3.0 | LLM Query Fan Out Simulator</p>
    <p>Created by <a href='https://www.linkedin.com/in/tyler-einberger' target='_blank'>Tyler Einberger</a> | 
    Based on <a href='https://qforia.streamlit.app/' target='_blank'>original</a> by Mike King</p>
</div>
""", unsafe_allow_html=True)
