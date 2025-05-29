import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import time
from datetime import datetime
import hashlib

# App config
st.set_page_config(page_title="Qforia Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
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
</style>
""", unsafe_allow_html=True)

st.title("🚀 Qforia Pro: Advanced AI Query Intelligence System")
st.markdown("*Powered by Multi-Model Query Generation & Contextual Understanding*")

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'generation_sessions' not in st.session_state:
    st.session_state.generation_sessions = []
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = []
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = {}

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# API Key Management
with st.sidebar.expander("🔑 API Configuration", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password", key="api_key")
    
    # Model Selection
    model_options = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash-exp"
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    
    # Advanced Settings
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 8000, 4000)

# Query Configuration
with st.sidebar.expander("🎯 Query Settings", expanded=True):
    user_query = st.text_area(
        "Primary Query", 
        "What's the best electric SUV for driving up mt rainier?",
        height=100,
        help="Enter your main search query"
    )
    
    # Search Mode Selection with descriptions
    mode = st.radio(
        "Search Mode",
        [
            "AI Overview (Simple)",
            "AI Mode (Complex)",
            "Research Mode (Deep)",
            "Comparative Analysis",
            "Multi-Perspective"
        ],
        help="Select the depth and complexity of search"
    )
    
    # Context Settings
    include_context = st.checkbox("Include User Context", value=True)
    include_related = st.checkbox("Generate Related Queries", value=True)
    include_implied = st.checkbox("Extract Implied Queries", value=True)
    
# Advanced Features
with st.sidebar.expander("🔬 Advanced Features"):
    enable_confidence = st.checkbox("Show Confidence Scores", value=True)
    enable_clustering = st.checkbox("Enable Query Clustering", value=True)
    enable_deduplication = st.checkbox("Remove Duplicate Queries", value=True)
    save_session = st.checkbox("Save Generation Session", value=True)

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(selected_model)
else:
    st.error("⚠️ Please enter your Gemini API Key to proceed.")
    st.stop()

# Enhanced Query Types based on Patent
QUERY_TYPES = {
    "reformulation": {
        "name": "Reformulations",
        "description": "Alternative phrasings of the original query",
        "icon": "🔄"
    },
    "related": {
        "name": "Related Queries",
        "description": "Queries exploring related aspects",
        "icon": "🔗"
    },
    "implicit": {
        "name": "Implicit Queries",
        "description": "Hidden questions within the main query",
        "icon": "💭"
    },
    "comparative": {
        "name": "Comparative Queries",
        "description": "Comparison-based explorations",
        "icon": "⚖️"
    },
    "entity_expansion": {
        "name": "Entity Expansions",
        "description": "Queries focusing on specific entities",
        "icon": "🎯"
    },
    "personalized": {
        "name": "Personalized Queries",
        "description": "Context-aware personalized queries",
        "icon": "👤"
    },
    "temporal": {
        "name": "Temporal Queries",
        "description": "Time-based variations",
        "icon": "⏰"
    },
    "location": {
        "name": "Location Queries",
        "description": "Geographic variations",
        "icon": "📍"
    },
    "technical": {
        "name": "Technical Deep-Dives",
        "description": "Technical specifications and details",
        "icon": "🔧"
    },
    "user_intent": {
        "name": "Intent Clarifications",
        "description": "Clarifying user's actual intent",
        "icon": "🎪"
    }
}

def generate_query_id(query):
    """Generate unique ID for each query"""
    return hashlib.md5(query.encode()).hexdigest()[:8]

def calculate_confidence_score(query, original_query, query_type):
    """Calculate confidence score for generated query"""
    # Simple heuristic - in production, use ML model
    base_score = 0.7
    
    # Adjust based on similarity to original
    common_words = set(original_query.lower().split()) & set(query.lower().split())
    similarity_bonus = len(common_words) / len(set(original_query.lower().split())) * 0.2
    
    # Type-based adjustments
    type_scores = {
        "reformulation": 0.05,
        "related": 0.03,
        "implicit": 0.08,
        "comparative": 0.04,
        "entity_expansion": 0.06
    }
    type_bonus = type_scores.get(query_type, 0.02)
    
    return min(base_score + similarity_bonus + type_bonus, 0.99)

def ENHANCED_QUERY_FANOUT_PROMPT(q, mode, context=None):
    """Enhanced prompt based on Google's patent methodology"""
    
    mode_configs = {
        "AI Overview (Simple)": {"min": 15, "max": 25, "depth": "basic"},
        "AI Mode (Complex)": {"min": 30, "max": 50, "depth": "comprehensive"},
        "Research Mode (Deep)": {"min": 50, "max": 80, "depth": "exhaustive"},
        "Comparative Analysis": {"min": 25, "max": 40, "depth": "comparative"},
        "Multi-Perspective": {"min": 40, "max": 60, "depth": "multi-angle"}
    }
    
    config = mode_configs.get(mode, mode_configs["AI Mode (Complex)"])
    
    context_str = ""
    if context:
        context_str = f"\nUser Context: {json.dumps(context)}\n"
    
    return f"""You are an advanced AI search query generation system implementing Google's stateful chat methodology.

Original Query: "{q}"
Mode: {mode}
Target Complexity: {config['depth']}
{context_str}

CRITICAL REQUIREMENTS:
1. Generate between {config['min']} and {config['max']} queries
2. Each query must be unique and add value
3. Include ALL query types from the taxonomy below
4. Queries should progressively build understanding
5. Consider temporal, geographic, and user intent factors

QUERY TAXONOMY (all types MUST be represented):
1. Reformulations - Alternative phrasings maintaining intent
2. Related Queries - Exploring adjacent topics
3. Implicit Queries - Uncovering hidden questions
4. Comparative Queries - Comparison-based explorations
5. Entity Expansions - Deep dives on mentioned entities
6. Personalized Queries - Context-aware variations
7. Temporal Queries - Time-based considerations
8. Location Queries - Geographic relevance
9. Technical Deep-Dives - Specifications and technical details
10. Intent Clarifications - Understanding true user needs

For the query "{q}", consider:
- What comparison is the user making?
- What entities need expansion?
- What temporal factors matter?
- What location-specific factors apply?
- What technical specifications are relevant?
- What might the user's ultimate goal be?

Return a JSON object with this EXACT structure:
{{
  "generation_metadata": {{
    "timestamp": "ISO timestamp",
    "model": "model name",
    "temperature": {temperature},
    "original_query_analysis": {{
      "entities": ["list of key entities"],
      "intent": "primary user intent",
      "complexity": "low/medium/high",
      "domain": "query domain"
    }},
    "generation_strategy": {{
      "total_queries": integer,
      "distribution": {{
        "reformulation": integer,
        "related": integer,
        "implicit": integer,
        "comparative": integer,
        "entity_expansion": integer,
        "personalized": integer,
        "temporal": integer,
        "location": integer,
        "technical": integer,
        "user_intent": integer
      }},
      "reasoning": "detailed explanation"
    }}
  }},
  "expanded_queries": [
    {{
      "id": "unique_id",
      "query": "the generated query",
      "type": "query type from taxonomy",
      "user_intent": "specific intent addressed",
      "reasoning": "why this query adds value",
      "confidence": 0.0-1.0,
      "priority": "high/medium/low",
      "entities": ["relevant entities"],
      "temporal_relevance": "if applicable",
      "geographic_relevance": "if applicable"
    }}
  ],
  "query_relationships": [
    {{
      "parent_id": "query_id",
      "child_id": "query_id",
      "relationship": "follows_from/contrasts_with/expands_on"
    }}
  ]
}}"""

def generate_enhanced_fanout(query, mode, context=None):
    """Generate queries using enhanced methodology"""
    prompt = ENHANCED_QUERY_FANOUT_PROMPT(query, mode, context)
    
    try:
        with st.spinner("🧠 Generating intelligent query variations..."):
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            
            json_text = response.text.strip()
            
            # Clean response
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            data = json.loads(json_text.strip())
            
            # Store metadata
            st.session_state.last_generation_metadata = data.get("generation_metadata", {})
            st.session_state.query_relationships = data.get("query_relationships", [])
            
            # Calculate confidence scores if not provided
            queries = data.get("expanded_queries", [])
            for q in queries:
                if "confidence" not in q:
                    q["confidence"] = calculate_confidence_score(
                        q["query"], query, q["type"]
                    )
                if "id" not in q:
                    q["id"] = generate_query_id(q["query"])
            
            return queries
            
    except Exception as e:
        st.error(f"🔴 Error: {str(e)}")
        return None

def display_generation_metadata():
    """Display generation metadata in an attractive format"""
    if hasattr(st.session_state, 'last_generation_metadata'):
        meta = st.session_state.last_generation_metadata
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", 
                     len(st.session_state.get('last_results', [])),
                     delta=None)
        
        with col2:
            strategy = meta.get('generation_strategy', {})
            st.metric("Query Types", 
                     len(strategy.get('distribution', {})),
                     delta=None)
        
        with col3:
            analysis = meta.get('original_query_analysis', {})
            st.metric("Query Complexity", 
                     analysis.get('complexity', 'N/A').upper(),
                     delta=None)

def display_query_distribution():
    """Display query type distribution"""
    if hasattr(st.session_state, 'last_results'):
        queries = st.session_state.last_results
        
        # Count by type
        type_counts = {}
        for q in queries:
            qtype = q.get('type', 'unknown')
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {"Type": QUERY_TYPES.get(k, {}).get('name', k), 
             "Count": v,
             "Icon": QUERY_TYPES.get(k, {}).get('icon', '❓')}
            for k, v in type_counts.items()
        ])
        
        if not df.empty:
            st.bar_chart(df.set_index('Type')['Count'])

# Main UI
tabs = st.tabs(["🚀 Generate", "📊 Results", "🔍 Analysis", "📚 History", "⚙️ Advanced"])

with tabs[0]:  # Generate Tab
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Query Configuration")
        st.info(f"**Current Query:** {user_query}")
        st.info(f"**Mode:** {mode}")
        
        # Context Builder
        if include_context:
            with st.expander("📝 Add Context (Optional)"):
                context = {
                    "user_location": st.text_input("Location", "Seattle, WA"),
                    "budget": st.select_slider("Budget Range", 
                                             options=["$", "$$", "$$$", "$$$$"],
                                             value="$$$"),
                    "timeframe": st.selectbox("Timeframe", 
                                            ["Immediate", "3 months", "6 months", "1 year"]),
                    "previous_searches": st.text_area("Previous Related Searches", 
                                                    placeholder="Enter previous searches...")
                }
        else:
            context = None
    
    with col2:
        st.markdown("### 🎮 Actions")
        
        if st.button("🚀 Generate Fan-Out", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("⚠️ Please enter a query.")
            else:
                # Clear previous results
                st.session_state.selected_queries = []
                
                # Generate queries
                results = generate_enhanced_fanout(user_query, mode, context)
                
                if results:
                    st.session_state.last_results = results
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": user_query,
                        "mode": mode,
                        "results_count": len(results)
                    })
                    
                    # Save session if enabled
                    if save_session:
                        st.session_state.generation_sessions.append({
                            "id": generate_query_id(user_query + str(time.time())),
                            "timestamp": datetime.now().isoformat(),
                            "original_query": user_query,
                            "mode": mode,
                            "context": context,
                            "results": results,
                            "metadata": st.session_state.last_generation_metadata
                        })
                    
                    st.success(f"✅ Generated {len(results)} intelligent queries!")
                    st.balloons()

with tabs[1]:  # Results Tab
    if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
        st.markdown("### 📋 Generated Queries")
        
        # Display metadata
        display_generation_metadata()
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.multiselect(
                "Filter by Type",
                options=list(QUERY_TYPES.keys()),
                format_func=lambda x: f"{QUERY_TYPES[x]['icon']} {QUERY_TYPES[x]['name']}"
            )
        with col2:
            filter_priority = st.multiselect(
                "Filter by Priority",
                options=["high", "medium", "low"]
            )
        with col3:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
        
        # Apply filters
        filtered_results = st.session_state.last_results
        if filter_type:
            filtered_results = [q for q in filtered_results if q.get('type') in filter_type]
        if filter_priority:
            filtered_results = [q for q in filtered_results if q.get('priority') in filter_priority]
        if min_confidence > 0:
            filtered_results = [q for q in filtered_results if q.get('confidence', 0) >= min_confidence]
        
        # Display queries in an interactive format
        st.markdown(f"**Showing {len(filtered_results)} of {len(st.session_state.last_results)} queries**")
        
        # Group by type
        grouped_queries = {}
        for q in filtered_results:
            qtype = q.get('type', 'unknown')
            if qtype not in grouped_queries:
                grouped_queries[qtype] = []
            grouped_queries[qtype].append(q)
        
        # Display grouped queries
        for qtype, queries in grouped_queries.items():
            type_info = QUERY_TYPES.get(qtype, {"name": qtype, "icon": "❓", "description": ""})
            
            with st.expander(f"{type_info['icon']} {type_info['name']} ({len(queries)} queries)", 
                           expanded=True):
                st.markdown(f"*{type_info['description']}*")
                
                for idx, q in enumerate(queries):
                    col1, col2, col3 = st.columns([0.7, 0.2, 0.1])
                    
                    with col1:
                        # Checkbox for selection
                        selected = st.checkbox(
                            q['query'],
                            key=f"query_{q.get('id', idx)}",
                            value=q.get('id') in st.session_state.selected_queries
                        )
                        
                        if selected and q.get('id') not in st.session_state.selected_queries:
                            st.session_state.selected_queries.append(q.get('id'))
                        elif not selected and q.get('id') in st.session_state.selected_queries:
                            st.session_state.selected_queries.remove(q.get('id'))
                        
                        # Show details
                        with st.container():
                            st.caption(f"💡 **Intent:** {q.get('user_intent', 'N/A')}")
                            st.caption(f"📝 **Reasoning:** {q.get('reasoning', 'N/A')}")
                            
                            # Additional metadata
                            metadata_cols = st.columns(4)
                            if q.get('entities'):
                                metadata_cols[0].caption(f"🏷️ Entities: {', '.join(q['entities'][:3])}")
                            if q.get('temporal_relevance'):
                                metadata_cols[1].caption(f"⏰ {q['temporal_relevance']}")
                            if q.get('geographic_relevance'):
                                metadata_cols[2].caption(f"📍 {q['geographic_relevance']}")
                    
                    with col2:
                        # Priority badge
                        priority = q.get('priority', 'medium')
                        priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        st.markdown(f"{priority_colors.get(priority, '⚪')} **{priority.upper()}**")
                    
                    with col3:
                        # Confidence score
                        confidence = q.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.0%}", label_visibility="collapsed")
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Selected", disabled=len(st.session_state.selected_queries) == 0):
                selected_data = [q for q in st.session_state.last_results 
                               if q.get('id') in st.session_state.selected_queries]
                csv = pd.DataFrame(selected_data).to_csv(index=False)
                st.download_button(
                    "Download Selected Queries",
                    data=csv.encode('utf-8'),
                    file_name=f"selected_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export All"):
                df = pd.DataFrame(st.session_state.last_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download All Queries",
                    data=csv.encode('utf-8'),
                    file_name=f"all_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔄 Regenerate"):
                st.experimental_rerun()
    
    else:
        st.info("👆 Generate queries first to see results here.")

with tabs[2]:  # Analysis Tab
    if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
        st.markdown("### 📊 Query Analysis")
        
        # Distribution visualization
        st.markdown("#### Query Type Distribution")
        display_query_distribution()
        
        # Confidence distribution
        st.markdown("#### Confidence Score Distribution")
        confidence_scores = [q.get('confidence', 0) for q in st.session_state.last_results]
        df_confidence = pd.DataFrame({
            'Confidence': confidence_scores,
            'Count': [1] * len(confidence_scores)
        })
        st.bar_chart(df_confidence.groupby(pd.cut(df_confidence['Confidence'], 
                                                  bins=[0, 0.5, 0.7, 0.85, 1.0],
                                                  labels=['Low', 'Medium', 'High', 'Very High'])).count())
        
        # Entity analysis
        st.markdown("#### Entity Coverage")
        all_entities = []
        for q in st.session_state.last_results:
            all_entities.extend(q.get('entities', []))
        
        entity_counts = pd.Series(all_entities).value_counts().head(10)
        if not entity_counts.empty:
            st.bar_chart(entity_counts)
        
        # Query relationships
        if hasattr(st.session_state, 'query_relationships') and st.session_state.query_relationships:
            st.markdown("#### Query Relationships")
            st.json(st.session_state.query_relationships)
    
    else:
        st.info("👆 Generate queries first to see analysis here.")

with tabs[3]:  # History Tab
    st.markdown("### 📚 Query Generation History")
    
    if st.session_state.generation_sessions:
        # Session selector
        session_options = [
            f"{s['timestamp']} - {s['original_query'][:50]}... ({len(s['results'])} queries)"
            for s in st.session_state.generation_sessions
        ]
        
        selected_session_idx = st.selectbox(
            "Select a session to review",
            range(len(session_options)),
            format_func=lambda x: session_options[x]
        )
        
        if selected_session_idx is not None:
            session = st.session_state.generation_sessions[selected_session_idx]
            
            # Display session details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original Query:** {session['original_query']}")
                st.markdown(f"**Mode:** {session['mode']}")
                st.markdown(f"**Timestamp:** {session['timestamp']}")
            
            with col2:
                st.markdown(f"**Total Queries:** {len(session['results'])}")
                if session.get('context'):
                    st.markdown("**Context:** ✅ Included")
                else:
                    st.markdown("**Context:** ❌ Not included")
            
            # Load session button
            if st.button("📂 Load This Session"):
                st.session_state.last_results = session['results']
                st.session_state.last_generation_metadata = session.get('metadata', {})
                st.success("✅ Session loaded! Go to Results tab.")
            
            # Display session queries
            with st.expander("View Queries from This Session"):
                df = pd.DataFrame(session['results'])
                st.dataframe(df[['query', 'type', 'priority', 'confidence']], 
                           use_container_width=True)
    else:
        st.info("No generation history yet. Start generating queries!")

with tabs[4]:  # Advanced Tab
    st.markdown("### ⚙️ Advanced Features")
    
    # Query Chaining
    st.markdown("#### 🔗 Query Chaining")
    st.info("Select queries from your results to generate follow-up queries based on them.")
    
    if st.session_state.selected_queries and hasattr(st.session_state, 'last_results'):
        selected_for_chaining = [q for q in st.session_state.last_results 
                               if q.get('id') in st.session_state.selected_queries]
        
        st.markdown(f"**{len(selected_for_chaining)} queries selected for chaining**")
        
        chain_mode = st.selectbox(
            "Chaining Strategy",
            ["Deep Dive", "Lateral Exploration", "Contrast & Compare", "Synthesis"]
        )
        
        if st.button("🔗 Generate Chained Queries"):
            # Implement query chaining logic
            st.info("Query chaining feature coming soon!")
    
    # Batch Processing
    st.markdown("#### 📦 Batch Query Processing")
    uploaded_file = st.file_uploader(
        "Upload CSV with queries",
        type=['csv'],
        help="CSV should have a 'query' column"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"Found {len(df)} queries to process")
        
        if st.button("🚀 Process Batch"):
            progress_bar = st.progress(0)
            results_container = st.container()
            
            batch_results = []
            for idx, row in df.iterrows():
                if 'query' in row:
                    # Process each query
                    results = generate_enhanced_fanout(row['query'], mode)
                    if results:
                        batch_results.extend(results)
                    
                    progress_bar.progress((idx + 1) / len(df))
            
            results_container.success(f"✅ Generated {len(batch_results)} total queries from {len(df)} inputs")
            
            # Export batch results
            batch_df = pd.DataFrame(batch_results)
            csv = batch_df.to_csv(index=False)
            st.download_button(
                "📥 Download Batch Results",
                data=csv.encode('utf-8'),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # API Integration Guide
    st.markdown("#### 🔌 API Integration")
    with st.expander("View API Integration Code"):
        st.code("""
# Example: Integrate Qforia Pro into your application

import requests
import json

def generate_queries(primary_query, api_key, mode="AI Mode (Complex)"):
    endpoint = "https://your-qforia-instance.com/api/generate"
    
    payload = {
        "query": primary_query,
        "mode": mode,
        "api_key": api_key,
        "options": {
            "include_confidence": True,
            "include_relationships": True,
            "min_confidence": 0.7
        }
    }
    
    response = requests.post(endpoint, json=payload)
    return response.json()

# Usage
results = generate_queries(
    "best practices for machine learning in production",
    "your-api-key"
)

for query in results['queries']:
    print(f"{query['confidence']:.0%} - {query['query']}")
""", language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Qforia Pro v2.0 | Powered by Advanced Query Intelligence</p>
    <p>Based on state-of-the-art search methodologies</p>
</div>
""", unsafe_allow_html=True)
