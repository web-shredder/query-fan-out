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

st.title("Qforia 2.0: Advanced LLM Query Intelligence System")
st.markdown("*Powered by Multi-Model Query Generation & Contextual Understanding*. Based on "Qforia by Mike King.")

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
        "gemini-1.5-pro-latest"
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    
    # Advanced Settings
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)

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
        "AI Overview (Simple)": {"min": 12, "max": 20, "depth": "basic"},
        "AI Mode (Complex)": {"min": 20, "max": 35, "depth": "comprehensive"},
        "Research Mode (Deep)": {"min": 35, "max": 50, "depth": "exhaustive"},
        "Comparative Analysis": {"min": 15, "max": 25, "depth": "comparative"},
        "Multi-Perspective": {"min": 25, "max": 40, "depth": "multi-angle"}
    }
    
    config = mode_configs.get(mode, mode_configs["AI Mode (Complex)"])
    
    context_str = ""
    if context:
        context_str = f"\nUser Context: {json.dumps(context)}\n"
    
    return f"""You are an advanced AI search query generation system. Generate diverse search queries based on the input.

Original Query: "{q}"
Mode: {mode}
{context_str}

Generate between {config['min']} and {config['max']} queries using these types:
1. Reformulations - Alternative phrasings
2. Related Queries - Adjacent topics
3. Implicit Queries - Hidden questions
4. Comparative Queries - Comparisons
5. Entity Expansions - Entity focus
6. Personalized Queries - Context-aware

Return ONLY a valid JSON object with this structure:
{{
  "expanded_queries": [
    {{
      "query": "generated query text",
      "type": "reformulation|related|implicit|comparative|entity_expansion|personalized",
      "user_intent": "what the user wants to know",
      "reasoning": "why this query is useful",
      "priority": "high|medium|low"
    }}
  ]
}}

Important: Return ONLY the JSON object, no other text."""

def clean_json_response(text):
    """Clean and extract JSON from response"""
    # Remove any markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    # Find JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0)
    
    return text.strip()

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
            
            # Clean the response
            json_text = clean_json_response(response.text)
            
            # Debug output
            with st.expander("🔍 Debug: Raw Response", expanded=False):
                st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)
                st.code(json_text[:500] + "..." if len(json_text) > 500 else json_text)
            
            # Parse JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                st.error(f"JSON Parse Error: {str(e)}")
                st.text("Attempting to fix common issues...")
                
                # Try to fix common JSON issues
                json_text = json_text.replace("'", '"')  # Replace single quotes
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)
                
                try:
                    data = json.loads(json_text)
                except:
                    # If still failing, create a simple response
                    st.warning("Using fallback query generation...")
                    return generate_fallback_queries(query, mode)
            
            # Extract queries
            queries = data.get("expanded_queries", [])
            
            # Add IDs and confidence scores
            for q in queries:
                q["id"] = generate_query_id(q.get("query", ""))
                q["confidence"] = calculate_confidence_score(
                    q.get("query", ""), 
                    query, 
                    q.get("type", "related")
                )
            
            return queries
            
    except Exception as e:
        st.error(f"🔴 Error: {str(e)}")
        st.warning("Using fallback query generation...")
        return generate_fallback_queries(query, mode)

def generate_fallback_queries(query, mode):
    """Fallback query generation when API fails"""
    base_queries = []
    
    # Simple transformations
    words = query.lower().split()
    
    # Type 1: Reformulations
    base_queries.append({
        "id": generate_query_id(f"how to {query}"),
        "query": f"how to {query}",
        "type": "reformulation",
        "user_intent": "Understanding the process",
        "reasoning": "User might want step-by-step guidance",
        "priority": "high",
        "confidence": 0.8
    })
    
    base_queries.append({
        "id": generate_query_id(f"best {query}"),
        "query": f"best {query}",
        "type": "reformulation",
        "user_intent": "Finding optimal options",
        "reasoning": "User wants recommendations",
        "priority": "high",
        "confidence": 0.85
    })
    
    # Type 2: Related
    base_queries.append({
        "id": generate_query_id(f"{query} reviews"),
        "query": f"{query} reviews",
        "type": "related",
        "user_intent": "Reading user experiences",
        "reasoning": "Reviews provide real-world insights",
        "priority": "medium",
        "confidence": 0.75
    })
    
    base_queries.append({
        "id": generate_query_id(f"{query} comparison"),
        "query": f"{query} comparison",
        "type": "comparative",
        "user_intent": "Comparing options",
        "reasoning": "User wants to evaluate alternatives",
        "priority": "high",
        "confidence": 0.8
    })
    
    # Type 3: Implicit
    base_queries.append({
        "id": generate_query_id(f"{query} cost"),
        "query": f"{query} cost",
        "type": "implicit",
        "user_intent": "Understanding pricing",
        "reasoning": "Cost is often an implicit concern",
        "priority": "medium",
        "confidence": 0.7
    })
    
    # Add more based on mode
    if mode in ["AI Mode (Complex)", "Research Mode (Deep)"]:
        base_queries.extend([
            {
                "id": generate_query_id(f"{query} alternatives"),
                "query": f"{query} alternatives",
                "type": "comparative",
                "user_intent": "Finding alternatives",
                "reasoning": "User might want other options",
                "priority": "medium",
                "confidence": 0.75
            },
            {
                "id": generate_query_id(f"{query} guide 2024"),
                "query": f"{query} guide 2024",
                "type": "temporal",
                "user_intent": "Current information",
                "reasoning": "User wants up-to-date information",
                "priority": "high",
                "confidence": 0.8
            }
        ])
    
    return base_queries

def display_generation_metadata():
    """Display generation metadata in an attractive format"""
    if hasattr(st.session_state, 'last_results'):
        results = st.session_state.last_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", len(results))
        
        with col2:
            unique_types = len(set(q.get('type', 'unknown') for q in results))
            st.metric("Query Types", unique_types)
        
        with col3:
            avg_confidence = sum(q.get('confidence', 0) for q in results) / len(results) if results else 0
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")

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
tabs = st.tabs(["🚀 Generate", "📊 Results", "🔍 Analysis", "📚 History"])

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
                            "results": results
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
                            key=f"query_{q.get('id', idx)}_{idx}",
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
                    
                    with col2:
                        # Priority badge
                        priority = q.get('priority', 'medium')
                        priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        st.markdown(f"{priority_colors.get(priority, '⚪')} **{priority.upper()}**")
                    
                    with col3:
                        # Confidence score
                        confidence = q.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.0%}", label_visibility="collapsed")
        
        # Export buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export All Results"):
                df = pd.DataFrame(st.session_state.last_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv.encode('utf-8'),
                    file_name=f"qforia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
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
        
        # Create bins for confidence scores
        bins = [0, 0.5, 0.7, 0.85, 1.0]
        labels = ['Low (0-50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (85-100%)']
        
        # Count queries in each bin
        confidence_dist = pd.cut(confidence_scores, bins=bins, labels=labels).value_counts()
        st.bar_chart(confidence_dist)
        
        # Priority distribution
        st.markdown("#### Priority Distribution")
        priority_counts = {}
        for q in st.session_state.last_results:
            priority = q.get('priority', 'medium')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        priority_df = pd.DataFrame.from_dict(priority_counts, orient='index', columns=['Count'])
        st.bar_chart(priority_df)
    
    else:
        st.info("👆 Generate queries first to see analysis here.")

with tabs[3]:  # History Tab
    st.markdown("### 📚 Query Generation History")
    
    if st.session_state.generation_sessions:
        # Session selector
        session_options = [
            f"{s['timestamp']} - {s['original_query'][:50]}... ({len(s['results'])} queries)"
            for s in reversed(st.session_state.generation_sessions)
        ]
        
        if session_options:
            selected_session_idx = st.selectbox(
                "Select a session to review",
                range(len(session_options)),
                format_func=lambda x: session_options[x]
            )
            
            if selected_session_idx is not None:
                # Get session (accounting for reversal)
                actual_idx = len(st.session_state.generation_sessions) - 1 - selected_session_idx
                session = st.session_state.generation_sessions[actual_idx]
                
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
                    st.success("✅ Session loaded! Go to Results tab.")
                
                # Display session queries
                with st.expander("View Queries from This Session"):
                    df = pd.DataFrame(session['results'])
                    if not df.empty:
                        display_cols = ['query', 'type', 'priority']
                        if 'confidence' in df.columns:
                            display_cols.append('confidence')
                        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No generation history yet. Start generating queries!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Qforia Pro v2.0 | Powered by Advanced Query Intelligence</p>
</div>
""", unsafe_allow_html=True)
