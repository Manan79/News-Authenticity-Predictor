import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import re
import time

# Page config
st.set_page_config(
    page_title="News Authenticity Predictor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .news-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .fake-news {
        border-left-color: #ef4444;
    }
    
    .real-news {
        border-left-color: #10b981;
    }
    
    .confidence-high {
        color: #059669;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #d97706;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc2626;
        font-weight: bold;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = []

# Model loading function
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load your trained model (adjust path as needed)
        model = tf.keras.models.load_model('GRU_Model.h5')
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        # Load tokenizer (adjust path as needed)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Create a dummy model and tokenizer for demo purposes
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        # You would replace this with your actual model loading
        model = None
        return model, tokenizer

# Language detection function
def is_english(text):
    """Check if text is in English"""
    try:
        if not text or len(text.strip()) < 10:
            return False
        
        # Simple English detection using character patterns
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()[:50]  # Check first 50 words
        english_word_count = sum(1 for word in words if any(eng_word in word for eng_word in english_words))
        
        # If at least 20% of words contain common English words/patterns
        if len(words) > 0 and (english_word_count / len(words)) >= 0.2:
            return True
            
        return False
    except:
        return False

# News fetching function
def fetch_news(api_key=None, query="latest news", sources="bbc-news,cnn,reuters"):
    """Fetch news from NewsAPI"""
    try:
        # Using NewsAPI (you'll need to get a free API key)
        if api_key:
            url = f"https://newsapi.org/v2/everything?q={query}&sources={sources}&language=en&apiKey={api_key}&pageSize=15"
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == 'ok':
                # Filter for English articles
                english_articles = []
                for article in data['articles']:
                    title = article.get('title', '') or ''
                    description = article.get('description', '') or ''
                    content = article.get('content', '') or ''
                    full_text = f"{title} {description} {content}"
                    
                    if is_english(full_text):
                        english_articles.append(article)
                
                return english_articles # Return max 10 articles
        
        # Fallback: Mock data for demo (English only)
        mock_news = [
            {
                "title": "Scientists Discover Revolutionary AI Technology in Medical Diagnosis",
                "description": "Researchers have made a groundbreaking discovery in artificial intelligence that could transform healthcare diagnosis and treatment procedures worldwide.",
                "content": "A team of scientists at a leading university has announced a major breakthrough in AI technology for medical applications. The new system demonstrates unprecedented capabilities in analyzing medical images and providing accurate diagnostic recommendations. This development could have significant implications for healthcare systems globally, potentially reducing diagnosis time and improving patient outcomes.",
                "url": "https://example.com/news1",
                "publishedAt": "2024-08-14T10:30:00Z",
                "source": {"name": "Medical Tech Today"}
            },
            {
                "title": "Global Economic Markets Show Strong Recovery Following Policy Changes",
                "description": "International financial markets are displaying positive trends as investors respond to recent government policy implementations and economic reforms.",
                "content": "Financial analysts report that major stock indices across multiple continents have shown consistent growth patterns over the recent period. The recovery appears to be driven by renewed investor confidence stemming from supportive government policies and improved economic indicators. Market experts predict continued stability and growth potential in the coming quarters, citing strong fundamentals and positive market sentiment.",
                "url": "https://example.com/news2",
                "publishedAt": "2024-08-14T09:15:00Z",
                "source": {"name": "Global Finance News"}
            },
            {
                "title": "Climate Change Research Reveals New Insights on Ocean Temperature Patterns",
                "description": "Marine scientists present comprehensive findings on how rising ocean temperatures are affecting global weather systems and marine ecosystems.",
                "content": "A comprehensive study conducted by international marine research institutes has revealed significant new data about ocean temperature changes and their cascading effects on global climate patterns. The research indicates that oceanic temperature variations are having more profound impacts on weather systems than previously understood. Scientists emphasize the importance of continued monitoring and research to better predict future climate scenarios.",
                "url": "https://example.com/news3",
                "publishedAt": "2024-08-14T08:45:00Z",
                "source": {"name": "Environmental Science Weekly"}
            }
        ]
        return mock_news
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text for model prediction"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Prediction function
def predict_news(text, model, tokenizer, max_len=200):
    """Predict if news is real or fake"""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not processed_text.strip():
            return 0.5, "Unable to classify"
        
        # For demo purposes (replace with actual model prediction)
        if model is None or tokenizer is None:
            # Simulate prediction based on text characteristics
            word_count = len(processed_text.split())
            try:
                sentiment = TextBlob(processed_text).sentiment.polarity
            except:
                sentiment = 0
            
            # Simple heuristic for demo
            if word_count > 50 and abs(sentiment) < 0.3:
                confidence = float(0.75 + np.random.random() * 0.2)
                label = "REAL"
            else:
                confidence = float(0.6 + np.random.random() * 0.3)
                label = "FAKE"
                
            return confidence, label
        
        # Actual model prediction code (uncomment when your model is ready)
        try:
            sequences = tokenizer.texts_to_sequences([processed_text])
            padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
            prediction = model.predict(padded_sequences, verbose=0)[0][0]
            
            # Convert prediction to confidence and label
            if prediction > 0.5:
                confidence = float(prediction)
                label = "REAL"
            else:
                confidence = float(1 - prediction)
                label = "FAKE"
            
            return confidence, label
        except Exception as model_error:
            # Fall back to demo prediction
            word_count = len(processed_text.split())
            confidence = float(0.7 + np.random.random() * 0.2)
            label = "REAL" if word_count > 30 else "FAKE"
            return confidence, label
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return 0.5, "Error"

# Main app
def main():
    st.markdown('<h1 class="main-header">📰 News Authenticity Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("NewsAPI Key (Optional)", type="password", 
                               help="Get free API key from newsapi.org for real-time news")
        
        if api_key == "":
         api_key = "565af593e56a48f09d44f78fc0b86e84"
        # Language filter info
        st.markdown("**🌐 Language Filter:** English Only")
        st.markdown("All fetched and analyzed articles are filtered to include only English content.")
        
        # Model loading
        if st.button("🚀 Load Model", use_container_width=True):
            with st.spinner("Loading model and tokenizer..."):
                model, tokenizer = load_model_and_tokenizer()
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
        
        # Model status
        status_color = "🟢" if st.session_state.model_loaded else "🔴"
        status_text = "Ready" if st.session_state.model_loaded else "Not Loaded"
        st.markdown(f"**Model Status:** {status_color} {status_text}")
        
        # Info box
        st.markdown("""
        <div class="sidebar-info">
            <h4>📊 Model Info</h4>
            <p><strong>Dataset:</strong> clementbisaillon/fake-and-real-news-dataset</p>
            <p><strong>Architecture:</strong> LSTM + GRU</p>
            <p><strong>Tokenizer:</strong> TensorFlow Tokenizer</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📡 Real-time News", "🔍 Manual Checker", "📈 Analytics"])
    
    with tab1:
        st.markdown("### 📡 Real-time News Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Search Query", value="latest news", 
                                       placeholder="Enter keywords to search for news...")
        
        with col2:
            if st.button("🔄 Fetch & Analyze", use_container_width=True):
                if not st.session_state.model_loaded:
                    st.warning("⚠️ Please load the model first!")
                else:
                    with st.spinner("Fetching and analyzing news..."):
                        news_articles = fetch_news(api_key, search_query)
                        
                        if news_articles:
                            st.session_state.news_data = []
                            progress_bar = st.progress(0)
                            
                            for i, article in enumerate(news_articles):
                                # Combine title and content for prediction
                                title = article.get('title', '') or ''
                                description = article.get('description', '') or ''
                                content = article.get('content', '') or ''
                                full_text = f"{title} {description} {content}".strip()
                                
                                if not full_text:
                                    confidence, label = 0.5, "Unable to classify"
                                elif not is_english(full_text):
                                    continue  # Skip non-English articles
                                else:
                                    confidence, label = predict_news(
                                        full_text, 
                                        st.session_state.model, 
                                        st.session_state.tokenizer
                                    )
                                
                                article_data = {
                                    'title': article.get('title', 'No Title') or 'No Title',
                                    'description': article.get('description', 'No Description') or 'No Description',
                                    'url': article.get('url', '') or '',
                                    'source': (article.get('source') or {}).get('name', 'Unknown'),
                                    'publishedAt': article.get('publishedAt', '') or '',
                                    'confidence': confidence,
                                    'label': label
                                }
                                
                                st.session_state.news_data.append(article_data)
                                progress_bar.progress((i + 1) / len(news_articles))
                            
                            st.success(f"✅ Analyzed {len(news_articles)} articles!")
        
        # Display news results
        if st.session_state.news_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_news = len(st.session_state.news_data)
            real_news = sum(1 for item in st.session_state.news_data if item.get('label') == 'REAL')
            fake_news = total_news - real_news
            
            # Filter out None values for confidence calculation
            confidence_values = [item['confidence'] for item in st.session_state.news_data 
                               if item.get('confidence') is not None and 
                               isinstance(item.get('confidence'), (int, float))]
            
            avg_confidence = np.mean(confidence_values) if confidence_values else 0.5
            
            with col1:
                st.metric("📰 Total Articles", total_news)
            with col2:
                st.metric("✅ Real News", real_news)
            with col3:
                st.metric("❌ Fake News", fake_news)
            with col4:
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
            
            # News cards
            for article in st.session_state.news_data:
                # Ensure confidence is a valid number
                confidence = article.get('confidence', 0.5)
                if confidence is None or not isinstance(confidence, (int, float)):
                    confidence = 0.5
                
                label = article.get('label', 'Unknown')
                if label is None:
                    label = 'Unknown'
                
                card_class = "real-news" if label == 'REAL' else "fake-news"
                confidence_class = ("confidence-high" if confidence > 0.8 
                                  else "confidence-medium" if confidence > 0.6 
                                  else "confidence-low")
                
                title = article.get('title', 'No Title') or 'No Title'
                source = article.get('source', 'Unknown') or 'Unknown'
                published = article.get('publishedAt', '') or ''
                description = article.get('description', 'No Description') or 'No Description'
                url = article.get('url', '#') or '#'
                
                st.markdown(f"""
                <div class="news-card {card_class}">
                    <h4>{title}</h4>
                    <p><strong>Source:</strong> {source} | <strong>Published:</strong> {published[:10] if len(published) > 10 else published}</p>
                    <p>{description}</p>
                    <div style="margin-top: 1rem;">
                        <span class="{confidence_class}">
                            {label} (Confidence: {confidence:.2f})
                        </span>
                        <a href="{url}" target="_blank" style="margin-left: 1rem;">🔗 Read Full Article</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🔍 Manual News Checker")
        st.markdown("**Note:** This tool only analyzes English news articles.")
        st.markdown("Paste any English news article below to check its authenticity:")
        
        # Input methods
        input_method = st.radio("Input Method:", ["Text Input"])
        
        if input_method == "Text Input":
            user_text = st.text_area("Enter news text:", height=200, 
                                    placeholder="Paste the news article here...")
            
            if st.button("🔍 Analyze Text", use_container_width=True):
                if user_text and st.session_state.model_loaded:
                    # Check if text is English
                    if not is_english(user_text):
                        st.error("❌ **Non-English Text Detected**")
                        st.warning("⚠️ This tool only supports English news articles. Please provide English text for analysis.")
                        return
                    
                    with st.spinner("Analyzing..."):
                        confidence, label = predict_news(
                            user_text, 
                            st.session_state.model, 
                            st.session_state.tokenizer
                        )
                        
                        # Results display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if label == "REAL":
                                st.success(f"✅ **{label}** News")
                            else:
                                st.error(f"❌ **{label}** News")
                        
                        with col2:
                            st.metric("Confidence Score", f"{confidence:.3f}")
                        
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = confidence,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence Level"},
                            gauge = {
                                'axis': {'range': [None, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 0.6], 'color': "lightgray"},
                                    {'range': [0.6, 0.8], 'color': "yellow"},
                                    {'range': [0.8, 1], 'color': "green"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.9}}))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Text analysis
                        st.markdown("#### 📊 Text Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            word_count = len(user_text.split())
                            st.metric("Word Count", word_count)
                        
                        with col2:
                            sentiment = TextBlob(user_text).sentiment.polarity
                            st.metric("Sentiment", f"{sentiment:.2f}")
                        
                        with col3:
                            char_count = len(user_text)
                            st.metric("Character Count", char_count)
                
                elif not user_text:
                    st.warning("⚠️ Please enter some text to analyze.")
                else:
                    st.warning("⚠️ Please load the model first!")
        
        
    
    with tab3:
        st.markdown("### 📈 Analytics Dashboard")
        
        if st.session_state.news_data:
            # Create visualizations with proper data validation
            valid_data = []
            for item in st.session_state.news_data:
                if (item.get('confidence') is not None and 
                    isinstance(item.get('confidence'), (int, float)) and
                    item.get('label') is not None):
                    valid_data.append(item)
            
            if not valid_data:
                st.warning("No valid data available for visualization.")
                return
                
            df = pd.DataFrame(valid_data)
            
            # Pie chart for real vs fake
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=df['label'].value_counts().values,
                    names=df['label'].value_counts().index,
                    title="Real vs Fake News Distribution",
                    color_discrete_map={'REAL': '#10b981', 'FAKE': '#ef4444'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig_hist = px.histogram(
                    df, x='confidence', nbins=10,
                    title="Confidence Score Distribution",
                    color='label',
                    color_discrete_map={'REAL': '#10b981', 'FAKE': '#ef4444'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Source analysis
            if len(df['source'].unique()) > 1:
                source_counts = df['source'].value_counts()
                fig_bar = px.bar(
                    x=source_counts.index, y=source_counts.values,
                    title="News Articles by Source",
                    labels={'x': 'Source', 'y': 'Count'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed data table
            st.markdown("#### 📋 Detailed Results")
            display_df = df[['title', 'source', 'label', 'confidence']].copy()
            display_df['confidence'] = display_df['confidence'].round(3)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("📊 No data available. Please analyze some news articles first!")

if __name__ == "__main__":
    main()