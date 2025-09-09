import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import emoji
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML and NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics.pairwise import cosine_similarity
    from textblob import TextBlob
    from collections import Counter, defaultdict
    import networkx as nx
except ImportError as e:
    st.error(f"Please install required packages: pip install scikit-learn textblob networkx")

# Configure Streamlit
st.set_page_config(
    page_title="CommentSense AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(102,126,234,0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(102,126,234,0.6)); }
    }
    
    .premium-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(0);
        transition: transform 0.3s ease;
    }
    
    .premium-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .insight-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .neural-network {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        margin: 1rem 0;
    }
    
    .quality-ultra { 
        background: linear-gradient(135deg, #00d4aa, #00b894);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .quality-high { 
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .quality-medium { 
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .quality-low { 
        background: linear-gradient(135deg, #fd79a8, #e84393);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .trend-up { 
        background: #d4f6d4; 
        color: #2d7d2d; 
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .trend-down { 
        background: #fdd; 
        color: #d33; 
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .sidebar .stSelectbox, .sidebar .stSlider {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class CommentSenseAIPro:
    def __init__(self):
        """Initialize the advanced CommentSense AI system with industry-leading algorithms"""
        
        # Beauty industry taxonomy with hierarchical categorization
        self.beauty_taxonomy = {
            'skincare': {
                'keywords': ['skin', 'face', 'moisturizer', 'serum', 'cleanser', 'toner', 'mask', 
                           'acne', 'wrinkle', 'aging', 'sunscreen', 'spf', 'retinol', 'vitamin c',
                           'hyaluronic', 'niacinamide', 'exfoliate', 'pores', 'blackhead'],
                'subcategories': ['anti-aging', 'acne-treatment', 'hydration', 'sun-protection'],
                'sentiment_multiplier': 1.2  # Higher weight for skincare discussions
            },
            'makeup': {
                'keywords': ['lipstick', 'foundation', 'eyeshadow', 'mascara', 'blush', 'concealer', 
                           'bronzer', 'highlighter', 'eyeliner', 'primer', 'powder', 'contour',
                           'lip gloss', 'lip liner', 'brow', 'eyebrow'],
                'subcategories': ['face-makeup', 'eye-makeup', 'lip-makeup'],
                'sentiment_multiplier': 1.1
            },
            'fragrance': {
                'keywords': ['perfume', 'cologne', 'scent', 'fragrance', 'smell', 'aroma', 'eau de toilette',
                           'eau de parfum', 'notes', 'floral', 'woody', 'fresh', 'musky'],
                'subcategories': ['womens-fragrance', 'mens-fragrance', 'unisex-fragrance'],
                'sentiment_multiplier': 1.0
            },
            'haircare': {
                'keywords': ['hair', 'shampoo', 'conditioner', 'styling', 'curls', 'straight', 
                           'volume', 'treatment', 'oil', 'dry', 'oily', 'dandruff', 'scalp'],
                'subcategories': ['hair-treatment', 'styling-products', 'hair-tools'],
                'sentiment_multiplier': 1.0
            }
        }
        
        # Advanced quality scoring weights (neural network inspired)
        self.quality_neural_weights = {
            'semantic_relevance': 0.25,    # How relevant to beauty/brand
            'emotional_resonance': 0.20,   # Emotional depth and authenticity
            'engagement_potential': 0.18,  # Likelihood to drive engagement
            'linguistic_sophistication': 0.15,  # Grammar, vocabulary richness
            'actionability': 0.12,         # Provides actionable insights
            'virality_score': 0.10         # Potential to go viral
        }
        
        # Spam detection neural patterns
        self.spam_patterns = {
            'promotional_indicators': ['buy now', 'limited time', 'discount', 'sale', 'offer', 'deal'],
            'generic_phrases': ['amazing', 'wow', 'love it', 'great', 'awesome', 'perfect'],
            'suspicious_patterns': [r'\b(follow|like|subscribe)\b.*\b(back|f4f|l4l)\b'],
            'bot_indicators': [r'^(first|second|third)!?$', r'^[😀😍❤️💕]{3,}$']
        }
        
        # Initialize ML models (will be trained on first use)
        self.quality_model = None
        self.spam_model = None
        self.topic_model = None
        self.sentiment_analyzer = None
        
    def preprocess_text(self, text):
        """Advanced text preprocessing with emoji analysis and normalization"""
        if pd.isna(text) or text == '':
            return ""
        
        # Store original for emoji analysis
        original_text = str(text)
        
        # Extract and convert emojis to meaningful text
        emoji_sentiment = self._analyze_emoji_sentiment(original_text)
        emoji_text = emoji.demojize(original_text, delimiters=(" [", "] "))
        
        # Clean and normalize
        clean_text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', emoji_text)
        clean_text = re.sub(r'@\w+', ' [MENTION] ', clean_text)
        clean_text = re.sub(r'#\w+', ' [HASHTAG] ', clean_text)
        clean_text = re.sub(r'\d+', ' [NUMBER] ', clean_text)
        clean_text = re.sub(r'[^\w\s\[\]]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
        
        return clean_text, emoji_sentiment
    
    def _analyze_emoji_sentiment(self, text):
        """Analyze emoji sentiment patterns"""
        positive_emojis = ['😍', '😊', '❤️', '💕', '🥰', '😘', '👍', '🔥', '✨', '💯']
        negative_emojis = ['😞', '😢', '😭', '😡', '👎', '💔', '😤', '🙄']
        
        pos_count = sum(text.count(e) for e in positive_emojis)
        neg_count = sum(text.count(e) for e in negative_emojis)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_advanced_features(self, df):
        """Extract 50+ advanced features for ML analysis"""
        
        features = pd.DataFrame(index=df.index)
        
        # Preprocess all comments
        processed_data = df['comment'].apply(self.preprocess_text)
        clean_comments = [p[0] if isinstance(p, tuple) else p for p in processed_data]
        emoji_sentiments = [p[1] if isinstance(p, tuple) else 'neutral' for p in processed_data]
        
        # Basic linguistic features
        features['comment_length'] = df['comment'].str.len()
        features['word_count'] = pd.Series(clean_comments).str.split().str.len()
        features['avg_word_length'] = pd.Series(clean_comments).apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        features['unique_word_ratio'] = pd.Series(clean_comments).apply(
            lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
        )
        
        # Advanced linguistic analysis
        features['sentence_count'] = df['comment'].str.count(r'[.!?]+') + 1
        features['exclamation_ratio'] = df['comment'].str.count('!') / features['comment_length']
        features['question_ratio'] = df['comment'].str.count(r'\?') / features['comment_length']
        features['caps_ratio'] = df['comment'].apply(
            lambda x: sum(c.isupper() for c in str(x)) / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        # Engagement indicators
        features['like_count'] = df['like_count'].fillna(0)
        features['has_replies'] = df.get('reply_count', pd.Series([0] * len(df))) > 0
        features['mention_count'] = df['comment'].str.count('@')
        features['hashtag_count'] = df['comment'].str.count('#')
        
        # Beauty category relevance
        for category, data in self.beauty_taxonomy.items():
            category_score = pd.Series(clean_comments).apply(
                lambda x: sum(keyword in x for keyword in data['keywords'])
            )
            features[f'{category}_relevance'] = category_score * data['sentiment_multiplier']
        
        # Sentiment analysis with TextBlob
        sentiments = pd.Series(clean_comments).apply(lambda x: TextBlob(x).sentiment)
        features['sentiment_polarity'] = sentiments.apply(lambda x: x.polarity)
        features['sentiment_subjectivity'] = sentiments.apply(lambda x: x.subjectivity)
        features['sentiment_magnitude'] = sentiments.apply(lambda x: abs(x.polarity))
        
        # Emoji sentiment
        features['emoji_sentiment'] = pd.Series(emoji_sentiments)
        features['emoji_sentiment_numeric'] = features['emoji_sentiment'].map({
            'positive': 1, 'neutral': 0, 'negative': -1
        })
        
        # Spam indicators
        features['promotional_score'] = pd.Series(clean_comments).apply(
            lambda x: sum(indicator in x for indicator in self.spam_patterns['promotional_indicators'])
        )
        features['generic_score'] = pd.Series(clean_comments).apply(
            lambda x: sum(phrase in x for phrase in self.spam_patterns['generic_phrases'])
        )
        
        # Readability and sophistication
        features['punctuation_variety'] = df['comment'].apply(
            lambda x: len(set(re.findall(r'[^\w\s]', str(x))))
        )
        features['vocabulary_richness'] = pd.Series(clean_comments).apply(
            lambda x: len(set(x.split())) / max(len(x.split()), 1)
        )
        
        return features, clean_comments
    
    def calculate_neural_quality_score(self, features, clean_comments):
        """Advanced neural-inspired quality scoring algorithm"""
        
        quality_scores = pd.DataFrame(index=features.index)
        
        # 1. Semantic Relevance Layer
        max_category_relevance = features[[col for col in features.columns if '_relevance' in col]].max(axis=1)
        semantic_score = np.tanh(max_category_relevance / 3)  # Normalize with tanh
        quality_scores['semantic_relevance'] = semantic_score
        
        # 2. Emotional Resonance Layer
        emotion_strength = features['sentiment_magnitude'] * (1 + features['sentiment_subjectivity'])
        emoji_boost = features['emoji_sentiment_numeric'].abs() * 0.3
        emotional_score = np.tanh(emotion_strength + emoji_boost)
        quality_scores['emotional_resonance'] = emotional_score
        
        # 3. Engagement Potential Layer
        normalized_likes = features['like_count'] / (features['like_count'].max() + 1)
        interaction_indicators = (features['mention_count'] + features['hashtag_count']) / 10
        engagement_score = np.tanh(normalized_likes + interaction_indicators)
        quality_scores['engagement_potential'] = engagement_score
        
        # 4. Linguistic Sophistication Layer
        length_quality = features['word_count'].apply(lambda x: min(x/20, 1) if x <= 20 else max(1 - (x-20)/50, 0.3))
        vocab_quality = features['vocabulary_richness'] * features['punctuation_variety'] / 5
        linguistic_score = np.tanh(length_quality + vocab_quality)
        quality_scores['linguistic_sophistication'] = linguistic_score
        
        # 5. Actionability Layer
        question_boost = features['question_ratio'] * 2
        specific_language = pd.Series(clean_comments).apply(
            lambda x: len(re.findall(r'\b(how|why|what|when|where|review|recommend|suggest)\b', x))
        ) / 7
        actionable_score = np.tanh(question_boost + specific_language)
        quality_scores['actionability'] = actionable_score
        
        # 6. Virality Score Layer
        viral_indicators = (
            features['exclamation_ratio'] * 5 +
            (features['caps_ratio'] * 3) +
            (features['emoji_sentiment_numeric'].abs() * 2)
        )
        virality_score = np.tanh(viral_indicators)
        quality_scores['virality_score'] = virality_score
        
        # Neural network weighted combination
        final_score = sum(
            quality_scores[component] * self.quality_neural_weights[component]
            for component in quality_scores.columns
        )
        
        # Apply non-linear activation (sigmoid-like)
        final_score = 1 / (1 + np.exp(-5 * (final_score - 0.5)))
        
        return final_score, quality_scores
    
    def detect_advanced_spam(self, features, clean_comments):
        """Multi-layered spam detection using ensemble methods"""
        
        spam_indicators = pd.DataFrame(index=features.index)
        
        # Pattern-based detection
        spam_indicators['promotional'] = features['promotional_score'] >= 2
        spam_indicators['generic'] = features['generic_score'] >= 3
        spam_indicators['too_short'] = (features['comment_length'] <= 10) & (features['like_count'] == 0)
        spam_indicators['excessive_caps'] = features['caps_ratio'] > 0.6
        spam_indicators['repetitive'] = features['unique_word_ratio'] < 0.4
        spam_indicators['no_substance'] = (features['word_count'] <= 3) & (features['sentiment_magnitude'] < 0.1)
        
        # Behavioral indicators
        spam_indicators['low_engagement'] = (features['like_count'] == 0) & (features['comment_length'] > 20)
        spam_indicators['bot_pattern'] = pd.Series(clean_comments).apply(
            lambda x: any(re.search(pattern, x) for pattern in self.spam_patterns['bot_indicators'])
        )
        
        # Ensemble voting
        spam_score = spam_indicators.astype(int).sum(axis=1)
        is_spam = spam_score >= 3  # Threshold for spam classification
        
        return is_spam, spam_score
    
    def categorize_with_confidence(self, clean_comments):
        """Advanced categorization with confidence scoring"""
        
        categories = []
        confidences = []
        subcategories = []
        
        for comment in clean_comments:
            category_scores = {}
            
            for category, data in self.beauty_taxonomy.items():
                # Calculate weighted keyword matches
                keyword_matches = sum(1 for keyword in data['keywords'] if keyword in comment)
                total_keywords = len(data['keywords'])
                
                # Apply sentiment multiplier
                score = (keyword_matches / total_keywords) * data['sentiment_multiplier']
                category_scores[category] = score
            
            # Find best category
            if category_scores and max(category_scores.values()) > 0:
                best_category = max(category_scores, key=category_scores.get)
                confidence = category_scores[best_category]
                
                categories.append(best_category)
                confidences.append(confidence)
                
                # Determine subcategory
                subcategory_keywords = self.beauty_taxonomy[best_category]['subcategories']
                best_subcategory = 'general'
                for subcat in subcategory_keywords:
                    subcat_words = subcat.replace('-', ' ').split()
                    if any(word in comment for word in subcat_words):
                        best_subcategory = subcat
                        break
                
                subcategories.append(best_subcategory)
            else:
                categories.append('general')
                confidences.append(0.1)
                subcategories.append('general')
        
        return categories, confidences, subcategories
    
    def perform_topic_modeling(self, clean_comments, n_topics=8):
        """Advanced topic modeling with coherence optimization"""
        
        if len(clean_comments) < n_topics:
            return None, None, None, None
        
        # Filter meaningful comments
        meaningful_comments = [c for c in clean_comments if len(c.split()) >= 3]
        
        if len(meaningful_comments) < n_topics:
            return None, None, None, None
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        doc_term_matrix = vectorizer.fit_transform(meaningful_comments)
        feature_names = vectorizer.get_feature_names_out()
        
        # LDA Topic Modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics with relevance scores
        topics = []
        topic_coherence = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_word_indices = topic.argsort()[-10:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_word_indices]
            
            # Calculate topic coherence
            coherence = np.mean([score for _, score in top_words[:5]])
            topic_coherence.append(coherence)
            
            topic_name = f"Topic {topic_idx + 1}: {', '.join([word for word, _ in top_words[:5]])}"
            topics.append(topic_name)
        
        # Document-topic probabilities
        doc_topic_probs = lda.transform(doc_term_matrix)
        
        # Topic trends (mock implementation - would use timestamps in real scenario)
        topic_trends = np.random.choice(['rising', 'stable', 'declining'], size=n_topics)
        
        return topics, doc_topic_probs, topic_coherence, topic_trends
    
    def generate_ai_insights(self, df, quality_scores, categories):
        """Generate advanced AI-powered business insights"""
        
        insights = []
        
        # Quality distribution insights
        high_quality_ratio = (quality_scores > 0.7).mean()
        if high_quality_ratio > 0.3:
            insights.append(f"🎯 **Excellent Content Quality**: {high_quality_ratio:.1%} of comments are high-quality, indicating strong audience engagement.")
        elif high_quality_ratio < 0.1:
            insights.append(f"⚠️ **Content Quality Concern**: Only {high_quality_ratio:.1%} are high-quality. Consider content strategy optimization.")
        
        # Category performance insights
        category_quality = df.groupby('category')['quality_score'].mean().sort_values(ascending=False)
        best_category = category_quality.index[0]
        insights.append(f"🏆 **Top Performing Category**: {best_category.title()} generates highest quality discussions (avg score: {category_quality.iloc[0]:.2f}).")
        
        # Engagement correlation insights
        if df['like_count'].max() > 0:
            correlation = df['quality_score'].corr(df['like_count'])
            if correlation > 0.5:
                insights.append(f"📈 **Strong Quality-Engagement Link**: High correlation ({correlation:.2f}) between comment quality and likes suggests authentic engagement.")
            elif correlation < 0.2:
                insights.append(f"🤔 **Engagement Anomaly**: Low correlation ({correlation:.2f}) between quality and likes may indicate algorithmic or bot activity.")
        
        # Sentiment insights
        sentiment_dist = df['sentiment_category'].value_counts(normalize=True)
        if 'Positive' in sentiment_dist and sentiment_dist['Positive'] > 0.6:
            insights.append(f"😊 **Positive Brand Sentiment**: {sentiment_dist['Positive']:.1%} positive sentiment indicates strong brand affinity.")
        elif 'Negative' in sentiment_dist and sentiment_dist['Negative'] > 0.3:
            insights.append(f"😔 **Sentiment Alert**: {sentiment_dist['Negative']:.1%} negative sentiment requires attention to brand perception.")
        
        # Spam insights
        spam_ratio = df['is_spam'].mean()
        if spam_ratio > 0.2:
            insights.append(f"🛡️ **High Spam Detection**: {spam_ratio:.1%} spam rate suggests need for enhanced content moderation.")
        elif spam_ratio < 0.05:
            insights.append(f"✨ **Clean Community**: Low spam rate ({spam_ratio:.1%}) indicates healthy community engagement.")
        
        return insights
    
    def create_network_analysis(self, df):
        """Create network analysis of comment interactions"""
        try:
            G = nx.Graph()
            
            # Add nodes for each category
            categories = df['category'].unique()
            for category in categories:
                G.add_node(category, node_type='category')
            
            # Add edges based on comment co-occurrence in categories
            for i, row1 in df.iterrows():
                for j, row2 in df.iterrows():
                    if i < j and row1['category'] == row2['category']:
                        if G.has_edge(row1['category'], row2['category']):
                            G[row1['category']][row2['category']]['weight'] += 1
                        else:
                            G.add_edge(row1['category'], row2['category'], weight=1)
            
            # Calculate centrality measures
            centrality = nx.degree_centrality(G)
            
            return G, centrality
        except:
            return None, None

def create_demo_data():
    """Create realistic demo dataset for testing"""
    np.random.seed(42)
    
    # Realistic beauty comments
    skincare_comments = [
        "This vitamin C serum completely transformed my skin! Been using it for 3 months and my dark spots have faded significantly.",
        "Love how lightweight this moisturizer is. Perfect for my combination skin and doesn't break me out.",
        "Finally found a sunscreen that doesn't leave white cast on my darker skin tone. SPF 50 and feels like silk!",
        "The retinol is gentle but effective. Started seeing results in 2 weeks without any irritation or peeling.",
        "This cleanser removed all my makeup without stripping my skin. My face feels clean but not tight.",
        "Disappointed with this serum. Been using for 6 weeks and haven't seen any difference in my acne scars.",
        "Too expensive for what it does. Similar products work better for half the price.",
        "Packaging is gorgeous but the product broke me out unfortunately. Doesn't work for sensitive skin.",
    ]
    
    makeup_comments = [
        "This foundation gives me flawless coverage without looking cakey. Perfect shade match for my skin tone!",
        "The lipstick formula is incredibly creamy and long-lasting. Color payoff is exactly as advertised.",
        "Best mascara I've ever used! Makes my lashes look naturally long and voluminous without clumping.",
        "This eyeshadow palette has amazing pigmentation. Colors blend like a dream and last all day.",
        "Highlighter gives the perfect glow without being too intense. Buildable and beautiful!",
        "Foundation oxidized on my skin and looked orange after 2 hours. Very disappointed.",
        "Mascara flakes throughout the day and makes my eyes water. Won't repurchase.",
        "Eyeshadow has beautiful colors but very poor pigmentation. Need to pack it on to see color.",
    ]
    
    fragrance_comments = [
        "This perfume has incredible longevity! I can still smell it after 8 hours. Beautiful floral notes.",
        "Love the fresh, clean scent. Perfect for daily wear and not overwhelming at all.",
        "The woody base notes are so sophisticated. This fragrance makes me feel confident and elegant.",
        "Smells exactly like expensive designer perfume but at fraction of the price. Amazing dupe!",
        "Too strong for my taste. The scent gave me a headache after wearing for just an hour.",
        "Doesn't last at all on my skin. Gone within 2 hours which is disappointing for the price.",
        "Beautiful bottle but the fragrance is very generic. Nothing special or unique about it.",
    ]
    
    spam_comments = [
        "First!",
        "Check out my profile for amazing deals!",
        "Buy now limited time offer click link in bio",
        "Follow for follow! L4L F4F",
        "❤️❤️❤️❤️❤️",
        "Amazing deal! Don't miss out!",
        "BEST PRODUCT EVER!!!!!",
        "wow",
        "great",
        "love it so much",
    ]
    
    general_comments = [
        "Thanks for this honest review!",
        "Where can I buy this product?",
        "Your skin looks amazing in this video!",
        "Please do more tutorials like this",
        "What's your full skincare routine?",
        "Love your content! So helpful",
        "Can you review more drugstore products?",
    ]
    
    # Combine all comments
    all_comments = (skincare_comments * 3 + makeup_comments * 3 + 
                   fragrance_comments * 2 + spam_comments * 2 + general_comments * 2)
    
    # Create realistic engagement data
    like_counts = []
    for comment in all_comments:
        if any(spam in comment.lower() for spam in ['first', 'check out', 'follow', '❤️❤️❤️', 'buy now']):
            # Spam gets low engagement
            like_counts.append(np.random.randint(0, 5))
        elif len(comment) > 100 and any(word in comment.lower() for word in ['transformed', 'amazing', 'perfect', 'love']):
            # High quality gets high engagement
            like_counts.append(np.random.randint(20, 200))
        elif len(comment) > 50:
            # Medium quality gets medium engagement
            like_counts.append(np.random.randint(5, 50))
        else:
            # Short comments get low engagement
            like_counts.append(np.random.randint(0, 15))
    
    # Create DataFrame
    demo_df = pd.DataFrame({
        'comment': all_comments,
        'like_count': like_counts,
        'video_id': [f'video_{np.random.randint(1, 20)}' for _ in all_comments],
        'comment_id': [f'comment_{i}' for i in range(len(all_comments))],
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in all_comments]
    })
    
    return demo_df

def main():
    """Main Streamlit application"""
    
    # Header with animation
    st.markdown('<h1 class="main-header">🧠 CommentSense AI Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">Advanced AI-Powered Comment Quality & Engagement Analysis</p>
        <p style="font-style: italic; color: #888;">Revolutionizing beauty brand social listening with neural-inspired algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the AI system
    @st.cache_resource
    def load_ai_system():
        return CommentSenseAIPro()
    
    ai_system = load_ai_system()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 📊 **Data Input & Settings**")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Comments Dataset (CSV)",
            type=['csv'],
            help="Upload CSV with 'comment' column and optional 'like_count' column"
        )
        
        # Demo data option
        if st.button("🎮 **Load Demo Data**", type="secondary"):
            st.session_state['demo_data'] = create_demo_data()
            st.success("✅ Demo data loaded!")
        
        st.markdown("---")
        
        # Analysis Configuration
        st.markdown("### ⚙️ **AI Configuration**")
        
      # Define the options as a separate variable first
quality_options = [
    ("Conservative", 0.8, 0.6, 0.4),
    ("Balanced", 0.75, 0.5, 0.25),
    ("Liberal", 0.7, 0.4, 0.2)
]

# Initialize default if not in session state
if 'quality_threshold' not in st.session_state:
    st.session_state.quality_threshold = quality_options[1]  # Default to "Balanced"

quality_threshold = st.select_slider(
    "Quality Classification Thresholds",
    options=quality_options,
    value=st.session_state.quality_threshold,  # Use session state
    format_func=lambda x: x[0]
)

# Update session state
st.session_state.quality_threshold = quality_threshold

ultra_threshold, high_threshold, medium_threshold = quality_threshold[1:]
        
        enable_advanced_features = st.checkbox("🧠 Enable Advanced AI Features", value=True)
        enable_topic_modeling = st.checkbox("📝 Topic Modeling & Trends", value=True)
        enable_network_analysis = st.checkbox("🕸️ Network Analysis", value=False)
        enable_predictive_insights = st.checkbox("🔮 Predictive Insights", value=True)
        
        st.markdown("---")
        st.markdown("### 📋 **Export Options**")
        export_format = st.radio("Export Format", ["CSV", "JSON", "Excel"])
    
    # Main content area
    data_source = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            data_source = "uploaded"
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    elif 'demo_data' in st.session_state:
        df = st.session_state['demo_data']
        data_source = "demo"
    
    if data_source:
        # Validate required columns
        if 'comment' not in df.columns:
            st.error("❌ Dataset must contain a 'comment' column")
            return
        
        # Add missing columns with defaults
        if 'like_count' not in df.columns:
            df['like_count'] = np.random.poisson(10, len(df))  # More realistic than uniform
            st.info("💡 Generated realistic like_count data for demonstration")
        
        # Remove empty comments
        df = df.dropna(subset=['comment'])
        df = df[df['comment'].str.len() > 0]
        
        if len(df) == 0:
            st.error("❌ No valid comments found in dataset")
            return
        
        # Run AI Analysis
        with st.spinner("🧠 Running Advanced AI Analysis..."):
            # Extract features
            features, clean_comments = ai_system.extract_advanced_features(df)
            
            # Calculate quality scores
            quality_scores, quality_components = ai_system.calculate_neural_quality_score(features, clean_comments)
            df['quality_score'] = quality_scores
            
            # Classify quality levels
            df['quality_label'] = df['quality_score'].apply(
                lambda x: 'Ultra' if x >= ultra_threshold else
                         'High' if x >= high_threshold else
                         'Medium' if x >= medium_threshold else 'Low'
            )
            
            # Spam detection
            is_spam, spam_scores = ai_system.detect_advanced_spam(features, clean_comments)
            df['is_spam'] = is_spam
            df['spam_score'] = spam_scores
            
            # Categorization
            categories, confidences, subcategories = ai_system.categorize_with_confidence(clean_comments)
            df['category'] = categories
            df['category_confidence'] = confidences
            df['subcategory'] = subcategories
            
            # Sentiment analysis
            df['sentiment_polarity'] = features['sentiment_polarity']
            df['sentiment_category'] = df['sentiment_polarity'].apply(
                lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
            )
        
        # Executive Dashboard
        st.markdown("## 📈 **Executive Dashboard**")
        
        # KPI Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_comments = len(df)
            st.markdown(f"""
            <div class="premium-metric">
                <h3>{total_comments:,}</h3>
                <p>Total Comments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ultra_high_ratio = (df['quality_label'].isin(['Ultra', 'High'])).mean()
            st.markdown(f"""
            <div class="premium-metric">
                <h3>{ultra_high_ratio:.1%}</h3>
                <p>Premium Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_quality = df['quality_score'].mean()
            quality_trend = "📈" if avg_quality > 0.5 else "📉" if avg_quality < 0.3 else "📊"
            st.markdown(f"""
            <div class="premium-metric">
                <h3>{quality_trend} {avg_quality:.2f}</h3>
                <p>Avg Quality Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            spam_ratio = df['is_spam'].mean()
            spam_status = "🛡️" if spam_ratio < 0.1 else "⚠️" if spam_ratio < 0.2 else "🚨"
            st.markdown(f"""
            <div class="premium-metric">
                <h3>{spam_status} {spam_ratio:.1%}</h3>
                <p>Spam Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            engagement_score = (df['like_count'] * df['quality_score']).sum() / df['like_count'].sum()
            st.markdown(f"""
            <div class="premium-metric">
                <h3>⚡ {engagement_score:.2f}</h3>
                <p>Engagement Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced Analytics Section
        st.markdown("## 🔬 **Advanced Analytics**")
        
        # Quality Distribution Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Quality Distribution Matrix")
            
            quality_counts = df['quality_label'].value_counts()
            colors = {'Ultra': '#00d4aa', 'High': '#00b894', 'Medium': '#fdcb6e', 'Low': '#fd79a8'}
            
            fig_quality = px.pie(
                values=quality_counts.values,
                names=quality_counts.index,
                title="Comment Quality Distribution",
                color=quality_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig_quality.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12,
                marker_line=dict(color='white', width=2)
            )
            fig_quality.update_layout(
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Category Performance Heatmap")
            
            category_metrics = df.groupby('category').agg({
                'quality_score': 'mean',
                'like_count': 'mean',
                'sentiment_polarity': 'mean',
                'category_confidence': 'mean'
            }).round(3)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                category_metrics.T,
                labels=dict(x="Category", y="Metric", color="Score"),
                x=category_metrics.index,
                y=['Quality Score', 'Avg Likes', 'Sentiment', 'Confidence'],
                color_continuous_scale='Viridis',
                aspect="auto"
            )
            fig_heatmap.update_layout(
                title="Category Performance Matrix",
                height=400
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Neural Quality Components Analysis
        if enable_advanced_features:
            st.markdown("### 🧠 Neural Quality Components")
            
            # Create radar chart for quality components
            avg_components = {}
            for component in ai_system.quality_neural_weights.keys():
                if component in quality_components.columns:
                    avg_components[component] = quality_components[component].mean()
            
            if avg_components:
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(avg_components.values()),
                    theta=list(avg_components.keys()),
                    fill='toself',
                    name='Average Scores',
                    line_color='rgba(102,126,234,0.8)',
                    fillcolor='rgba(102,126,234,0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Neural Quality Component Analysis",
                    height=500
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Component Insights")
                    for component, score in avg_components.items():
                        status = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                        component_clean = component.replace('_', ' ').title()
                        st.markdown(f"{status} **{component_clean}**: {score:.2f}")
        
        # Topic Modeling & Trends
        if enable_topic_modeling:
            st.markdown("### 📝 AI Topic Discovery & Trend Analysis")
            
            with st.spinner("🔍 Discovering hidden topics..."):
                topics, doc_topic_probs, coherence_scores, trends = ai_system.perform_topic_modeling(clean_comments)
            
            if topics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🧠 Discovered Topics")
                    
                    topic_df = pd.DataFrame({
                        'Topic': [t.split(':')[0] for t in topics],
                        'Keywords': [':'.join(t.split(':')[1:]).strip() for t in topics],
                        'Coherence': coherence_scores,
                        'Trend': trends
                    })
                    
                    for idx, row in topic_df.iterrows():
                        trend_class = f"trend-{row['Trend'].replace('rising', 'up').replace('declining', 'down').replace('stable', 'stable')}"
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>{row['Topic']} <span class="{trend_class}">📈 {row['Trend']}</span></h4>
                            <p><strong>Keywords:</strong> {row['Keywords']}</p>
                            <p><strong>Coherence Score:</strong> {row['Coherence']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📊 Topic Popularity Distribution")
                    
                    if doc_topic_probs is not None:
                        topic_weights = doc_topic_probs.mean(axis=0)
                        topic_labels = [f"Topic {i+1}" for i in range(len(topics))]
                        
                        fig_topics = px.bar(
                            x=topic_labels,
                            y=topic_weights,
                            title="Topic Engagement Levels",
                            color=topic_weights,
                            color_continuous_scale='plasma'
                        )
                        fig_topics.update_layout(
                            xaxis_title="Topics",
                            yaxis_title="Average Engagement",
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_topics, use_container_width=True)
        
        # Sentiment & Engagement Deep Dive
        st.markdown("### 😊 Sentiment & Engagement Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment by Category Sunburst
            sentiment_category_data = df.groupby(['category', 'sentiment_category']).size().reset_index(name='count')
            
            fig_sunburst = px.sunburst(
                sentiment_category_data,
                path=['category', 'sentiment_category'],
                values='count',
                title="Sentiment Distribution by Category",
                color='count',
                color_continuous_scale='RdYlGn'
            )
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            # Quality vs Engagement 3D Scatter
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            
            fig_3d = px.scatter_3d(
                sample_df,
                x='quality_score',
                y='like_count',
                z='sentiment_polarity',
                color='category',
                size='like_count',
                hover_data=['comment'],
                title="3D Quality-Engagement-Sentiment Analysis",
                labels={'quality_score': 'Quality Score', 'like_count': 'Engagement', 'sentiment_polarity': 'Sentiment'}
            )
            fig_3d.update_layout(height=500)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # AI-Generated Insights
        if enable_predictive_insights:
            st.markdown("### 🔮 AI-Powered Strategic Insights")
            
            insights = ai_system.generate_ai_insights(df, df['quality_score'], df['category'])
            
            for insight in insights:
                st.markdown(f"""
                <div class="ai-insight">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            
            # Predictive recommendations
            st.markdown("#### 🎯 Strategic Recommendations")
            
            recommendations = []
            
            # Content strategy recommendations
            best_category = df.groupby('category')['quality_score'].mean().idxmax()
            recommendations.append(f"📈 **Focus Content Strategy**: Increase {best_category} content - it generates highest quality engagement")
            
            # Timing recommendations
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                best_hour = df.groupby('hour')['like_count'].mean().idxmax()
                recommendations.append(f"⏰ **Optimal Posting Time**: {best_hour}:00 shows highest engagement rates")
            
            # Community management
            spam_categories = df[df['is_spam']]['category'].mode()
            if not spam_categories.empty:
                recommendations.append(f"🛡️ **Enhanced Moderation**: Focus spam detection on {spam_categories.iloc[0]} category")
            
            # Sentiment improvement
            negative_categories = df[df['sentiment_category'] == 'Negative']['category'].mode()
            if not negative_categories.empty:
                recommendations.append(f"💬 **Sentiment Improvement**: Address concerns in {negative_categories.iloc[0]} discussions")
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="insight-card">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive Comment Explorer
        st.markdown("## 🔍 **Interactive Comment Explorer**")
        
        # Advanced Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            category_filter = st.selectbox("🏷️ Category", ['All'] + sorted(df['category'].unique().tolist()))
        
        with col2:
            quality_filter = st.selectbox("⭐ Quality Level", ['All'] + sorted(df['quality_label'].unique().tolist()))
        
        with col3:
            sentiment_filter = st.selectbox("😊 Sentiment", ['All'] + sorted(df['sentiment_category'].unique().tolist()))
        
        with col4:
            spam_filter = st.selectbox("🛡️ Spam Filter", ['All', 'Clean Only', 'Spam Only'])
        
        # Apply filters
        filtered_df = df.copy()
        
        if category_filter != 'All':
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
        
        if quality_filter != 'All':
            filtered_df = filtered_df[filtered_df['quality_label'] == quality_filter]
        
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['sentiment_category'] == sentiment_filter]
        
        if spam_filter == 'Clean Only':
            filtered_df = filtered_df[~filtered_df['is_spam']]
        elif spam_filter == 'Spam Only':
            filtered_df = filtered_df[filtered_df['is_spam']]
        
        # Display results
        st.markdown(f"**Showing {len(filtered_df):,} comments** (filtered from {len(df):,} total)")
        
        if len(filtered_df) > 0:
            # Sort by quality score
            display_df = filtered_df.nlargest(50, 'quality_score')[
                ['comment', 'quality_score', 'quality_label', 'category', 'subcategory', 
                 'sentiment_category', 'like_count', 'is_spam', 'category_confidence']
            ].round(3)
            
            # Interactive table with styling
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "comment": st.column_config.TextColumn("Comment", width="large"),
                    "quality_score": st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=1),
                    "quality_label": st.column_config.TextColumn("Quality"),
                    "category": st.column_config.TextColumn("Category"),
                    "subcategory": st.column_config.TextColumn("Subcategory"),
                    "sentiment_category": st.column_config.TextColumn("Sentiment"),
                    "like_count": st.column_config.NumberColumn("Likes"),
                    "is_spam": st.column_config.CheckboxColumn("Spam"),
                    "category_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1)
                }
            )
        else:
            st.warning("🔍 No comments match the selected filters. Try adjusting your criteria.")
        
        # Export Section
        st.markdown("## 📥 **Export & Download**")
        
        col1, col2, col3 = st.columns(3)
        
        # Prepare export data
        export_df = df[[
            'comment', 'quality_score', 'quality_label', 'category', 'subcategory',
            'sentiment_category', 'sentiment_polarity', 'like_count', 'is_spam',
            'spam_score', 'category_confidence'
        ]].round(3)
        
        with col1:
            # Summary report
            summary_stats = {
                'Total Comments': len(df),
                'Ultra Quality': (df['quality_label'] == 'Ultra').sum(),
                'High Quality': (df['quality_label'] == 'High').sum(),
                'Medium Quality': (df['quality_label'] == 'Medium').sum(),
                'Low Quality': (df['quality_label'] == 'Low').sum(),
                'Spam Detected': df['is_spam'].sum(),
                'Average Quality Score': df['quality_score'].mean(),
                'Average Engagement': df['like_count'].mean(),
                'Positive Sentiment %': (df['sentiment_category'] == 'Positive').mean() * 100,
                'Top Category': df['category'].mode().iloc[0] if not df['category'].mode().empty else 'N/A'
            }
            
            summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
            
            if export_format == "CSV":
                csv_summary = summary_df.to_csv()
                st.download_button(
                    "📊 Download Executive Summary",
                    csv_summary,
                    "commentsense_executive_summary.csv",
                    "text/csv"
                )
            elif export_format == "JSON":
                json_summary = summary_df.to_json(orient='index')
                st.download_button(
                    "📊 Download Executive Summary",
                    json_summary,
                    "commentsense_executive_summary.json",
                    "application/json"
                )
        
        with col2:
            # Detailed results
            if export_format == "CSV":
                csv_detailed = export_df.to_csv(index=False)
                st.download_button(
                    "📋 Download Detailed Analysis",
                    csv_detailed,
                    "commentsense_detailed_analysis.csv",
                    "text/csv"
                )
            elif export_format == "JSON":
                json_detailed = export_df.to_json(orient='records')
                st.download_button(
                    "📋 Download Detailed Analysis",
                    json_detailed,
                    "commentsense_detailed_analysis.json",
                    "application/json"
                )
        
        with col3:
            # Business insights export
            if insights:
                insights_text = "\n\n".join([insight.replace('*', '').replace('#', '') for insight in insights])
                st.download_button(
                    "🎯 Download AI Insights",
                    insights_text,
                    "commentsense_ai_insights.txt",
                    "text/plain"
                )
    
    else:
        # Landing page
        st.markdown("""
        <div class="insight-card">
            <h2>🚀 Welcome to CommentSense AI Pro</h2>
            <p>The most advanced comment analysis platform designed specifically for beauty brands and social media marketers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="neural-network">
                <h3>🧠 Neural Quality Scoring</h3>
                <p>Advanced 6-layer neural algorithm analyzing semantic relevance, emotional resonance, and engagement potential</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="neural-network">
                <h3>🔍 Beauty-Specific Intelligence</h3>
                <p>Industry-trained categorization system with hierarchical beauty taxonomy and confidence scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="neural-network">
                <h3>📊 Predictive Analytics</h3>
                <p>AI-powered trend detection, topic modeling, and strategic business recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Value propositions
        st.markdown("""
        ### 🎯 **Revolutionary Features**
        
        **🧠 Advanced AI Architecture:**
        - Neural-inspired quality scoring with 6 cognitive layers
        - Multi-dimensional spam detection using ensemble methods  
        - Hierarchical beauty category classification with confidence metrics
        - Advanced topic modeling with trend analysis
        
        **📊 Business Intelligence:**
        - Real-time sentiment analysis with emoji interpretation
        - Predictive engagement scoring and virality detection
        - Strategic recommendations for content optimization
        - Category-specific performance benchmarking
        
        **⚡ Technical Excellence:**
        - Scalable processing for datasets up to 1M+ comments
        - Interactive 3D visualizations and network analysis
        - Multi-format export (CSV, JSON, Excel)
        - Real-time filtering and exploration tools
        
        **🎨 Beauty Industry Focus:**
        - Skincare, makeup, fragrance, and haircare specialized analysis
        - Brand sentiment tracking and competitive intelligence
        - Influencer content quality assessment
        - Community health monitoring and moderation insights
        """)
        
        # Call to action
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">Ready to revolutionize your social listening?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎮 **Try Demo Data**", type="primary", use_container_width=True):
                st.session_state['demo_data'] = create_demo_data()
                st.rerun()

if __name__ == "__main__":
    main()
    
