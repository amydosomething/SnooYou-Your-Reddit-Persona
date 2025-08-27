#!/usr/bin/env python3
"""
Reddit Profile Analyzer - Streamlit GUI
A beautiful web interface for analyzing Reddit user profiles
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from PIL import Image
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import sys

# Import your Reddit analyzer
from reddit_analyzer import RedditAnalyzer

# Streamlit page configuration
st.set_page_config(
    page_title="Reddit Profile Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4500;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .content-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .top-post-card {
    border-left: 5px solid #4CAF50;
    background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
    color: red; /* 🔴 makes all text inside this card red */
}

.top-comment-card {
    border-left: 5px solid #2196F3;
    background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
    color: red; /* 🔴 text red inside comment card */
}

.downvoted-card {
    border-left: 5px solid #f44336;
    background: linear-gradient(135deg, #fff8f8 0%, #ffe8e8 100%);
    color: red; /* 🔴 text red inside downvoted card */
}

    .subreddit-tag {
        background: #FF4500;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .score-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .score-negative {
        color: #f44336;
        font-weight: bold;
    }
    
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Reddit Profile Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze any Reddit user's activity, engagement, and interests")
    
    # Sidebar for inputs
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Input fields
    username = st.sidebar.text_input(
        "Reddit Username",
        placeholder="Enter username (without u/)",
        help="Enter the Reddit username you want to analyze (don't include u/)"
    )
    
    limit = st.sidebar.slider(
        "Posts/Comments to fetch",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="More data = better analysis but slower processing"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Profile", type="primary")
    
    # Add some info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 What you'll get:")
    st.sidebar.markdown("""
    - **Profile Stats**: Karma, account age, activity metrics
    - **Top Content**: Best and worst posts/comments
    - **Subreddit Analysis**: Favorite communities
    - **Word Cloud**: Most used words
    - **AI Insights**: Personality analysis
    - **Interactive Charts**: Visual data exploration
    """)
    
    # Main content area
    if not username and not analyze_button:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ## 👋 Welcome to Reddit Profile Analyzer!
            
            This tool provides comprehensive analysis of any public Reddit profile, including:
            
            🏆 **Content Highlights** - Top posts and comments with scores  
            📊 **Activity Patterns** - Favorite subreddits and engagement style  
            ☁️ **Word Clouds** - Most frequently used words  
            🤖 **AI Insights** - Personality analysis powered by Gemini AI  
            📈 **Visual Analytics** - Interactive charts and graphs  
            
            ### How to use:
            1. Enter a Reddit username in the sidebar (without u/)
            2. Choose how many posts/comments to analyze
            3. Click "Analyze Profile" and wait for results
            
            ### Example usernames to try:
            - `spez` (Reddit CEO)
            - `GallowBoob` (Famous Reddit user)
            - Or any public Reddit username!
            """)
    
    elif analyze_button and username:
        # Analysis process
        with st.spinner(f"🔍 Analyzing u/{username}... This may take a moment!"):
            try:
                # Initialize analyzer
                analyzer = RedditAnalyzer()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Fetch user data
                status_text.text("📥 Fetching user profile...")
                progress_bar.progress(20)
                user_data = analyzer.fetch_user_data(username, limit)
                
                if not user_data:
                    st.error(f"❌ Could not find user 'u/{username}' or failed to fetch data. Please check:")
                    st.markdown("""
                    - Username spelling is correct
                    - User account exists and is public
                    - Your API credentials are properly configured
                    """)
                    return
                
                # Analyze activity
                status_text.text("🔍 Analyzing activity patterns...")
                progress_bar.progress(40)
                activity_data = analyzer.analyze_activity_patterns()
                
                # Generate word cloud
                status_text.text("☁️ Generating word cloud...")
                progress_bar.progress(60)
                
                # Generate AI insights
                status_text.text("🤖 Getting AI insights...")
                progress_bar.progress(80)
                ai_insights = analyzer.generate_ai_insights()
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                display_results(user_data, activity_data, ai_insights, analyzer)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.markdown("""
                ### Possible solutions:
                - Check your .env file has valid API credentials
                - Ensure the username exists and is public
                - Try a smaller fetch limit
                - Check your internet connection
                """)
    
    elif analyze_button and not username:
        st.warning("⚠️ Please enter a username first!")

def display_results(user_data, activity_data, ai_insights, analyzer):
    """Display all analysis results in a beautiful layout"""
    
    username = user_data.get('username', 'Unknown')
    
    # Header with user info
    st.markdown(f"# 📊 Analysis Results for u/{username}")
    
    # Key metrics at the top
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🏆 Total Karma", 
            value=f"{user_data['total_karma']:,}",
            help="Combined post and comment karma"
        )
    
    with col2:
        st.metric(
            label="📝 Post Karma", 
            value=f"{user_data['link_karma']:,}"
        )
    
    with col3:
        st.metric(
            label="💬 Comment Karma", 
            value=f"{user_data['comment_karma']:,}"
        )
    
    with col4:
        st.metric(
            label="📅 Account Age", 
            value=f"{user_data['account_age_days']} days",
            delta=f"{user_data['account_age_days']//365} years" if user_data['account_age_days'] > 365 else None
        )
    
    with col5:
        total_activity = activity_data['activity_stats']['total_posts'] + activity_data['activity_stats']['total_comments']
        st.metric(
            label="📊 Total Activity Analyzed", 
            value=f"{total_activity:,}",
            help="Total posts and comments analyzed"
        )
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Content Highlights", 
        "📊 Activity Analysis", 
        "☁️ Word Cloud", 
        "🤖 AI Insights",
        "📈 Advanced Analytics"
    ])
    
    with tab1:
        display_content_highlights(activity_data)
    
    with tab2:
        display_activity_analysis(activity_data, user_data)
    
    with tab3:
        display_word_cloud(analyzer)
    
    with tab4:
        display_ai_insights(ai_insights)
    
    with tab5:
        display_advanced_analytics(activity_data, analyzer)

def display_content_highlights(activity_data):
    """Display top and worst content in beautiful cards"""
    st.markdown("## 🏆 Content Highlights")
    
    top_content = activity_data['top_content']
    
    # Top content row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Post")
        if top_content['top_post']:
            post = top_content['top_post']
            
            st.markdown(f"""
            <div class="content-card top-post-card">
                <h4>{post['title']}</h4>
                <span class="subreddit-tag">r/{post['subreddit']}</span>
                <p><span class="score-positive">↑ {post['score']:,} upvotes</span></p>
                <p><em>"{(post['selftext'][:150] + '...') if post['selftext'] and len(post['selftext']) > 150 else (post['selftext'] or 'Link post - no text content')}"</em></p>
                <small>Posted on {datetime.fromtimestamp(post['created_utc']).strftime('%B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💬 Top Comment")
        if top_content['top_comment']:
            comment = top_content['top_comment']
            
            st.markdown(f"""
            <div class="content-card top-comment-card">
                <p><strong>Thread:</strong> {comment['submission_title'][:50]}{'...' if len(comment['submission_title']) > 50 else ''}</p>
                <span class="subreddit-tag">r/{comment['subreddit']}</span>
                <p><span class="score-positive">↑ {comment['score']:,} upvotes</span></p>
                <p><em>"{(comment['body'][:150] + '...') if len(comment['body']) > 150 else comment['body']}"</em></p>
                <small>Posted on {datetime.fromtimestamp(comment['created_utc']).strftime('%B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No comments found")
    
    # Controversial content row
    st.markdown("### 👎 Most Controversial Content")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Most Downvoted Post")
        if top_content['worst_post'] and top_content['worst_post']['score'] < 0:
            post = top_content['worst_post']
            
            st.markdown(f"""
            <div class="content-card downvoted-card">
                <h4>{post['title']}</h4>
                <span class="subreddit-tag">r/{post['subreddit']}</span>
                <p><span class="score-negative">↓ {post['score']:,} points</span></p>
                <p><em>"{(post['selftext'][:150] + '...') if post['selftext'] and len(post['selftext']) > 150 else (post['selftext'] or 'Link post - no text content')}"</em></p>
                <small>Posted on {datetime.fromtimestamp(post['created_utc']).strftime('%B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No significantly downvoted posts found")
    
    with col4:
        st.markdown("#### Most Downvoted Comment")
        if top_content['worst_comment'] and top_content['worst_comment']['score'] < 0:
            comment = top_content['worst_comment']
            
            st.markdown(f"""
            <div class="content-card downvoted-card">
                <p><strong>Thread:</strong> {comment['submission_title'][:50]}{'...' if len(comment['submission_title']) > 50 else ''}</p>
                <span class="subreddit-tag">r/{comment['subreddit']}</span>
                <p><span class="score-negative">↓ {comment['score']:,} points</span></p>
                <p><em>"{(comment['body'][:150] + '...') if len(comment['body']) > 150 else comment['body']}"</em></p>
                <small>Posted on {datetime.fromtimestamp(comment['created_utc']).strftime('%B %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No significantly downvoted comments found")

def display_activity_analysis(activity_data, user_data):
    """Display activity patterns and subreddit analysis"""
    st.markdown("## 📊 Activity Analysis")
    
    # Activity breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Posts vs Comments pie chart
        activity_stats = activity_data['activity_stats']
        
        fig = go.Figure(data=[go.Pie(
            labels=['Posts', 'Comments'],
            values=[activity_stats['total_posts'], activity_stats['total_comments']],
            hole=.3,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )])
        
        fig.update_layout(
            title="Posts vs Comments Distribution",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity summary
        st.markdown(f"""
        **Activity Summary:**
        - **{activity_stats['posting_ratio']:.1f}%** Posts
        - **{activity_stats['commenting_ratio']:.1f}%** Comments
        - **Average post score:** {activity_stats['avg_post_score']:.1f}
        - **Average comment score:** {activity_stats['avg_comment_score']:.1f}
        """)
    
    with col2:
        # Top subreddits bar chart
        top_subs = dict(list(activity_data['top_subreddits'].items())[:10])
        
        if top_subs:
            fig = go.Figure([go.Bar(
                x=list(top_subs.values()),
                y=list(top_subs.keys()),
                orientation='h',
                marker_color='#FF4500'
            )])
            
            fig.update_layout(
                title="Top 10 Subreddits by Activity",
                xaxis_title="Number of Posts + Comments",
                yaxis_title="Subreddit",
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No subreddit data available")
    
    # Engagement style analysis
    st.markdown("### 🎯 Engagement Style")
    
    total_posts = activity_stats['total_posts']
    total_comments = activity_stats['total_comments']
    
    if total_posts > total_comments * 2:
        style = "Content Creator 📝"
        description = "This user primarily creates original posts rather than commenting on others' content."
    elif total_comments > total_posts * 5:
        style = "Active Commenter 💬"
        description = "This user prefers engaging in discussions through comments rather than creating posts."
    elif total_posts + total_comments < 50:
        style = "Lurker 👀"
        description = "This user has relatively low activity and may prefer reading over posting."
    else:
        style = "Balanced Contributor ⚖️"
        description = "This user maintains a good balance between posting content and commenting."
    
    st.info(f"**{style}**: {description}")

def display_word_cloud(analyzer):
    """Display word cloud visualization"""
    st.markdown("## ☁️ Most Used Words")
    
    try:
        # Generate word cloud
        all_text = []
        
        for post in analyzer.posts_data:
            if post['title']:
                all_text.append(post['title'])
            if post['selftext']:
                all_text.append(post['selftext'])
        
        for comment in analyzer.comments_data:
            if comment['body'] and comment['body'] != '[deleted]':
                all_text.append(comment['body'])
        
        if all_text:
            text = ' '.join(all_text)
            # Clean text
            import re
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5
            ).generate(text)
            
            # Convert to image
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            st.pyplot(fig)
            
            # Word frequency table
            st.markdown("### 📋 Top Words")
            word_freq = wordcloud.words_
            if word_freq:
                df = pd.DataFrame(
                    list(word_freq.items())[:20], 
                    columns=['Word', 'Frequency']
                )
                df['Frequency'] = df['Frequency'].round(3)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No text data available for word cloud generation")
            
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

def display_ai_insights(ai_insights):
    """Display AI-generated insights"""
    st.markdown("## 🤖 AI-Powered Personality Analysis")
    
    if "Error" not in ai_insights:
        st.markdown(ai_insights)
    else:
        st.warning("AI insights are currently unavailable")
        st.markdown(ai_insights)
        
        # Provide manual insights as fallback
        st.markdown("### 💡 Manual Analysis")
        st.info("Check the Activity Analysis tab for detailed engagement patterns and subreddit preferences.")

def display_advanced_analytics(activity_data, analyzer):
    """Display advanced analytics and network graphs"""
    st.markdown("## 📈 Advanced Analytics")
    
    # Subreddit network graph
    st.markdown("### 🕸️ Interest Network")
    
    try:
        top_subs = dict(list(activity_data['top_subreddits'].items())[:15])
        
        if len(top_subs) > 2:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for sub, count in top_subs.items():
                G.add_node(sub, size=count)
            
            # Add edges based on co-occurrence
            subreddit_list = list(top_subs.keys())
            for i, sub1 in enumerate(subreddit_list):
                for sub2 in subreddit_list[i+1:]:
                    if abs(top_subs[sub1] - top_subs[sub2]) < max(top_subs.values()) * 0.5:
                        G.add_edge(sub1, sub2)
            
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Prepare data for Plotly
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = [f"r/{node}<br>Activity: {top_subs[node]}" for node in G.nodes()]
            node_size = [max(10, min(50, top_subs[node] * 2)) for node in G.nodes()]
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_text,
                text=[f"r/{node}" for node in G.nodes()],
                textposition="middle center",
                marker=dict(
                    size=node_size,
                    color='#FF4500',
                    line=dict(width=2, color='white')
                ),
                name='Subreddits'
            ))
            
            fig.update_layout(
                title="Subreddit Interest Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Subreddits are connected based on similar activity levels")
        else:
            st.info("Not enough subreddit data to create network graph")
    
    except Exception as e:
        st.error(f"Error creating network graph: {str(e)}")
    
    # Activity timeline
    st.markdown("### 📅 Activity Timeline")
    
    try:
        if analyzer.posts_data:
            # Convert timestamps to dates
            post_dates = [datetime.fromtimestamp(post['created_utc']) for post in analyzer.posts_data]
            comment_dates = [datetime.fromtimestamp(comment['created_utc']) for comment in analyzer.comments_data]
            
            # Create timeline data
            timeline_data = []
            
            for date in post_dates:
                timeline_data.append({'Date': date.date(), 'Type': 'Post', 'Count': 1})
            
            for date in comment_dates:
                timeline_data.append({'Date': date.date(), 'Type': 'Comment', 'Count': 1})
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                df_grouped = df.groupby(['Date', 'Type']).count().reset_index()
                
                fig = px.line(df_grouped, x='Date', y='Count', color='Type',
                             title="Activity Over Time")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
        else:
            st.info("No activity data available for timeline")
    
    except Exception as e:
        st.error(f"Error creating timeline: {str(e)}")

if __name__ == "__main__":
    main()