# Reddit Profile Analyzer
# A comprehensive tool for analyzing Reddit user profiles with AI-powered insights

import os
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import google.generativeai as genai
from collections import Counter, defaultdict
from datetime import datetime, timezone
import re
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedditAnalyzer:
    def __init__(self):
        """Initialize the Reddit analyzer with API credentials"""
        # Reddit API credentials
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'RedditAnalyzer/1.0')
        )
        
        # Gemini AI configuration
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        
        # Initialize data storage
        self.user_data = {}
        self.posts_data = []
        self.comments_data = []
        
    def fetch_user_data(self, username: str, limit: int = 100) -> Dict:
        """
        Fetch comprehensive user data from Reddit
        
        Args:
            username: Reddit username (without u/)
            limit: Maximum number of posts/comments to fetch
            
        Returns:
            Dictionary containing user profile data
        """
        try:
            user = self.reddit.redditor(username)
            
            # Basic user info
            self.user_data = {
                'username': username,
                'created_utc': user.created_utc,
                'account_age_days': (datetime.now(timezone.utc) - datetime.fromtimestamp(user.created_utc, timezone.utc)).days,
                'total_karma': user.link_karma + user.comment_karma,
                'link_karma': user.link_karma,
                'comment_karma': user.comment_karma,
                'is_gold': user.is_gold,
                'is_mod': user.is_mod,
                'has_verified_email': user.has_verified_email
            }
            
            # Fetch posts
            print(f"Fetching posts for u/{username}...")
            posts = list(user.submissions.new(limit=limit))
            self.posts_data = []
            
            for post in posts:
                self.posts_data.append({
                    'id': post.id,
                    'title': post.title,
                    'subreddit': str(post.subreddit),
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'selftext': post.selftext,
                    'url': post.url,
                    'is_self': post.is_self,
                    'domain': post.domain,
                    'gilded': post.gilded
                })
            
            # Fetch comments
            print(f"Fetching comments for u/{username}...")
            comments = list(user.comments.new(limit=limit))
            self.comments_data = []
            
            for comment in comments:
                self.comments_data.append({
                    'id': comment.id,
                    'body': comment.body,
                    'subreddit': str(comment.subreddit),
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'gilded': comment.gilded,
                    'parent_id': comment.parent_id,
                    'submission_title': comment.submission.title if hasattr(comment, 'submission') else '',
                    'is_submitter': comment.is_submitter
                })
                
            print(f"Fetched {len(self.posts_data)} posts and {len(self.comments_data)} comments")
            return self.user_data
            
        except Exception as e:
            print(f"Error fetching user data: {str(e)}")
            return {}
    
    def analyze_activity_patterns(self) -> Dict:
        """Analyze user's activity patterns and engagement"""
        if not self.posts_data and not self.comments_data:
            return {}
        
        # Subreddit activity
        post_subs = Counter([post['subreddit'] for post in self.posts_data])
        comment_subs = Counter([comment['subreddit'] for comment in self.comments_data])
        
        # Combine and get top subreddits
        all_subs = post_subs + comment_subs
        top_subreddits = dict(all_subs.most_common(20))
        
        # Activity ratios
        total_posts = len(self.posts_data)
        total_comments = len(self.comments_data)
        total_activity = total_posts + total_comments
        
        posting_ratio = (total_posts / total_activity * 100) if total_activity > 0 else 0
        commenting_ratio = (total_comments / total_activity * 100) if total_activity > 0 else 0
        
        # Engagement metrics
        avg_post_score = np.mean([post['score'] for post in self.posts_data]) if self.posts_data else 0
        avg_comment_score = np.mean([comment['score'] for comment in self.comments_data]) if self.comments_data else 0
        
        # Find top and worst content
        top_post = max(self.posts_data, key=lambda x: x['score']) if self.posts_data else None
        worst_post = min(self.posts_data, key=lambda x: x['score']) if self.posts_data else None
        top_comment = max(self.comments_data, key=lambda x: x['score']) if self.comments_data else None
        worst_comment = min(self.comments_data, key=lambda x: x['score']) if self.comments_data else None
        
        return {
            'top_subreddits': top_subreddits,
            'activity_stats': {
                'total_posts': total_posts,
                'total_comments': total_comments,
                'posting_ratio': posting_ratio,
                'commenting_ratio': commenting_ratio,
                'avg_post_score': avg_post_score,
                'avg_comment_score': avg_comment_score
            },
            'top_content': {
                'top_post': top_post,
                'worst_post': worst_post,
                'top_comment': top_comment,
                'worst_comment': worst_comment
            }
        }
    
    def generate_word_cloud(self, save_path: str = 'wordcloud.png') -> str:
        """Generate word cloud from user's posts and comments"""
        # Combine all text
        all_text = []
        
        for post in self.posts_data:
            if post['title']:
                all_text.append(post['title'])
            if post['selftext']:
                all_text.append(post['selftext'])
        
        for comment in self.comments_data:
            if comment['body'] and comment['body'] != '[deleted]':
                all_text.append(comment['body'])
        
        if not all_text:
            return "No text data available for word cloud"
        
        # Clean and combine text
        text = ' '.join(all_text)
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Save word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for u/{self.user_data.get('username', 'Unknown')}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return save_path
    
    def create_interest_network(self, save_path: str = 'interest_network.html') -> str:
        """Create an interactive network graph of subreddit interests"""
        if not hasattr(self, 'activity_data'):
            self.activity_data = self.analyze_activity_patterns()
        
        top_subs = self.activity_data['top_subreddits']
        
        if not top_subs:
            return "No subreddit data available"
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (subreddits)
        for sub, count in top_subs.items():
            G.add_node(sub, size=count)
        
        # Add edges based on co-occurrence (simplified)
        subreddit_list = list(top_subs.keys())
        for i, sub1 in enumerate(subreddit_list):
            for sub2 in subreddit_list[i+1:]:
                # Simple heuristic: connect if both have similar activity levels
                if abs(top_subs[sub1] - top_subs[sub2]) < max(top_subs.values()) * 0.3:
                    G.add_edge(sub1, sub2)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare data for Plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"r/{node}<br>Activity: {top_subs[node]}")
            node_size.append(max(10, min(50, top_subs[node] * 2)))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
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
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        ))
        
        fig.update_layout(
            title=f"Interest Network for u/{self.user_data.get('username', 'Unknown')}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Subreddits connected by similar activity levels",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#999', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        fig.write_html(save_path)
        return save_path
    
    def create_content_highlights_html(self, save_path: str = 'content_highlights.html') -> str:
        """Create beautiful HTML cards for top/worst posts and comments"""
        if not hasattr(self, 'activity_data'):
            self.activity_data = self.analyze_activity_patterns()
        
        top_content = self.activity_data['top_content']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Content Highlights for u/{self.user_data.get('username', 'Unknown')}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    color: white;
                    margin-bottom: 40px;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin: 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .highlights-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 30px;
                    margin-bottom: 40px;
                }}
                .highlight-card {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }}
                .highlight-card:hover {{
                    transform: translateY(-5px);
                }}
                .card-header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }}
                .card-icon {{
                    font-size: 2em;
                    margin-right: 15px;
                }}
                .card-title {{
                    font-size: 1.4em;
                    font-weight: bold;
                    color: #333;
                }}
                .post-card {{
                    border-left: 5px solid #4CAF50;
                }}
                .comment-card {{
                    border-left: 5px solid #2196F3;
                }}
                .downvoted-card {{
                    border-left: 5px solid #f44336;
                    background: linear-gradient(135deg, #fff 0%, #ffebee 100%);
                }}
                .content-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #1a1a1a;
                    margin: 10px 0;
                    line-height: 1.3;
                }}
                .subreddit-tag {{
                    background: #ff4500;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: bold;
                    display: inline-block;
                    margin: 5px 0;
                }}
                .score {{
                    font-size: 1.5em;
                    font-weight: bold;
                    padding: 8px 15px;
                    border-radius: 25px;
                    display: inline-block;
                    margin: 10px 0;
                }}
                .score.positive {{
                    background: #4CAF50;
                    color: white;
                }}
                .score.negative {{
                    background: #f44336;
                    color: white;
                }}
                .content-preview {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #ddd;
                    margin: 15px 0;
                    font-style: italic;
                    color: #555;
                    line-height: 1.5;
                }}
                .thread-info {{
                    background: #e3f2fd;
                    padding: 10px 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                    font-size: 0.9em;
                    color: #1565c0;
                }}
                .no-content {{
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    padding: 40px 20px;
                    background: #f5f5f5;
                    border-radius: 10px;
                }}
                .stats-overlay {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background: rgba(0,0,0,0.8);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.8em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 Content Highlights</h1>
                    <p>Best and worst content for u/{self.user_data.get('username', 'Unknown')}</p>
                </div>
                
                <div class="highlights-grid">
        """
        
        # Top Post
        if top_content['top_post']:
            post = top_content['top_post']
            preview_text = post['selftext'][:200] + '...' if post['selftext'] and len(post['selftext']) > 200 else post['selftext'] or 'Link post - no text content'
            
            html_content += f"""
                    <div class="highlight-card post-card">
                        <div class="stats-overlay">#{1} Post</div>
                        <div class="card-header">
                            <div class="card-icon">🏆</div>
                            <div class="card-title">Top Post</div>
                        </div>
                        <div class="content-title">{post['title']}</div>
                        <div class="subreddit-tag">r/{post['subreddit']}</div>
                        <div class="score positive">↑ {post['score']:,} upvotes</div>
                        <div class="content-preview">
                            "{preview_text}"
                        </div>
                        <small style="color: #666;">Posted {datetime.fromtimestamp(post['created_utc']).strftime('%B %d, %Y')}</small>
                    </div>
            """
        else:
            html_content += """
                    <div class="highlight-card">
                        <div class="no-content">No posts found</div>
                    </div>
            """
        
        # Top Comment
        if top_content['top_comment']:
            comment = top_content['top_comment']
            comment_text = comment['body'][:300] + '...' if len(comment['body']) > 300 else comment['body']
            
            html_content += f"""
                    <div class="highlight-card comment-card">
                        <div class="stats-overlay">#{1} Comment</div>
                        <div class="card-header">
                            <div class="card-icon">💬</div>
                            <div class="card-title">Top Comment</div>
                        </div>
                        <div class="thread-info">
                            💭 Thread: "{comment['submission_title'][:50]}{'...' if len(comment['submission_title']) > 50 else ''}"
                        </div>
                        <div class="subreddit-tag">r/{comment['subreddit']}</div>
                        <div class="score positive">↑ {comment['score']:,} upvotes</div>
                        <div class="content-preview">
                            "{comment_text}"
                        </div>
                        <small style="color: #666;">Posted {datetime.fromtimestamp(comment['created_utc']).strftime('%B %d, %Y')}</small>
                    </div>
            """
        else:
            html_content += """
                    <div class="highlight-card">
                        <div class="no-content">No comments found</div>
                    </div>
            """
        
        # Worst Post
        if top_content['worst_post']:
            post = top_content['worst_post']
            preview_text = post['selftext'][:200] + '...' if post['selftext'] and len(post['selftext']) > 200 else post['selftext'] or 'Link post - no text content'
            
            html_content += f"""
                    <div class="highlight-card downvoted-card">
                        <div class="stats-overlay">Controversial</div>
                        <div class="card-header">
                            <div class="card-icon">👎</div>
                            <div class="card-title">Most Downvoted Post</div>
                        </div>
                        <div class="content-title">{post['title']}</div>
                        <div class="subreddit-tag">r/{post['subreddit']}</div>
                        <div class="score negative">↓ {post['score']:,} points</div>
                        <div class="content-preview">
                            "{preview_text}"
                        </div>
                        <small style="color: #666;">Posted {datetime.fromtimestamp(post['created_utc']).strftime('%B %d, %Y')}</small>
                    </div>
            """
        else:
            html_content += """
                    <div class="highlight-card">
                        <div class="no-content">No controversial posts found</div>
                    </div>
            """
        
        # Worst Comment
        if top_content['worst_comment']:
            comment = top_content['worst_comment']
            comment_text = comment['body'][:300] + '...' if len(comment['body']) > 300 else comment['body']
            
            html_content += f"""
                    <div class="highlight-card downvoted-card">
                        <div class="stats-overlay">Controversial</div>
                        <div class="card-header">
                            <div class="card-icon">💬</div>
                            <div class="card-title">Most Downvoted Comment</div>
                        </div>
                        <div class="thread-info">
                            💭 Thread: "{comment['submission_title'][:50]}{'...' if len(comment['submission_title']) > 50 else ''}"
                        </div>
                        <div class="subreddit-tag">r/{comment['subreddit']}</div>
                        <div class="score negative">↓ {comment['score']:,} points</div>
                        <div class="content-preview">
                            "{comment_text}"
                        </div>
                        <small style="color: #666;">Posted {datetime.fromtimestamp(comment['created_utc']).strftime('%B %d, %Y')}</small>
                    </div>
            """
        else:
            html_content += """
                    <div class="highlight-card">
                        <div class="no-content">No controversial comments found</div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return save_path

    def create_activity_dashboard(self, save_path: str = 'dashboard.html') -> str:
        """Create a comprehensive dashboard with multiple visualizations"""
        if not hasattr(self, 'activity_data'):
            self.activity_data = self.analyze_activity_patterns()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Subreddits', 'Activity Breakdown', 'Score Distribution', 'Temporal Activity'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # 1. Top Subreddits Bar Chart
        top_subs = dict(list(self.activity_data['top_subreddits'].items())[:10])
        fig.add_trace(
            go.Bar(x=list(top_subs.keys()), y=list(top_subs.values()), name='Activity'),
            row=1, col=1
        )
        
        # 2. Activity Breakdown Pie Chart
        activity_stats = self.activity_data['activity_stats']
        fig.add_trace(
            go.Pie(labels=['Posts', 'Comments'], 
                   values=[activity_stats['total_posts'], activity_stats['total_comments']]),
            row=1, col=2
        )
        
        # 3. Score Distribution
        all_scores = [post['score'] for post in self.posts_data] + [comment['score'] for comment in self.comments_data]
        if all_scores:
            fig.add_trace(
                go.Histogram(x=all_scores, name='Score Distribution'),
                row=2, col=1
            )
        
        # 4. Temporal Activity (simplified)
        if self.posts_data:
            post_dates = [datetime.fromtimestamp(post['created_utc']) for post in self.posts_data]
            post_counts = Counter([date.strftime('%Y-%m') for date in post_dates])
            
            fig.add_trace(
                go.Scatter(x=list(post_counts.keys()), y=list(post_counts.values()), 
                          mode='lines+markers', name='Posts over Time'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"Reddit Activity Dashboard for u/{self.user_data.get('username', 'Unknown')}",
            showlegend=False
        )
        
        fig.write_html(save_path)
        return save_path
    
    def generate_ai_insights(self) -> str:
        """Use Gemini AI to generate insights about the user's profile"""
        if not hasattr(self, 'activity_data'):
            self.activity_data = self.analyze_activity_patterns()
        
        # Prepare data for AI analysis
        user_summary = {
            'username': self.user_data.get('username', 'Unknown'),
            'account_age_days': self.user_data.get('account_age_days', 0),
            'total_karma': self.user_data.get('total_karma', 0),
            'top_subreddits': list(self.activity_data['top_subreddits'].keys())[:10],
            'activity_stats': self.activity_data['activity_stats'],
            'top_post_title': self.activity_data['top_content']['top_post']['title'] if self.activity_data['top_content']['top_post'] else None,
            'avg_scores': {
                'posts': self.activity_data['activity_stats']['avg_post_score'],
                'comments': self.activity_data['activity_stats']['avg_comment_score']
            }
        }
        
        prompt = f"""
        Analyze this Reddit user profile and provide insights:
        
        User Data: {json.dumps(user_summary, indent=2)}
        
        Please provide:
        1. A personality snapshot based on their subreddit activity
        2. Estimated age range and interests
        3. Engagement style (lurker, active commenter, content creator)
        4. Notable patterns or characteristics
        5. A fun nickname that captures their Reddit persona
        
        Keep it insightful but respectful and avoid making assumptions about sensitive personal details.
        """
        
        try:
            # Try different model names in order of preference
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'models/gemini-1.5-flash']
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as model_error:
                    print(f"Failed with model {model_name}: {str(model_error)}")
                    continue
            
            # If all models fail, try to list available models
            try:
                models = genai.list_models()
                available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                return f"""AI insights unavailable due to model compatibility issues.
                
Available models: {', '.join(available_models[:5])}

Please update the model name in the code to one of the available models.

Based on the data analysis:
- Account age: {user_summary['account_age_days']} days
- Total karma: {user_summary['total_karma']}
- Top subreddits: {', '.join(user_summary['top_subreddits'][:5])}
- Posting style: {user_summary['activity_stats']['posting_ratio']:.1f}% posts, {user_summary['activity_stats']['commenting_ratio']:.1f}% comments
                """
            except Exception as list_error:
                return f"""Error generating AI insights: Could not access Gemini models.

Error details: {str(list_error)}

Manual analysis based on data:
- Account age: {user_summary['account_age_days']} days
- Total karma: {user_summary['total_karma']}
- Most active subreddits: {', '.join(user_summary['top_subreddits'][:5])}
- Activity ratio: {user_summary['activity_stats']['posting_ratio']:.1f}% posts vs {user_summary['activity_stats']['commenting_ratio']:.1f}% comments
- Average post score: {user_summary['avg_scores']['posts']:.1f}
- Average comment score: {user_summary['avg_scores']['comments']:.1f}

To fix AI insights:
1. Check your Gemini API key
2. Verify your account has access to Gemini models
3. Try updating the google-generativeai library: pip install --upgrade google-generativeai
                """
                
        except Exception as e:
            return f"""Error generating AI insights: {str(e)}

This could be due to:
1. Invalid API key
2. Rate limit exceeded  
3. Model availability issues
4. Network connectivity problems

Please check your GEMINI_API_KEY in the .env file and ensure you have access to the Gemini API.
            """
    
    def generate_full_report(self, username: str, output_dir: str = 'reddit_analysis') -> Dict[str, str]:
        """Generate a complete analysis report with all visualizations"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting analysis for u/{username}...")
        
        # Fetch data
        user_data = self.fetch_user_data(username)
        if not user_data:
            return {"error": "Failed to fetch user data"}
        
        # Analyze activity
        self.activity_data = self.analyze_activity_patterns()
        
        # Generate visualizations
        print("Generating visualizations...")
        wordcloud_path = os.path.join(output_dir, 'wordcloud.png')
        network_path = os.path.join(output_dir, 'interest_network.html')
        dashboard_path = os.path.join(output_dir, 'dashboard.html')
        content_highlights_path = os.path.join(output_dir, 'content_highlights.html')
        
        self.generate_word_cloud(wordcloud_path)
        self.create_interest_network(network_path)
        self.create_activity_dashboard(dashboard_path)
        self.create_content_highlights_html(content_highlights_path)
        
        # Generate AI insights
        print("Generating AI insights...")
        ai_insights = self.generate_ai_insights()
        
        # Create summary report
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Reddit Profile Analysis Report\n")
            f.write(f"## User: u/{username}\n\n")
            f.write(f"### Profile Stats\n")
            f.write(f"- Total Karma: {user_data.get('total_karma', 0)}\n")
            f.write(f"- Post Karma: {user_data.get('link_karma', 0)}\n")
            f.write(f"- Comment Karma: {user_data.get('comment_karma', 0)}\n")
            f.write(f"- Account Age: {user_data.get('account_age_days', 0)} days\n")
            f.write(f"- Total Posts: {self.activity_data['activity_stats']['total_posts']}\n")
            f.write(f"- Total Comments: {self.activity_data['activity_stats']['total_comments']}\n\n")
            
            f.write(f"### Top Subreddits\n")
            for sub, count in list(self.activity_data['top_subreddits'].items())[:10]:
                f.write(f"- r/{sub}: {count} activities\n")
            
            f.write(f"\n### Content Highlights\n")
            if self.activity_data['top_content']['top_post']:
                top_post = self.activity_data['top_content']['top_post']
                f.write(f"\n🏆 **Top Post**\n")
                f.write(f"- **Title:** \"{top_post['title']}\"\n")
                f.write(f"- **Subreddit:** r/{top_post['subreddit']}\n")
                f.write(f"- **Upvotes:** {top_post['score']:,}\n")
                if top_post['selftext']:
                    preview = top_post['selftext'][:100] + "..." if len(top_post['selftext']) > 100 else top_post['selftext']
                    f.write(f"- **Preview:** \"{preview}\"\n")
            
            if self.activity_data['top_content']['top_comment']:
                top_comment = self.activity_data['top_content']['top_comment']
                f.write(f"\n💬 **Top Comment**\n")
                f.write(f"- **Thread Title:** \"{top_comment['submission_title']}\"\n")
                f.write(f"- **Subreddit:** r/{top_comment['subreddit']}\n")
                f.write(f"- **Upvotes:** {top_comment['score']:,}\n")
                comment_preview = top_comment['body'][:100] + "..." if len(top_comment['body']) > 100 else top_comment['body']
                f.write(f"- **Comment:** \"{comment_preview}\"\n")
            
            if self.activity_data['top_content']['worst_post'] and self.activity_data['top_content']['worst_post']['score'] < 0:
                worst_post = self.activity_data['top_content']['worst_post']
                f.write(f"\n👎 **Most Downvoted Post**\n")
                f.write(f"- **Title:** \"{worst_post['title']}\"\n")
                f.write(f"- **Subreddit:** r/{worst_post['subreddit']}\n")
                f.write(f"- **Downvotes:** {worst_post['score']:,}\n")
                if worst_post['selftext']:
                    preview = worst_post['selftext'][:100] + "..." if len(worst_post['selftext']) > 100 else worst_post['selftext']
                    f.write(f"- **Preview:** \"{preview}\"\n")
            
            if self.activity_data['top_content']['worst_comment'] and self.activity_data['top_content']['worst_comment']['score'] < 0:
                worst_comment = self.activity_data['top_content']['worst_comment']
                f.write(f"\n💬 **Most Downvoted Comment**\n")
                f.write(f"- **Thread Title:** \"{worst_comment['submission_title']}\"\n")
                f.write(f"- **Subreddit:** r/{worst_comment['subreddit']}\n")
                f.write(f"- **Downvotes:** {worst_comment['score']:,}\n")
                comment_preview = worst_comment['body'][:100] + "..." if len(worst_comment['body']) > 100 else worst_comment['body']
                f.write(f"- **Comment:** \"{comment_preview}\"\n")
            
            f.write(f"\n### AI-Generated Insights\n")
            f.write(ai_insights)
            
            if self.activity_data['top_content']['top_post']:
                top_post = self.activity_data['top_content']['top_post']
                f.write(f"\n### Top Post\n")
                f.write(f"- Title: {top_post['title']}\n")
                f.write(f"- Subreddit: r/{top_post['subreddit']}\n")
                f.write(f"- Score: {top_post['score']}\n")
        
        print(f"Analysis complete! Files saved to {output_dir}/")
        
        return {
            'report': report_path,
            'wordcloud': wordcloud_path,
            'network': network_path,
            'dashboard': dashboard_path,
            'content_highlights': content_highlights_path,
            'status': 'success'
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RedditAnalyzer()
    
    # Analyze a user (replace with actual username)
    username = input("Enter Reddit username (without u/): ").strip()
    
    if username:
        results = analyzer.generate_full_report(username)
        
        if 'error' not in results:
            print("\n✅ Analysis completed successfully!")
            print(f"📁 Files generated:")
            for key, path in results.items():
                if key != 'status':
                    print(f"   - {key.title()}: {path}")
        else:
            print(f"❌ {results['error']}")
    else:
        print("Please provide a valid username.")