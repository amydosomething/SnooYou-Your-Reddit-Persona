# SnooYou – Your Reddit Persona

Analyze any Reddit user and generate a detailed **persona profile** with visuals, insights, and highlights.  

---

## ✨ Features

### 👤 Reddit Persona Summary
Get a compact identity card for any Reddit user:
- Total karma (posts, comments)  
- Account age & activity level  
- Posting vs commenting breakdown (**pie chart**) 🥧  
- Engagement style (lurker, poster, chatterbox)  

---

### 🧠 AI-Generated Identity Snapshot
Suggests a Reddit alias (e.g., *TechNomad*)  
Infers:
- Age group (Gen Z / Millennial / Gen X)  
- Occupation / interests  
- Possible location (based on subs, language)
- Notable Patterns and Characteristics

---

### 📊 Visual Insights

#### 🔹 1. Interest Constellation Graph
- Force-directed graph of subreddit activity  
- Connected by topic similarity  
- Node size = activity level  

#### 🔹 2. Word Cloud
- Most-used words in posts/comments  
- Highlights passions, tone, quirks  

#### 🔹 3. Posts vs Comments Pie Chart 🥧
- Visualizes the ratio of posts vs comments  

#### 🔹 4. Top Subreddits Bar Chart 📊
- Shows the **Top 10 Subreddits** by activity level  

---

### 🏆 Content Highlights
- **Top Post**: Title, subreddit, upvotes, preview  
- **Top Comment**: From a high-upvote thread  
- **Most Downvoted Post/Comment**: Optional roast section 🔥  

---

## 🛠 Tech Stack
- **Frontend**: Streamlit  
- **Visuals**: matplotlib, networkx, wordcloud, plotly  
- **Data**: Pushshift API, PRAW (Reddit API), Gemini API


## 🧪 Try It Locally

```bash
git clone https://github.com/amydosomething/SnooYou-Your-Reddit-Persona.git
cd SnooYou-Your-Reddit-Persona
```

## 🔑 Environment Variables

Before running the project, create a `.env` file in the root directory and add the following variables:

```env
# Reddit API credentials
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Gemini API key
GEMINI_API_KEY=your_gemini_api_key
```
```
pip install -r requirements.txt
streamlit run reddit_analyzer_streamlit.py
```
