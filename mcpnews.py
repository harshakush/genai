import feedparser
import re
import requests

RSS_FEEDS = {
    "RT": {
        "url": "https://www.rt.com/rss/news/",
        "country": "Russia"
    },
    "Al Jazeera": {
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "country": "Qatar"
    },
    "CNN": {
        "url": "http://rss.cnn.com/rss/edition.rss",
        "country": "USA"
    },
    "Fox News": {
        "url": "http://feeds.foxnews.com/foxnews/latest",
        "country": "USA"
    },
    "BBC News": {
        "url": "http://feeds.bbci.co.uk/news/rss.xml",
        "country": "UK"
    },
    "The Guardian": {
        "url": "https://www.theguardian.com/world/rss",
        "country": "UK"
    },
    "Reuters": {
        "url": "http://feeds.reuters.com/reuters/topNews",
        "country": "UK"
    },
    "Deutsche Welle": {
        "url": "https://rss.dw.com/rdf/rss-en-all",
        "country": "Germany"
    },
    "France 24": {
        "url": "https://www.france24.com/en/rss",
        "country": "France"
    },
    "ABC News (Australia)": {
        "url": "https://www.abc.net.au/news/feed/51120/rss.xml",
        "country": "Australia"
    },
    "NDTV": {
        "url": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "country": "India"
    },
    "The Times of India": {
        "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "country": "India"
    },
    "China Daily": {
        "url": "http://www.chinadaily.com.cn/rss/china_rss.xml",
        "country": "China"
    },
    "Japan Times": {
        "url": "https://www.japantimes.co.jp/feed/",
        "country": "Japan"
    },
    "The New York Times": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "country": "USA"
    },
    "Firstpost": {
        "url": "https://www.firstpost.com/feed/",
        "country": "India"
    }
}

def fetch_articles(feed_url, topic, max_articles=3):
    feed = feedparser.parse(feed_url)
    articles = []
    topic_pattern = re.compile(re.escape(topic), re.IGNORECASE)
    for entry in feed.entries:
        # print(feed)
        summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
        if topic_pattern.search(entry.title) or topic_pattern.search(summary):
            articles.append({
                "title": entry.title,
                "summary": summary,
                "link": entry.link
            })
        if len(articles) >= max_articles:
            break
    return articles

def build_mcp_context(topic, all_articles):
    return {
        "topic": topic,
        "articles": all_articles
    }

def build_prompt(context):
    articles = context['articles']
    topic = context['topic']
    if len(articles) == 1:
        article = articles[0]
        prompt = (
            f"Here is a news article about '{topic}':\n\n"
            f"Source: {article['source']} ({article['country']})\n"
            f"Title: {article['title']}\n"
            f"Summary: {article['summary']}\n"
            f"Link: {article['link']}\n\n"
            "Analyze the fairness and any noticeable bias in the article above. "
            "Does the language appear neutral or does it favor a particular perspective? "
            "Write a short paragraph with your analysis, citing specific phrases or examples."
        )
    else:
        prompt = f"Analyze the following news coverage on '{topic}' from multiple sources. "
        prompt += "Assess the fairness, bias, and any missing perspectives. Here are the articles:\n\n"
        for article in articles:
            prompt += (
                f"Source: {article['source']} ({article['country']})\n"
                f"Title: {article['title']}\n"
                f"Summary: {article['summary']}\n"
                f"Link: {article['link']}\n\n"
            )
        prompt += ( "Compare the coverage. Write a short paragraph summarizing the overall fairness and any noticeable bias, "
    "citing specific examples for each of the articles. "
    "Respond in JSON format with two keys: 'summary' (a paragraph) and 'articles' (a list of objects, each with 'newsoutlet', 'newsanalysis', and 'bias_level'). "
    "Here is an example of the expected JSON schema:\n"
    "{\n"
    '  "summary": "string",\n'
    '  "articles": [\n'
    '    {\n'
    '      "newsoutlet": "string",\n'
    '      "newsanalysis": "string",\n'
    '      "country_of_origin": "string",\n'
    '      "bias_level": "Neutral, Slightly Negative, Extreme Bias (Distraction)"\n'
    '    },\n'
    '    ...\n'
    '  ]\n'
    "}\n"
    "Please strictly follow this format in your response and dont not exclude any news outlet from the input in response and include atleast one response from each of the countries France, USA, Japan, China, India, United Kingdom, Germany, Australia, Russia for bias give one of the values Neutral, Slightly Negative, Extreme Bias (Distraction) and use one source one news for clarity")
       
    return prompt

def call_ollama_phi(prompt, model="gemma3:latest"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "")
    else:
        print("Error communicating with Ollama:", response.text)
        return ""

def get_mcp_analysis(topic, model="gemma3:latest"):
    all_articles = []
    for source, info in RSS_FEEDS.items():
        articles = fetch_articles(info['url'], topic)
        for article in articles:
            article['source'] = source
            article['country'] = info['country']
            all_articles.append(article)
            print(f"Found {len(all_articles)} articles from {source} on topic '{topic}'")

    if not all_articles:
        return None, f"No articles found for topic: {topic}"

    context = build_mcp_context(topic, all_articles)
    prompt = build_prompt(context)
    print(prompt)
    response = call_ollama_phi(prompt, model=model)
    return {
        "prompt": prompt,
        "response": response,
        "articles": all_articles
    }, None

# Example usage as a script
if __name__ == "__main__":
    topic = input("Enter the topic you want to analyze: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
    else:
        result, error = get_mcp_analysis(topic)
        if error:
            print(error)
        else:
            print("=== Prompt for LLM ===")
            print(result["prompt"])
            print("\n=== LLM Response ===")
            print(result["response"])
