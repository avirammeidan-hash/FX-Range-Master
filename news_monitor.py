"""
news_monitor.py -- Real-time news sentiment monitor for USD/ILS trading.

Polls free RSS feeds and news APIs for market-moving headlines,
scores sentiment for USD/ILS impact, and alerts the dashboard.

Supports:
  - RSS feeds (Reuters, CNBC, MarketWatch, Globes, Calcalist)
  - NewsAPI.org (optional, requires API key in config.yaml)
  - Keyword-based sentiment scoring tuned for USD/ILS

No API key needed for RSS feeds -- works out of the box.
"""

import re
import time
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from typing import Optional


# -- RSS feeds (no API key needed) --------------------------------------------

RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "MarketWatch Top": "http://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Currencies": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Forex Live": "https://www.forexlive.com/feed/",
}

# -- Keyword scoring for USD/ILS impact --------------------------------------

# Keywords that strengthen USD (weaken ILS) -> positive score
USD_BULLISH = {
    # Fed / monetary
    "rate hike": 3, "hawkish": 3, "tightening": 2, "fed funds": 2,
    "inflation hot": 3, "inflation higher": 3, "cpi beat": 3,
    "strong jobs": 2, "nfp beat": 3, "payrolls beat": 3,
    "gdp beat": 2, "gdp strong": 2,
    # Geopolitical -> safe haven USD
    "tariff": 3, "tariffs": 3, "trade war": 3, "sanctions": 3,
    "escalation": 2, "missile": 2, "attack": 2, "war": 2,
    "iran": 2, "hezbollah": 2, "hamas": 2,
    "risk off": 2, "risk-off": 2, "flight to safety": 3,
    # Dollar strength
    "dollar rally": 3, "dollar surge": 3, "dollar strength": 2,
    "dxy higher": 2, "greenback": 1,
    # Israel negative
    "israel downgrade": 3, "shekel weak": 3, "boi cut": 2,
    "israel deficit": 2, "israel debt": 2,
    # Trump specific
    "trump tariff": 4, "trump trade": 3, "trump truth": 2,
    "truth social": 2, "executive order": 2,
}

# Keywords that weaken USD (strengthen ILS) -> negative score
USD_BEARISH = {
    # Fed dovish
    "rate cut": -3, "dovish": -3, "easing": -2, "pivot": -3,
    "pause": -2, "skip": -1,
    "inflation cool": -3, "inflation lower": -3, "cpi miss": -3,
    "weak jobs": -2, "nfp miss": -3, "payrolls miss": -3,
    # Dollar weakness
    "dollar fall": -3, "dollar drop": -3, "dollar weak": -2,
    "dxy lower": -2,
    # Israel positive
    "shekel strong": -3, "boi hike": -2,
    "israel growth": -2, "israel upgrade": -3,
    "ceasefire": -2, "peace": -1,
    # Risk on
    "risk on": -2, "risk-on": -2, "rally": -1,
}

# High-impact source keywords (boost score)
HIGH_IMPACT_SOURCES = ["reuters", "bloomberg", "cnbc", "federal reserve", "trump"]


def _fetch_rss(url: str, timeout: int = 10) -> list[dict]:
    """Fetch and parse an RSS feed, return list of items."""
    items = []
    try:
        req = Request(url, headers={"User-Agent": "FX-Range-Master/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        root = ET.fromstring(data)
        # Handle RSS 2.0 and Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for item in root.iter("item"):
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pub = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            items.append({
                "title": title.strip(),
                "description": _clean_html(desc.strip())[:300],
                "published": pub.strip(),
                "link": link.strip(),
            })

        # Atom format fallback
        if not items:
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                title = entry.findtext("{http://www.w3.org/2005/Atom}title", "")
                summary = entry.findtext("{http://www.w3.org/2005/Atom}summary", "")
                updated = entry.findtext("{http://www.w3.org/2005/Atom}updated", "")
                link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                link = link_el.get("href", "") if link_el is not None else ""
                items.append({
                    "title": title.strip(),
                    "description": _clean_html(summary.strip())[:300],
                    "published": updated.strip(),
                    "link": link.strip(),
                })

    except (URLError, ET.ParseError, Exception):
        pass

    return items


def _clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def _score_text(text: str) -> tuple[int, list[str]]:
    """Score a text for USD/ILS sentiment. Returns (score, matched_keywords)."""
    text_lower = text.lower()
    score = 0
    matched = []

    for kw, val in {**USD_BULLISH, **USD_BEARISH}.items():
        if kw in text_lower:
            score += val
            matched.append(kw)

    # Boost if from high-impact source
    for src in HIGH_IMPACT_SOURCES:
        if src in text_lower:
            score = int(score * 1.5)
            break

    return score, matched


def _item_id(item: dict) -> str:
    """Generate unique ID for a news item."""
    raw = (item.get("title", "") + item.get("link", "")).encode()
    return hashlib.md5(raw).hexdigest()[:12]


class NewsMonitor:
    """Monitors news feeds and scores sentiment for USD/ILS impact."""

    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self._seen: set[str] = set()
        self._alerts: list[dict] = []
        self._last_poll: Optional[datetime] = None
        self._poll_interval = timedelta(minutes=5)

    def poll(self) -> list[dict]:
        """
        Poll all news sources. Returns new high-impact alerts since last poll.
        Automatically deduplicates.
        """
        now = datetime.now()
        if self._last_poll and (now - self._last_poll) < self._poll_interval:
            return []

        self._last_poll = now
        new_alerts = []

        # Poll RSS feeds
        for source, url in RSS_FEEDS.items():
            items = _fetch_rss(url)
            for item in items[:10]:  # Only check recent items
                item_id = _item_id(item)
                if item_id in self._seen:
                    continue
                self._seen.add(item_id)

                # Score the headline + description
                full_text = item["title"] + " " + item.get("description", "")
                score, keywords = _score_text(full_text)

                if abs(score) >= 3:  # Only alert on significant items
                    alert = {
                        "time": now.strftime("%H:%M:%S"),
                        "source": source,
                        "title": item["title"][:120],
                        "score": score,
                        "impact": "USD+" if score > 0 else "USD-",
                        "keywords": keywords[:5],
                        "severity": "HIGH" if abs(score) >= 6 else "MEDIUM",
                        "link": item.get("link", ""),
                    }
                    new_alerts.append(alert)
                    self._alerts.append(alert)

        # Poll NewsAPI if key available
        if self.newsapi_key:
            newsapi_alerts = self._poll_newsapi()
            new_alerts.extend(newsapi_alerts)

        # Keep last 100 alerts
        self._alerts = self._alerts[-100:]
        # Keep seen set manageable
        if len(self._seen) > 5000:
            self._seen = set(list(self._seen)[-2000:])

        return new_alerts

    def _poll_newsapi(self) -> list[dict]:
        """Poll NewsAPI.org for USD/ILS relevant news."""
        alerts = []
        queries = [
            "USD ILS shekel",
            "trump tariff trade",
            "federal reserve rate",
            "israel economy",
        ]

        for q in queries:
            try:
                url = (
                    f"https://newsapi.org/v2/everything?"
                    f"q={q.replace(' ', '+')}&"
                    f"sortBy=publishedAt&pageSize=5&"
                    f"apiKey={self.newsapi_key}"
                )
                req = Request(url, headers={"User-Agent": "FX-Range-Master/1.0"})
                with urlopen(req, timeout=10) as resp:
                    import json
                    data = json.loads(resp.read())

                for article in data.get("articles", []):
                    title = article.get("title", "")
                    desc = article.get("description", "")
                    item_id = hashlib.md5(title.encode()).hexdigest()[:12]

                    if item_id in self._seen:
                        continue
                    self._seen.add(item_id)

                    full_text = title + " " + (desc or "")
                    score, keywords = _score_text(full_text)

                    if abs(score) >= 3:
                        alert = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "source": "NewsAPI",
                            "title": title[:120],
                            "score": score,
                            "impact": "USD+" if score > 0 else "USD-",
                            "keywords": keywords[:5],
                            "severity": "HIGH" if abs(score) >= 6 else "MEDIUM",
                            "link": article.get("url", ""),
                        }
                        alerts.append(alert)
                        self._alerts.append(alert)

            except Exception:
                pass

            time.sleep(0.5)  # Rate limit

        return alerts

    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        """Get the N most recent alerts."""
        return self._alerts[-n:]

    def get_sentiment_summary(self) -> dict:
        """Get aggregate sentiment from recent alerts."""
        recent = self._alerts[-20:]  # Last 20 alerts
        if not recent:
            return {"sentiment": "NEUTRAL", "score": 0, "alert_count": 0}

        total_score = sum(a["score"] for a in recent)
        high_count = sum(1 for a in recent if a["severity"] == "HIGH")

        if total_score > 5:
            sentiment = "USD_BULLISH"
        elif total_score < -5:
            sentiment = "USD_BEARISH"
        else:
            sentiment = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "score": total_score,
            "alert_count": len(recent),
            "high_impact_count": high_count,
            "latest": recent[-1] if recent else None,
        }

    def force_poll(self):
        """Force a poll regardless of interval."""
        self._last_poll = None
        return self.poll()
