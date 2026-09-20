#!/usr/bin/env python3
"""Scrape recent Euronews articles into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

DEFAULT_SITEMAP = "https://www.euronews.com/sitemap/articles.xml"
DEFAULT_DB = Path("data/news.db")
USER_AGENT = "nlp-scraper/1.0 (+https://github.com/legion2440/nlp-scraper)"


@dataclass(frozen=True)
class Article:
    article_id: str
    url: str
    date_scraped: str
    date_published: str | None
    headline: str
    body: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def request_text(session: requests.Session, url: str, timeout: int = 20) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def iter_json_objects(value: object) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_objects(child)


def clean_inline_text(value: str | None) -> str:
    return " ".join((value or "").split())


def extract_article(html: str, url: str) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    headline: str | None = None
    body: str | None = None
    published: str | None = None

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in iter_json_objects(payload):
            article_type = obj.get("@type")
            if isinstance(article_type, list):
                is_article = any(
                    t in {"NewsArticle", "Article", "ReportageNewsArticle"}
                    for t in article_type
                )
            else:
                is_article = article_type in {
                    "NewsArticle",
                    "Article",
                    "ReportageNewsArticle",
                }
            if not is_article:
                continue
            headline = headline or obj.get("headline")
            body = body or obj.get("articleBody")
            published = published or obj.get("datePublished")

    if not headline:
        h1 = soup.find("h1")
        headline = h1.get_text(" ", strip=True) if h1 else None

    if not body:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.select('article p, [data-testid="paragraph"]')
            if p.get_text(" ", strip=True)
        ]
        body = "\n".join(dict.fromkeys(paragraphs))

    headline = clean_inline_text(headline)
    body = (body or "").strip()
    if not headline or len(body) < 200:
        return None

    return Article(
        article_id=str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
        url=url,
        date_scraped=utc_now().isoformat(),
        date_published=published,
        headline=headline,
        body=body,
    )


def parse_sitemap(xml_text: str) -> tuple[str, list[tuple[str, str | None]]]:
    root = ET.fromstring(xml_text)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"

    if root.tag.endswith("sitemapindex"):
        entries = []
        for node in root.findall(f"{namespace}sitemap"):
            loc = node.findtext(f"{namespace}loc")
            lastmod = node.findtext(f"{namespace}lastmod")
            if loc:
                entries.append((loc.strip(), lastmod.strip() if lastmod else None))
        return "index", entries

    entries = []
    for node in root.findall(f"{namespace}url"):
        loc = node.findtext(f"{namespace}loc")
        lastmod = node.findtext(f"{namespace}lastmod")
        if loc:
            entries.append((loc.strip(), lastmod.strip() if lastmod else None))
    return "urls", entries


def discover_recent_urls(
    session: requests.Session,
    sitemap_url: str,
    cutoff: datetime,
) -> list[str]:
    kind, entries = parse_sitemap(request_text(session, sitemap_url))
    if kind == "urls":
        return [
            url
            for url, modified in entries
            if not modified or (parse_datetime(modified) or cutoff) >= cutoff
        ]

    child_sitemaps = [
        url
        for url, modified in entries
        if not modified or (parse_datetime(modified) or cutoff) >= cutoff
    ]

    urls: list[str] = []
    for child_url in child_sitemaps:
        try:
            child_kind, child_entries = parse_sitemap(
                request_text(session, child_url)
            )
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"Skipping sitemap {child_url}: {exc}")
            continue
        if child_kind != "urls":
            continue
        for article_url, modified in child_entries:
            modified_at = parse_datetime(modified)
            if modified_at is None or modified_at >= cutoff:
                urls.append(article_url)

    return list(dict.fromkeys(urls))


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            article_id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            date_scraped TEXT NOT NULL,
            date_published TEXT,
            headline TEXT NOT NULL,
            body TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def stored_urls(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT url FROM articles")}


def insert_article(connection: sqlite3.Connection, article: Article) -> bool:
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO articles
            (article_id, url, date_scraped, date_published, headline, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            article.article_id,
            article.url,
            article.date_scraped,
            article.date_published,
            article.headline,
            article.body,
        ),
    )
    connection.commit()
    return cursor.rowcount == 1


def scrape(args: argparse.Namespace) -> int:
    cutoff = utc_now() - timedelta(days=args.days)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    )

    print(
        f"Discovering articles newer than {cutoff.date()} from {args.sitemap}"
    )
    try:
        urls = discover_recent_urls(session, args.sitemap, cutoff)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"Failed to load sitemap {args.sitemap}: {exc}")
        return 2

    video_urls = [url for url in urls if "/video/" in url]
    urls = [url for url in urls if "/video/" not in url]
    print(
        f"Discovered {len(urls)} candidate URLs "
        f"after excluding {len(video_urls)} video pages"
    )

    connection = initialize_database(args.db)
    existing = stored_urls(connection)
    target_new = max(0, args.limit - len(existing))
    saved = 0

    if target_new == 0:
        connection.close()
        print(f"Database already contains at least {args.limit} articles")
        return 0

    try:
        for index, url in enumerate(urls, start=1):
            if url in existing:
                continue
            if saved >= target_new:
                break

            print(f"{index}. scraping {url}")
            try:
                html = request_text(session, url)
                article = extract_article(html, url)
            except requests.RequestException as exc:
                print(f"   request failed: {exc}")
                continue

            if article is None:
                print("   skipped: article content could not be extracted")
                continue

            published_at = parse_datetime(article.date_published)
            if published_at is not None and published_at < cutoff:
                print("   skipped: article is older than the requested window")
                continue

            if insert_article(connection, article):
                saved += 1
                existing.add(url)
                total_progress = len(existing)
                print(f"   saved ({total_progress}/{args.limit})")

            if args.delay:
                time.sleep(args.delay)
    finally:
        total = connection.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]
        connection.close()

    print(
        f"Saved {saved} new articles; database contains {total} articles"
    )
    return 0 if total >= args.limit else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sitemap",
        default=DEFAULT_SITEMAP,
        help="News sitemap or sitemap index URL",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="SQLite database path"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Only consider articles from this many recent days",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Target number of stored articles",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay between article requests in seconds",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(scrape(build_parser().parse_args()))
