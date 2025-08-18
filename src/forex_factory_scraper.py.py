import time
import re
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import requests
from bs4 import BeautifulSoup
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(r"C:\Users\Babblu\OneDrive\Documents\GitHub\Stock_Analysis.AI\src\forex_factory_scraper.py"))))


from src.config import FF_BASE, SCRAPE_DELAY_SEC
from src.database import insert_fx_news

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.com/bot)"
}

def _day_url(day: datetime):
    # ForexFactory supports day-based calendar pages like /calendar?day=Aug16.2025
    # We'll format like "Aug16.2025"
    slug = day.strftime("%b%d.%Y")
    return f"{FF_BASE}/calendar?day={slug}"

def _normalize_text(t):
    return re.sub(r"\s+", " ", (t or "").strip())

def _parse_numeric(s):
    if s is None: return None
    s = s.replace(",", "").strip()
    # drop percent symbol
    s = s.replace("%", "")
    # handle "n/a" etc.
    if s.lower() in {"", "n/a", "na", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _impact_from_el(el):
    # Try by title/class; fallback text search
    cls = " ".join(el.get("class", []))
    if "impact" in cls and "high" in cls.lower(): return "High"
    if "impact" in cls and "medium" in cls.lower(): return "Medium"
    if "impact" in cls and "low" in cls.lower(): return "Low"
    # textual fallback
    text = el.get_text(" ", strip=True).lower()
    if "high" in text: return "High"
    if "medium" in text: return "Medium"
    if "low" in text: return "Low"
    return "Medium"

def _find_header_indices(header_row):
    # map column names -> index by fuzzy matching
    headers = [h.get_text(" ", strip=True).lower() for h in header_row.find_all(["th","td"])]
    def idx(names):
        for n in names:
            for i, h in enumerate(headers):
                if n in h:
                    return i
        return None
    return {
        "time": idx(["time"]),
        "currency": idx(["currency"]),
        "impact": idx(["impact"]),
        "event": idx(["event"]),
        "actual": idx(["actual"]),
        "forecast": idx(["forecast"]),
        "previous": idx(["previous"]),
    }

def scrape_day(day: datetime, tz="UTC", currency_filter="USD"):
    url = _day_url(day)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # find calendar table (structure can change; we stay resilient)
    table = soup.find("table")
    if not table:
        print(f"[WARN] No table found on {url}")
        return 0

    rows = table.find_all("tr")
    if not rows:
        print(f"[WARN] No rows found on {url}")
        return 0

    # find header row for indices
    header_idx = None
    for r in rows:
        ths = r.find_all("th")
        if len(ths) >= 5:
            header_idx = _find_header_indices(r)
            break
    if not header_idx:
        # fallback naive indices
        header_idx = {"time":0,"currency":1,"impact":2,"event":3,"actual":4,"forecast":5,"previous":6}

    inserted = 0
    current_time_str = None

    for r in rows:
        cells = r.find_all(["td","th"])
        if len(cells) < 4: 
            continue

        # keep track of time cells that sometimes span multiple events
        t_idx = header_idx["time"]
        if t_idx is not None and t_idx < len(cells):
            maybe_time = _normalize_text(cells[t_idx].get_text())
            if maybe_time:
                current_time_str = maybe_time

        # currency
        c_idx = header_idx["currency"]
        if c_idx is None or c_idx >= len(cells):
            continue
        currency = _normalize_text(cells[c_idx].get_text())
        if currency != currency_filter:
            continue

        # event
        e_idx = header_idx["event"]
        event = _normalize_text(cells[e_idx].get_text()) if e_idx is not None and e_idx < len(cells) else None
        if not event:
            continue

        # impact
        i_idx = header_idx["impact"]
        impact_el = cells[i_idx] if i_idx is not None and i_idx < len(cells) else None
        impact = _impact_from_el(impact_el) if impact_el else "Medium"

        a_idx = header_idx["actual"]; f_idx = header_idx["forecast"]; p_idx = header_idx["previous"]
        actual_txt   = _normalize_text(cells[a_idx].get_text()) if a_idx is not None and a_idx < len(cells) else ""
        forecast_txt = _normalize_text(cells[f_idx].get_text()) if f_idx is not None and f_idx < len(cells) else ""
        previous_txt = _normalize_text(cells[p_idx].get_text()) if p_idx is not None and p_idx < len(cells) else ""

        # time: combine page date + time text
        # ForexFactory time format varies; we'll parse liberally
        tm = None
        if current_time_str and current_time_str.lower() not in {"all day", "tentative"}:
            # e.g., "8:30am"
            ts = f"{day.strftime('%Y-%m-%d')} {current_time_str}"
            try:
                tm = dateparser.parse(ts, fuzzy=True)
            except Exception:
                tm = day
        else:
            tm = day

        # store
        insert_fx_news(
            currency=currency,
            event=event,
            impact=impact,
            actual=actual_txt,
            forecast=forecast_txt,
            previous=previous_txt,
            event_time=tm,
            source_url=url
        )
        inserted += 1

    time.sleep(SCRAPE_DELAY_SEC)
    print(f"[INFO] {inserted} USD events on {day.date()}")
    return inserted

def fetch_and_store_fx_news_history(start_date: str, end_date: str, currency="USD"):
    """
    Dates in 'YYYY-MM-DD' format (inclusive)
    """
    start = datetime.fromisoformat(start_date)
    end   = datetime.fromisoformat(end_date)
    d = start
    total = 0
    while d <= end:
        total += scrape_day(d, currency_filter=currency)
        d += timedelta(days=1)
    print(f"[INFO] Total events inserted: {total}")
    return total

if __name__ == "__main__":
    # Example: last 90 days
    today = datetime.utcnow().date()
    start = (today - timedelta(days=90)).isoformat()
    end = today.isoformat()
    fetch_and_store_fx_news_history(start, end, currency="USD")