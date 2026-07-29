from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    fetched_at: str
    sha256: str
    text: str
    from_cache: bool


def fetch_text(
    url: str,
    cache_dir: Path,
    refresh: bool = False,
    attempts: int = 3,
) -> FetchResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    text_path = cache_dir / f"{key}.txt"
    meta_path = cache_dir / f"{key}.json"
    if not refresh and text_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return FetchResult(text=text_path.read_text(encoding="utf-8"), from_cache=True, **metadata)

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 LimitlessEnvironmentAudit/1.0"})
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                final_url = response.geturl()
                charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            metadata = {
                "url": url,
                "final_url": final_url,
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            text_path.write_text(text, encoding="utf-8")
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return FetchResult(text=text, from_cache=False, **metadata)
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")
