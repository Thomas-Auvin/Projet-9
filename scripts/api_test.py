from __future__ import annotations

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("P9_API_URL", "http://127.0.0.1:8000")


def main() -> None:
    q = "Quels concerts de jazz sont prévus en Nouvelle-Aquitaine ?"
    payload = {"question": q, "k": 6}

    r = httpx.post(f"{BASE_URL}/ask", json=payload, timeout=60.0)
    print("Status:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    main()
