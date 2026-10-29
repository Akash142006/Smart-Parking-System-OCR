import json
import os
import time

import requests


def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

REMOTE_DB_URL = os.getenv("REMOTE_DB_URL", "").rstrip("/")
REMOTE_DB_KEY = os.getenv("REMOTE_DB_KEY", "")
_REMOTE_SESSION = None


def _normalize_plate(plate_number):
    return str(plate_number or "").upper().strip().replace(" ", "").replace("-", "")


def _assert_remote_config():
    if not REMOTE_DB_URL or not REMOTE_DB_KEY:
        raise RuntimeError("Remote DB is required. Set REMOTE_DB_URL and REMOTE_DB_KEY.")


def _get_session():
    global _REMOTE_SESSION
    if _REMOTE_SESSION is None:
        _REMOTE_SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        _REMOTE_SESSION.mount("https://", adapter)
        _REMOTE_SESSION.mount("http://", adapter)
    return _REMOTE_SESSION


def _remote_request(method, path, query=None, body=None):
    _assert_remote_config()
    url = f"{REMOTE_DB_URL}{path}"

    headers = {
        "apikey": REMOTE_DB_KEY,
        "Authorization": f"Bearer {REMOTE_DB_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }

    timeout_raw = str(os.getenv("REMOTE_DB_TIMEOUT", "0")).strip().lower()
    timeout_seconds = None if timeout_raw in {"", "0", "none", "off", "false"} else float(timeout_raw)
    retry_count = int(os.getenv("REMOTE_DB_RETRIES", "3"))
    retry_backoff = float(os.getenv("REMOTE_DB_RETRY_BACKOFF", "1.0"))

    session = _get_session()

    last_error = None
    for attempt in range(retry_count + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                params=query,
                data=json.dumps(body) if body is not None else None,
                headers=headers,
                timeout=timeout_seconds,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Remote DB HTTP error {response.status_code}. {response.text}")

            raw = response.text.strip()
            return json.loads(raw) if raw else []
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_backoff * (attempt + 1))
                continue
            reason = str(exc)
            timeout_label = "disabled" if timeout_seconds is None else f"{timeout_seconds}s"
            raise RuntimeError(
                f"Cannot connect to remote DB after {retry_count + 1} attempt(s). "
                f"Timeout={timeout_label}. Reason: {reason}. "
                "Check internet connection, firewall/proxy, and Supabase region reachability."
            ) from exc
        except RuntimeError:
            raise

    if last_error is not None:
        raise RuntimeError(f"Cannot connect to remote DB: {last_error}") from last_error


def init_db():
    _assert_remote_config()
    print("Remote DB configuration loaded")


def check_plate(plate_number):
    plate = _normalize_plate(plate_number)
    rows = _remote_request(
        "GET",
        "/rest/v1/registered_plates",
        {
            "select": "id,plate_number,owner_name",
            "plate_number": f"eq.{plate}",
            "limit": "1",
        },
    )
    if rows:
        row = rows[0]
        return {"found": True, "plate": row["plate_number"], "owner": row["owner_name"]}
    return {"found": False, "plate": plate, "owner": None}


def get_all_plates():
    rows = _remote_request(
        "GET",
        "/rest/v1/registered_plates",
        {"select": "id,plate_number,owner_name", "order": "id.asc"},
    )
    return [
        {"id": row.get("id"), "plate": row.get("plate_number"), "owner": row.get("owner_name")}
        for row in rows
    ]


def add_plate(plate_number, owner_name):
    plate = _normalize_plate(plate_number)
    owner = str(owner_name or "").strip()

    if not plate:
        return {"success": False, "message": "Plate number is required"}
    if not owner:
        return {"success": False, "message": "Owner name is required"}

    existing = _remote_request(
        "GET",
        "/rest/v1/registered_plates",
        {"select": "id", "plate_number": f"eq.{plate}", "limit": "1"},
    )
    if existing:
        return {"success": False, "message": "Plate already exists"}

    _remote_request(
        "POST",
        "/rest/v1/registered_plates",
        None,
        {"plate_number": plate, "owner_name": owner},
    )
    return {"success": True, "message": "Plate added successfully"}


def add_plates_bulk(raw_lines, default_owner="Bulk Upload"):
    lines = str(raw_lines or "").splitlines()
    saved = 0
    skipped = 0
    for line in lines:
        plate = _normalize_plate(line)
        if not plate:
            skipped += 1
            continue
        try:
            result = add_plate(plate, default_owner)
            if result.get("success"):
                saved += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    return {
        "success": True,
        "message": f"Bulk upload complete. Saved: {saved}, Skipped: {skipped}",
        "saved": saved,
        "skipped": skipped,
    }


def delete_plate(plate_number):
    plate = _normalize_plate(plate_number)

    _remote_request(
        "DELETE",
        "/rest/v1/registered_plates",
        {"plate_number": f"eq.{plate}"},
    )
    return {"success": True, "message": "Plate deleted"}