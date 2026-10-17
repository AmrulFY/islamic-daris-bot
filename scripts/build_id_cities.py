"""Build muslim_bot/data/id_cities.csv from GeoNames (Indonesian places with 5,000+ people).

Data: GeoNames (https://www.geonames.org), licensed CC BY 4.0. Keep the attribution in the README
when the CSV is shipped.

    python scripts/build_id_cities.py

The CSV is committed; run this only to refresh it.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
import zipfile
from pathlib import Path

CITIES_URL = "https://download.geonames.org/export/dump/cities5000.zip"
ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
OUTPUT = Path(__file__).resolve().parents[1] / "muslim_bot" / "data" / "id_cities.csv"

# GeoNames' English province names -> the names Indonesians use.
PROVINCES = {
    "Aceh": "Aceh",
    "North Sumatra": "Sumatera Utara",
    "West Sumatra": "Sumatera Barat",
    "Riau": "Riau",
    "Riau Islands": "Kepulauan Riau",
    "Jambi": "Jambi",
    "South Sumatra": "Sumatera Selatan",
    "Bangka–Belitung Islands": "Kepulauan Bangka Belitung",
    "Bengkulu": "Bengkulu",
    "Lampung": "Lampung",
    "Jakarta": "DKI Jakarta",
    "Banten": "Banten",
    "West Java": "Jawa Barat",
    "Central Java": "Jawa Tengah",
    "Yogyakarta": "DI Yogyakarta",
    "East Java": "Jawa Timur",
    "Bali": "Bali",
    "West Nusa Tenggara": "Nusa Tenggara Barat",
    "East Nusa Tenggara": "Nusa Tenggara Timur",
    "West Kalimantan": "Kalimantan Barat",
    "Central Kalimantan": "Kalimantan Tengah",
    "South Kalimantan": "Kalimantan Selatan",
    "East Kalimantan": "Kalimantan Timur",
    "North Kalimantan": "Kalimantan Utara",
    "North Sulawesi": "Sulawesi Utara",
    "Gorontalo": "Gorontalo",
    "Central Sulawesi": "Sulawesi Tengah",
    "West Sulawesi": "Sulawesi Barat",
    "South Sulawesi": "Sulawesi Selatan",
    "Southeast Sulawesi": "Sulawesi Tenggara",
    "Maluku": "Maluku",
    "North Maluku": "Maluku Utara",
    "Papua": "Papua",
    "West Papua": "Papua Barat",
    "Southwest Papua": "Papua Barat Daya",
    "Central Papua": "Papua Tengah",
    "Highland Papua": "Papua Pegunungan",
    "South Papua": "Papua Selatan",
}
INDONESIAN_ZONES = {"Asia/Jakarta", "Asia/Pontianak", "Asia/Makassar", "Asia/Jayapura"}
# GeoNames files a few western-Indonesia places under Vietnam's zone (same offset as WIB).
ZONE_FIXES = {"Asia/Ho_Chi_Minh": "Asia/Pontianak"}


def download(url: str, attempts: int = 3) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError:  # GeoNames occasionally resets connections; a retry usually works.
            if attempt == attempts:
                raise
            time.sleep(3 * attempt)
    raise AssertionError("unreachable")


def main() -> None:
    admin1: dict[str, str] = {}
    for line in download(ADMIN1_URL).decode("utf-8").splitlines():
        code, name, *_ = line.split("\t")
        if code.startswith("ID."):
            admin1[code.split(".", 1)[1]] = name

    with zipfile.ZipFile(io.BytesIO(download(CITIES_URL))) as archive:
        lines = archive.read("cities5000.txt").decode("utf-8").splitlines()

    rows = []
    for line in lines:
        fields = line.split("\t")
        if fields[8] != "ID":
            continue
        zone = ZONE_FIXES.get(fields[17], fields[17])
        if zone not in INDONESIAN_ZONES:
            raise SystemExit(f"unexpected time zone {zone!r} for {fields[1]}")
        if fields[10] not in admin1:  # no province on file: it could not be told apart from a namesake
            print(f"skipping {fields[1]} (no province)")
            continue
        province = PROVINCES[admin1[fields[10]]]
        rows.append((fields[1], province, round(float(fields[4]), 4), round(float(fields[5]), 4), zone, int(fields[14])))

    rows.sort(key=lambda row: (-row[5], row[0]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "province", "latitude", "longitude", "timezone", "population"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} places to {OUTPUT}")


if __name__ == "__main__":
    main()
