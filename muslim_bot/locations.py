from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CITIES_CSV = Path(__file__).resolve().parent / "data" / "id_cities.csv"

# A place beyond this distance is not "near" a listed city, so its time zone is not borrowed from it.
NEAREST_CITY_MAX_KM = 300.0
MAX_CHOICES = 5

# What people type for places GeoNames lists under another name.
ALIASES = {
    "jogja": "yogyakarta",
    "jogjakarta": "yogyakarta",
    "yogya": "yogyakarta",
    "solo": "surakarta",
    "dki jakarta": "jakarta",
    "jakarta pusat": "jakarta",
    "jakarta selatan": "jakarta",
    "jakarta barat": "jakarta",
    "jakarta timur": "jakarta",
    "jakarta utara": "jakarta",
    "bau bau": "baubau",
    "pangkal pinang": "pangkalpinang",
}


@dataclass(frozen=True)
class City:
    name: str
    province: str
    latitude: float
    longitude: float
    timezone: str
    population: int

    @property
    def label(self) -> str:
        return f"{self.name}, {self.province}"


@lru_cache(maxsize=1)
def load_cities() -> tuple[City, ...]:
    with CITIES_CSV.open(encoding="utf-8", newline="") as handle:
        return tuple(
            City(
                row["name"],
                row["province"],
                float(row["latitude"]),
                float(row["longitude"]),
                row["timezone"],
                int(row["population"]),
            )
            for row in csv.DictReader(handle)
        )


def normalize(text: str) -> str:
    """Lowercase, no punctuation or "kota/kabupaten" prefix: "Kota Bandung" -> "bandung"."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("-", " "))
    words = cleaned.split()
    while words and words[0] in {"kota", "kabupaten", "kab", "kodya", "kotamadya", "di"}:
        words.pop(0)
    return " ".join(words)


@dataclass(frozen=True)
class CityMatch:
    """Result of a name lookup: one city, several to choose from, or none."""

    cities: tuple[City, ...] = ()

    @property
    def found(self) -> City | None:
        return self.cities[0] if len(self.cities) == 1 else None

    @property
    def ambiguous(self) -> bool:
        return len(self.cities) > 1


def find_city(query: str) -> CityMatch:
    """Look a place up by name, optionally with a province: "Depok, Jawa Barat".

    An exact name beats a partial one. Namesakes (Depok, Banjar, Tuban) are returned together so
    the caller can ask which one, instead of silently picking the wrong coordinates.
    """
    name_part, _, province_part = query.partition(",")
    name = normalize(name_part)
    name = ALIASES.get(name, name)
    if len(name) < 3:
        return CityMatch()
    province = normalize(province_part)
    cities = [city for city in load_cities() if not province or province in normalize(city.province)]

    for matches in (
        [city for city in cities if normalize(city.name) == name],
        [city for city in cities if normalize(city.name).startswith(name)],
        [city for city in cities if len(name) >= 4 and name in normalize(city.name)],
    ):
        if matches:
            ordered = sorted(matches, key=lambda city: -city.population)
            return CityMatch(tuple(ordered[:MAX_CHOICES]))
    return CityMatch()


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_city(latitude: float, longitude: float, max_km: float = NEAREST_CITY_MAX_KM) -> City | None:
    best: City | None = None
    best_km = max_km
    for city in load_cities():
        km = distance_km(latitude, longitude, city.latitude, city.longitude)
        if km <= best_km:
            best, best_km = city, km
    return best
