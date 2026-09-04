from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.yaml"


@dataclass
class Source:
    name: str
    source_type: str
    url: str
    category: str


def load_sources_config() -> dict:
    with open(SOURCES_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def discover_sources() -> List[Source]:
    config = load_sources_config()

    discovered = []

    for source_key, source_config in config["sources"].items():

        if not source_config.get("enabled", True):
            continue

        base_url = source_config.get("base_url", "")

        for category in source_config.get("categories", []):

            discovered.append(
                Source(
                    name=source_config["name"],
                    source_type=source_config["type"],
                    url=base_url,
                    category=category,
                )
            )

    return discovered


if __name__ == "__main__":

    sources = discover_sources()

    print(f"Discovered {len(sources)} source categories\n")

    for source in sources:
        print(
            f"{source.source_type:12} | "
            f"{source.category:30} | "
            f"{source.url}"
        )