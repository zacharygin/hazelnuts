"""
Historical named-peril events for the hazelnut insurance contract.

These are manually researched binary events. Where historical frequency is too
sparse, expected loss uses ANNUAL_PROBABILITIES from trigger_params.yaml rather
than empirical frequency.

Sources consulted:
- EFB: EPPO (European and Mediterranean Plant Protection Organization) alert list;
  Nicese et al. 2021 "Spread of EFB in Europe"
- Export disruptions: Turkish Hazelnut Producers Association (FİSKOBİRLİK) records;
  Reuters/Bloomberg archives
- Bosporus: IMO incident reports; Bogazici University maritime incident database
"""
from __future__ import annotations

# EFB (Eastern Filbert Blight - Anisogramma anomala)
# Status: Not yet present in Turkey as of 2025. Present in North America
# and spreading through EU (Belgium, France, Germany confirmed since 2000s).
# Expected arrival in Turkey: difficult to estimate; 0.5% annual probability
# is used as a conservative floor (see trigger_params.yaml).
EFB_EVENTS: list[dict] = []
# If/when detected: append {'year': YYYY, 'description': '...', 'severity': 1.0}

# Turkish hazelnut export disruptions
# Research scope: interventions by Turkish government or major supply chain
# breaks affecting >10% of annual export volume during the export season (Aug-Dec)
EXPORT_DISRUPTION_EVENTS: list[dict] = [
    # 1996: FİSKOBİRLİK floor price collapse; government intervention in export quotas
    # Note: impact uncertain — included as partial event
    {
        "year": 1996,
        "description": "FİSKOBİRLİK price floor collapse; temporary export disruption",
        "severity": 0.5,
    },
    # 2004: EU food safety recall of Turkish hazelnuts (aflatoxin limits)
    {
        "year": 2004,
        "description": "EU aflatoxin-related restrictions on Turkish hazelnut imports",
        "severity": 0.75,
    },
    # No confirmed full-severity events in the primary backtest window (1990-2024)
    # that meet the contract definition of a major export disruption.
    # The 3% annual probability in config is a forward-looking estimate.
]

# Bosporus logistics disruptions (>30 days during August–December)
# The Bosporus is regulated by the 1936 Montreux Convention; Turkey cannot close
# it to commercial traffic. Historical closures are short (hours to days, typically
# due to fog or accidents). A 30+ day closure during the export season is extremely
# rare — no confirmed events in the modern era.
LOGISTICS_DISRUPTION_EVENTS: list[dict] = [
    # 1979: Independenta tanker collision — partial disruption for ~2 weeks
    # Does not meet the 30-day threshold.
    # No events found meeting the full contract definition.
]

# Combined registry
NAMED_EVENTS: dict[str, list[dict]] = {
    "efb_outbreak": EFB_EVENTS,
    "export_disruption": EXPORT_DISRUPTION_EVENTS,
    "logistics_disruption": LOGISTICS_DISRUPTION_EVENTS,
}


def get_events_for_year(peril: str, year: int) -> list[dict]:
    return [e for e in NAMED_EVENTS.get(peril, []) if e["year"] == year]


def fired(peril: str, year: int) -> bool:
    return len(get_events_for_year(peril, year)) > 0


def severity(peril: str, year: int) -> float:
    """Return max severity across all events for a peril in a given year (0-1)."""
    events = get_events_for_year(peril, year)
    return max((e["severity"] for e in events), default=0.0)
