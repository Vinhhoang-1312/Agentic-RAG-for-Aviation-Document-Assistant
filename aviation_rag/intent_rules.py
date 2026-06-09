from __future__ import annotations

import pandas as pd


def map_row_to_intent(row: pd.Series) -> str:
    """Map aviation dataset rows into Hoang's phase-1 intent taxonomy.

    Note: the local ASRS-style dataset is suitable for three practical labels:
    Incident_Report, Technical_Procedure, and Metadata_Query. Factoid examples
    are added as seed training rows before the TF-IDF + Logistic Regression fit.
    """

    primary_problem = str(row.get("primary_problem", "")).lower()
    weather_blob = " ".join(
        str(row.get(col, "")).lower()
        for col in [
            "weather_conditions",
            "flight_conditions",
            "report_summary",
            "report1_narrative",
        ]
    )
    technical_blob = " ".join(
        str(row.get(col, "")).lower()
        for col in [
            "component_name",
            "component_problem",
            "event_anomaly",
            "report_summary",
        ]
    )

    if any(
        token in primary_problem
        for token in ["procedure", "manuals", "policy", "mel", "logbook", "part"]
    ):
        return "Technical_Procedure"
    if any(
        token in technical_blob
        for token in [
            "procedure",
            "checklist",
            "manual",
            "maintenance",
            "mel",
            "logbook",
            "troubleshoot",
            "deferral",
            "deferred",
        ]
    ):
        return "Technical_Procedure"

    if any(
        token in primary_problem
        for token in ["weather", "turbulence", "icing", "rain", "snow", "fog", "wind", "storm"]
    ):
        return "Metadata_Query"
    if any(
        token in weather_blob
        for token in [
            "turbulence",
            "icing",
            "snow",
            "rain",
            "fog",
            "crosswind",
            "weather",
            "runway condition",
        ]
    ):
        return "Metadata_Query"

    return "Incident_Report"
