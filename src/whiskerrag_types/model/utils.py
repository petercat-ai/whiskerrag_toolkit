from datetime import datetime, timezone

from dateutil import parser


def parse_datetime(value: str) -> datetime:
    try:
        dt: datetime = parser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as e:
        raise ValueError(f"Invalid datetime format: {value}") from e


__all__ = ["parse_datetime"]
