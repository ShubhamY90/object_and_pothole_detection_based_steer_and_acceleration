import sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from filtering.region_filter import (
    Region,
    classify_region,
    is_in_near_half,
    filter_detections,
    select_primary_threat,
    FilterResult,
)

__all__ = [
    "Region",
    "classify_region",
    "is_in_near_half",
    "filter_detections",
    "select_primary_threat",
    "FilterResult",
]
