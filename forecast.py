import time
from typing import List

def summarize_tag(data, tag):
    # Extract values for the tag, ignoring missing ones
    values = [item[tag] for item in data if tag in item and item[tag] is not None]

    if not values:
        return {"min": None, "max": None, "mean": None}

    end_time = time.time()

    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2)
    }


def summarize_multiple_tags(data, tags: List[str]):

    print({tag: summarize_tag(data, tag) for tag in tags})

    return {tag: summarize_tag(data, tag) for tag in tags}