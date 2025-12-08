"""
Coreference assignment for mentions.

Currently uses simple heuristics (same lowercase text and type) to cluster
mentions. Replace with a more sophisticated coref model when available.
"""

from collections import defaultdict
from typing import List

from clinical_kg.data_models import Mention, Turn

_CLUSTER_ID_TEMPLATE = "c{index:04d}"


def add_coref_clusters(mentions: List[Mention], turns: List[Turn]) -> List[Mention]:
    """
    Assign coref_cluster_id to mentions that refer to the same entity.
    """
    # Group by (type, normalized text)
    buckets = defaultdict(list)
    for mention in mentions:
        key = (mention.type.upper(), mention.text.lower())
        buckets[key].append(mention)

    clustered: List[Mention] = []
    cluster_counter = 1
    for mentions_in_bucket in buckets.values():
        cluster_id = _CLUSTER_ID_TEMPLATE.format(index=cluster_counter)
        cluster_counter += 1
        for m in mentions_in_bucket:
            m.coref_cluster_id = cluster_id
            clustered.append(m)

    return clustered
