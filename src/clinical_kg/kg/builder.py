"""
Utilities to turn grouped entities into knowledge-graph nodes.
"""

from typing import Any, Dict, List

from clinical_kg.data_models import Mention
from clinical_kg.kg.schema import NodeSchema, schema_for_entity_type


def _as_dict(entity: Any) -> Dict[str, Any]:
    if isinstance(entity, Mention):
        return entity.__dict__
    return entity if isinstance(entity, dict) else {}


def _filter_attributes(entity: Dict[str, Any], schema: NodeSchema) -> Dict[str, Any]:
    attrs = entity.get("attributes") or {}
    if not schema.attribute_options or not isinstance(attrs, dict):
        return {}
    return {k: v for k, v in attrs.items() if k in schema.attribute_options}


def build_nodes(entities: List[Any], encounter_id: str) -> List[Dict[str, Any]]:
    """
    Convert grouped entities (with ontology annotations) into KG node payloads.
    """
    nodes: List[Dict[str, Any]] = []
    for idx, entity in enumerate(entities):
        data = _as_dict(entity)
        etype = data.get("entity_type") or data.get("type")
        schema = schema_for_entity_type(etype)

        canonical = data.get("canonical_name") or data.get("text")
        node_id = data.get("entity_id") or data.get("id") or f"{encounter_id}_n{idx + 1:04d}"

        attributes = _filter_attributes(data, schema)
        # Ensure a name attribute is available for person nodes
        if schema.class_name == "Person" and canonical:
            attributes.setdefault("name", canonical)

        nodes.append(
            {
                "id": node_id,
                "encounter_id": encounter_id,
                "class": schema.class_name,
                "entity_type": etype,
                "shacl_shape_id": schema.shacl_shape.shape_id,
                "canonical_name": canonical,
                "ontology": data.get("ontology"),
                "ontology_strategy": data.get("ontology_strategy"),
                "turn_ids": data.get("turn_ids") or [],
                "mentions": data.get("mentions") or [],
                "attribute_options": schema.attribute_options,
                "attributes": attributes,
            }
        )

    return nodes
