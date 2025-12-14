"""
Print a JSON summary of generic FHIR resource types and their attributes using fhir.resources.

Default resources are kept generic: Patient (person), Practitioner (clinician),
Condition (disease/symptom), Observation (test/measure), Medication, MedicationRequest,
Procedure, and ServiceRequest. You can override the list with --resources.

Examples:
  python scripts/print_fhir_resource_attributes.py
  python scripts/print_fhir_resource_attributes.py --resources Patient Observation
"""

import argparse
import importlib
import json
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

DEFAULT_RESOURCES: Dict[str, str] = {
    "Patient": "fhir.resources.patient.Patient",
    "Practitioner": "fhir.resources.practitioner.Practitioner",
    "Condition": "fhir.resources.condition.Condition",
    "Observation": "fhir.resources.observation.Observation",
    "Medication": "fhir.resources.medication.Medication",
    "MedicationRequest": "fhir.resources.medicationrequest.MedicationRequest",
    "Procedure": "fhir.resources.procedure.Procedure",
    "ServiceRequest": "fhir.resources.servicerequest.ServiceRequest",
}


def _friendly_type(ann: Any) -> str:
    if ann is None:
        return "Any"
    origin = get_origin(ann)
    args = get_args(ann)
    if origin is None:
        if hasattr(ann, "__name__"):
            return ann.__name__
        if hasattr(ann, "__qualname__"):
            return ann.__qualname__
        return str(ann)
    if origin in (list, List):
        inner = ", ".join(_friendly_type(a) for a in args) or "Any"
        return f"List[{inner}]"
    if origin is tuple:
        inner = ", ".join(_friendly_type(a) for a in args) or "Any"
        return f"Tuple[{inner}]"
    if str(origin).endswith("Union"):
        inner = " | ".join(_friendly_type(a) for a in args) or "Any"
        return inner
    return str(origin)


def _is_list_annotation(ann: Any) -> bool:
    origin = get_origin(ann)
    return origin in (list, List, tuple)


def _load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _field_from_v2(name: str, field: Any) -> Dict[str, Any]:
    alias = getattr(field, "alias", name) or name
    description = getattr(field, "description", None)
    if not description:
        extra = getattr(field, "json_schema_extra", None) or {}
        description = extra.get("description")
    annotation = getattr(field, "annotation", None)
    required = False
    if hasattr(field, "is_required") and callable(field.is_required):
        required = bool(field.is_required())
    return {
        "name": alias,
        "type": _friendly_type(annotation),
        "required": required,
        "is_array": _is_list_annotation(annotation),
        "description": description,
    }


def _field_from_v1(name: str, field: Any) -> Dict[str, Any]:
    alias = getattr(field, "alias", name) or name
    description = getattr(getattr(field, "field_info", None), "description", None)
    typ = getattr(field, "outer_type_", None) or getattr(field, "type_", None)
    required = bool(getattr(field, "required", False))
    is_array = False
    try:
        from pydantic.fields import SHAPE_LIST, SHAPE_SEQUENCE, SHAPE_TUPLE, SHAPE_SET

        shape = getattr(field, "shape", None)
        is_array = shape in {SHAPE_LIST, SHAPE_SEQUENCE, SHAPE_TUPLE, SHAPE_SET}
    except Exception:
        pass
    return {
        "name": alias,
        "type": _friendly_type(typ),
        "required": required,
        "is_array": is_array,
        "description": description,
    }


def _attributes_for_class(cls) -> List[Dict[str, Any]]:
    if hasattr(cls, "model_fields"):  # pydantic v2
        fields = getattr(cls, "model_fields") or {}
        return [_field_from_v2(name, field) for name, field in fields.items()]
    if hasattr(cls, "__fields__"):  # pydantic v1
        fields = getattr(cls, "__fields__") or {}
        return [_field_from_v1(name, field) for name, field in fields.items()]
    return []


def summarize_resources(resource_names: List[str]) -> Dict[str, Any]:
    summary: List[Dict[str, Any]] = []
    for name in resource_names:
        path = DEFAULT_RESOURCES.get(name)
        if not path:
            summary.append({"resource": name, "error": "not in default generic set"})
            continue
        try:
            cls = _load_class(path)
            attributes = _attributes_for_class(cls)
            summary.append({"resource": name, "attributes": attributes})
        except Exception as exc:
            summary.append({"resource": name, "error": str(exc)})
    return {"resources": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Print generic FHIR resource attribute definitions as JSON using fhir.resources."
    )
    parser.add_argument(
        "--resources",
        nargs="+",
        help="Resource names to include (default: generic set: Patient, Practitioner, Condition, Observation, Medication, MedicationRequest, Procedure, ServiceRequest).",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    parser.add_argument(
        "--output",
        required=True,
        help="File path to write the JSON output.",
    )
    args = parser.parse_args()

    selected = args.resources or list(DEFAULT_RESOURCES.keys())
    result = summarize_resources(selected)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=args.indent)


if __name__ == "__main__":
    main()
