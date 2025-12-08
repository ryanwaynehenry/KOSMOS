#!/usr/bin/env python3

from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Import the lookup helper from your MySQL-backed terminology module

from ontology_test import lookup_official



def build_patient_encounter_graph() -> Graph:
    g = Graph()

    # Namespaces
    EX = Namespace("http://example.org/clinical#")
    SNOMED = Namespace("http://snomed.info/id/")
    LOINC = Namespace("http://loinc.org/")
    RXNORM = Namespace("http://rxnorm.info/id/")

    # Bind prefixes for nicer Turtle output
    g.bind("ex", EX)
    g.bind("snomed", SNOMED)
    g.bind("loinc", LOINC)
    g.bind("rxnorm", RXNORM)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Helper to create a terminology concept node using MySQL lookup,
    # with a safe fallback to a default code and label.
    def add_ontology_concept(
        ontology: str,
        mention: str,
        ns: Namespace,
        default_code: str,
        default_label: str,
    ):
        """
        ontology: "snomed", "loinc", or "rxnorm"
        mention:  text phrase to look up
        ns:       rdflib Namespace for this coding system
        default_code/default_label: used if lookup fails or lookup_official is unavailable
        """
        code = default_code
        label = default_label

        
        try:
            result = lookup_official(ontology, mention)
            if result is not None:
                label, code = result  # lookup_official returns (display, code)
        except Exception:
            # If anything goes wrong (no MySQL, bad connection, etc),
            # keep the defaults so the script still works.
            pass

        concept_node = ns[code]
        g.add((concept_node, RDFS.label, Literal(label)))
        return concept_node

    # Classes in your local model
    Patient = EX.Patient
    Encounter = EX.Encounter
    Condition = EX.Condition
    Observation = EX.Observation
    MedicationStatement = EX.MedicationStatement

    # Properties in your local model
    hasIdentifier = EX.hasIdentifier
    hasGivenName = EX.hasGivenName
    hasFamilyName = EX.hasFamilyName
    hasBirthDate = EX.hasBirthDate
    hasGender = EX.hasGender

    hasSubject = EX.hasSubject
    hasEncounter = EX.hasEncounter

    hasCode = EX.hasCode          # link to terminology concept node
    hasEffectiveDateTime = EX.hasEffectiveDateTime

    hasValueQuantity = EX.hasValueQuantity
    hasValue = EX.hasValue
    hasUnit = EX.hasUnit

    hasStatus = EX.hasStatus
    hasDosageText = EX.hasDosageText

    # Instances in your graph
    patient = EX.patient_Aisha
    encounter = EX.encounter_2023_09_07
    angina = EX.condition_angina
    obs_bp = EX.observation_bp
    med_atorvastatin = EX.medication_atorvastatin

    # Terminology concept nodes, with MySQL-backed lookup
    angina_concept = add_ontology_concept(
        ontology="snomed",
        mention="Angina pectoris",
        ns=SNOMED,
        default_code="194828000",
        default_label="Angina (disorder)",
    )

    bp_concept = add_ontology_concept(
        ontology="loinc",
        mention="Blood pressure panel with all children",
        ns=LOINC,
        default_code="85354-9",
        default_label="Blood pressure panel",
    )

    atorvastatin_concept = add_ontology_concept(
        ontology="rxnorm",
        mention="Atorvastatin 20 MG Oral Tablet",
        ns=RXNORM,
        default_code="83367",
        default_label="Atorvastatin 20 MG Oral Tablet",
    )

    # Patient
    g.add((patient, RDF.type, Patient))
    g.add((patient, hasIdentifier, Literal("PAT-12345")))
    g.add((patient, hasGivenName, Literal("Aisha")))
    g.add((patient, hasFamilyName, Literal("Patel")))
    g.add((patient, hasBirthDate, Literal("1960-05-10", datatype=XSD.date)))
    g.add((patient, hasGender, Literal("female")))

    # Encounter
    g.add((encounter, RDF.type, Encounter))
    g.add((encounter, hasSubject, patient))
    g.add(
        (
            encounter,
            hasEffectiveDateTime,
            Literal("2023-09-07T09:30:00", datatype=XSD.dateTime),
        )
    )

    # Condition: angina
    g.add((angina, RDF.type, Condition))
    g.add((angina, hasSubject, patient))
    g.add((angina, hasEncounter, encounter))
    g.add((angina, hasCode, angina_concept))

    # Observation: blood pressure
    g.add((obs_bp, RDF.type, Observation))
    g.add((obs_bp, hasSubject, patient))
    g.add((obs_bp, hasEncounter, encounter))
    g.add(
        (
            obs_bp,
            hasEffectiveDateTime,
            Literal("2023-09-07T09:35:00", datatype=XSD.dateTime),
        )
    )
    g.add((obs_bp, hasCode, bp_concept))

    # Simple combined BP value node
    bp_value = EX.bp_value
    g.add((obs_bp, hasValueQuantity, bp_value))
    g.add((bp_value, hasValue, Literal("135/85")))
    g.add((bp_value, hasUnit, Literal("mm[Hg]")))

    # Medication statement: atorvastatin
    g.add((med_atorvastatin, RDF.type, MedicationStatement))
    g.add((med_atorvastatin, hasSubject, patient))
    g.add((med_atorvastatin, hasEncounter, encounter))
    g.add((med_atorvastatin, hasStatus, Literal("active")))
    g.add((med_atorvastatin, hasDosageText, Literal("20 mg once daily")))
    g.add((med_atorvastatin, hasCode, atorvastatin_concept))

    return g


def main():
    g = build_patient_encounter_graph()

    output_file = "patient_encounter.ttl"
    g.serialize(destination=output_file, format="turtle")
    print(f"Wrote Turtle graph to {output_file}")


if __name__ == "__main__":
    main()
