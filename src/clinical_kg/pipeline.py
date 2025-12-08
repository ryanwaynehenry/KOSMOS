from clinical_kg.config import load_config
from clinical_kg import data_models as dm
from clinical_kg.nlp import preprocessing, ner, coref, relations
from clinical_kg.umls import lookup
from clinical_kg.mapping import rules
from clinical_kg.instances import clustering, builder as inst_builder
from clinical_kg.kg import schema, builder as kg_builder, relations as kg_rel, validators, export

def run_pipeline_for_transcript(transcript_path: str, encounter_id: str, output_path: str):
    cfg = load_config()

    # Step 1: load and preprocess
    turns: list[dm.Turn] = preprocessing.load_and_segment(transcript_path, encounter_id)

    # Step 2: mention-level extraction
    mentions = ner.extract_mentions(turns)
    mentions = coref.add_coref(mentions)
    mentions = relations.attach_attributes(mentions)

    # Step 3: UMLS lookup
    mentions_with_codes = lookup.annotate_mentions_with_codes(mentions, cfg)

    # Step 4: instances
    instances = inst_builder.build_instances(mentions_with_codes, cfg)

    # Step 5: graph
    g = kg_builder.build_base_graph(encounter_id, instances)
    kg_rel.add_relations(g, instances)
    validators.validate_encounter_graph(g)

    # Step 6: export
    export.to_turtle(g, output_path)
