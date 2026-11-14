"""Transparent partial SNOMED annotation proxies, never clinical negative labels.

Known positive correspondences come from the already-local Phase2 crosswalk.
No network/codebook inference. Every unrecognized code is explicitly excluded.
The broad PTB-XL SCP endpoints are not claimed clinically equivalent to these unions.
"""

CORRESPONDENCES = {
    "270492004": ("CD", "first-degree atrioventricular block"),
    "713426002": ("CD", "incomplete right bundle branch block"),
    "713427006": ("CD", "complete right bundle branch block"),
    "59118001": ("CD", "unspecified right bundle branch block"),
    "733534002": ("CD", "complete left bundle branch block"),
    "445118002": ("CD", "left anterior fascicular block"),
    "698252002": ("CD", "nonspecific intraventricular conduction block"),
    "164934002": ("STTC", "T-wave abnormality"),
    "428750005": ("STTC", "nonspecific ST-T abnormality"),
    "429622005": ("STTC", "ST depression"),
    "164930006": ("STTC", "ST elevation"),
    "59931005": ("STTC", "T-wave inversion"),
}
KNOWN_EXCLUSIONS = {
    "164909002": "Excluded: documented LBBB/LFBBB conversion collision; no specific subtype inferred",
    "233917008": "Generic AV block: excluded from this restricted proxy; no degree guessed",
    "426783006": "Sinus rhythm is not a completely normal ECG or confirmed CD/STTC negative",
    "164889003": "Atrial fibrillation: no matching model output; not a CD/STTC subtype",
    "164890007": "Atrial flutter: no matching model output; do not confuse abbreviation AF",
    "698247007": "Cardiac dysrhythmia, not the PTB-XL CD conduction superclass",
    "164947007": "Prolonged PR alone is not automatically an AV-block diagnosis",
}

# Fixed condition families shared by target code unions and source SCP key unions.
# 164909002 has a documented conversion collision and is excluded entirely.
CONDITIONS = {
    "IAVB": {"label": "CD", "snomed": ["270492004"], "scp": ["1AVB"]},
    "RBBB_family": {
        "label": "CD",
        "snomed": ["59118001", "713426002", "713427006"],
        "scp": ["IRBBB", "CRBBB"],
    },
    "complete_LBBB": {
        "label": "CD",
        "snomed": ["733534002"],
        "scp": ["CLBBB"],
    },
    "LAFB": {"label": "CD", "snomed": ["445118002"], "scp": ["LAFB"]},
    "IVCD": {"label": "CD", "snomed": ["698252002"], "scp": ["IVCD"]},
    "ST_depression": {"label": "STTC", "snomed": ["429622005"], "scp": ["STD_"]},
    "ST_elevation": {"label": "STTC", "snomed": ["164930006"], "scp": ["STE_"]},
    "T_abnormality": {"label": "STTC", "snomed": ["164934002"], "scp": ["TAB_", "NT_", "NDT"]},
    "T_inversion": {"label": "STTC", "snomed": ["59931005"], "scp": ["INVT"]},
    "nonspecific_ST_T": {"label": "STTC", "snomed": ["428750005"], "scp": ["NST_", "NT_", "NDT"]},
}
PER_CODE = {
    "270492004": {"name": "IAVB", "label": "CD", "scp": ["1AVB"]},
    "713426002": {"name": "IRBBB", "label": "CD", "scp": ["IRBBB"]},
    "713427006": {"name": "CRBBB", "label": "CD", "scp": ["CRBBB"]},
    "59118001": {
        "name": "recorded unspecified RBBB",
        "label": "CD",
        "scp": ["IRBBB", "CRBBB"],
        "limitation": "Source reference includes specified RBBB subtypes; descriptive family proxy",
    },
    "733534002": {"name": "complete LBBB", "label": "CD", "scp": ["CLBBB"]},
    "445118002": {"name": "LAFB", "label": "CD", "scp": ["LAFB"]},
    "698252002": {"name": "IVCD", "label": "CD", "scp": ["IVCD"]},
    "429622005": {"name": "ST depression", "label": "STTC", "scp": ["STD_"]},
    "164930006": {"name": "ST elevation", "label": "STTC", "scp": ["STE_"]},
    "164934002": {"name": "T-wave abnormality", "label": "STTC", "scp": ["TAB_", "NT_", "NDT"]},
    "59931005": {"name": "T-wave inversion", "label": "STTC", "scp": ["INVT"]},
    "428750005": {
        "name": "nonspecific ST/T annotation",
        "label": "STTC",
        "scp": ["NST_", "NT_", "NDT"],
    },
}


def mapping(codes):
    return {
        "endpoint": "recorded-code annotation-union proxies, ordered CD/STTC",
        "clinical_equivalence_established": False,
        "negative_semantics": "No included code recorded, not confirmed clinical absence",
        "source_semantic_limit": "Partial target unions do not exhaust PTB-XL SCP CD/STTC; reporting practice can cause apparent drop",
        "source": "docs/phase2/DIAGNOSTIC_LABEL_CROSSWALK.md; offline local dictionary evidence only",
        "conditions": CONDITIONS,
        "per_code_descriptive_tasks": PER_CODE,
        "proxy_snomed": {
            label: sorted(
                {code for c in CONDITIONS.values() if c["label"] == label for code in c["snomed"]}
            )
            for label in ["CD", "STTC"]
        },
        "proxy_scp": {
            label: sorted(
                {code for c in CONDITIONS.values() if c["label"] == label for code in c["scp"]}
            )
            for label in ["CD", "STTC"]
        },
        "entries": {
            code: {
                "label": CORRESPONDENCES[code][0] if code in CORRESPONDENCES else None,
                "meaning": CORRESPONDENCES[code][1] if code in CORRESPONDENCES else None,
                "reason": "Documented positive category correspondence; subtype equivalence not assumed"
                if code in CORRESPONDENCES
                else KNOWN_EXCLUSIONS.get(
                    code,
                    "Excluded: no verified local mapping to requested outputs; no diagnosis guessed",
                ),
            }
            for code in sorted(set(codes))
        },
        "per_code_scoring": "Human-authorized descriptive AUROC uses the existing CD/STTC score; not a dedicated per-code output or probability. AF is not evaluable. 164909002 is excluded from per-code LBBB inference due conversion ambiguity.",
    }


def targets(codes, frozen):
    if any(code not in frozen["entries"] for code in codes):
        raise ValueError("Code absent from frozen mapping")
    labels = {frozen["entries"][code]["label"] for code in codes}
    return [int("CD" in labels), int("STTC" in labels)]
