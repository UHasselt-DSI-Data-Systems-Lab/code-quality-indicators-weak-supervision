import pandas as pd
import numpy as np
from clinical_rag import ClinicalRAG
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# CONFIGURATION
# =========================
FILE = "label_model.feather"
# FILE = "test_set.feather"


MODEL_PATH = "/model/gemma/google_gemma-4-31B-it-Q5_K_S.gguf"

CLINICAL_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Je bent een klinisch beslissingssysteem.\n\n"
    "Je taak is vast te stellen of een klinische conditie AANWEZIG is bij de PATIËNT op basis van de ontslagbrief.\n\n"
    "Begin je antwoord met één woord: Ja of Nee.\n\n"
    'Beantwoord "Ja" ALLEEN als er EXPLICIET bewijs is dat de patiënt de aandoening heeft (heden of in het verleden), '
    "ook als dit onderdeel is van een berekende klinische score (zoals CHADS-VASc) of een lijst met actieve problemen.\n"
    'Beantwoord "Ja" als de medicatie in de medicatielijst staat, houd rekening met de dosis indien deze vermeld is.\n\n'
    'Antwoord "Nee" in de volgende gevallen:\n'
    '* De aandoening wordt ontkend (bv. "geen hartfalen")\n'
    "* Het is een vermoeden of vraagstelling\n"
    "* Het betreft een risico of screening\n"
    "* Het betreft educatie\n"
    "* Het betreft familiale voorgeschiedenis (bv. moeder, vader, broer, zus)\n"
    "* Het betreft een andere persoon dan de patiënt\n\n"
    "Gebruik alleen informatie uit de klinische context.<|im_end|>\n"
    "<|im_start|>user\n"
    "Klinische context:\n{context}\n\n"
    "Vraag: {question}\n"
    "Antwoord (Ja/Nee):<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Antwoord:"
)


TEMPLATES = {
    "condition": "Heeft de patiënt {item}?",
    "medication": "Neemt de patiënt de medicatie {item}?",
    "procedure": "Heeft de patiënt een {item} ondergaan of doorgemaakt?",
    "sign": "Is er een positieve signaal voor {item}?",
    "sex": "Is het een vrouw/patiënte?",
}

RISK_FACTOR_CONFIG = {
    "C": [
        ("condition", ["hartfalen", "cordedecompensatie", "hartdecompensatie"]),
        ("medication", ["jardiance", "forxiga", "aldactone", "burinex", "lasix"]),
    ],
    "H": [("condition", ["arteriële hypertensie", "hypertensie", "hypertensief"])],
    "D": [
        ("condition", ["diabetes mellitus", "diabetes", "dm", "glucosemetabolisme"]),
        ("medication", ["metformine", "metformax"]),
        ("sign", ["glucosemetabolisme"]),
    ],
    "S": [("procedure", ["cva", "tia"])],
    "V": [
        (
            "condition",
            ["coronair lijden", "perifeer vaatlijden", "arterieel vaatlijden"],
        ),
        (
            "condition",
            [
                "drietakslijden",
                "meervatslijden",
                "eentakslijden",
                "tweetakslijden",
                "taksziekte",
            ],
        ),
        ("medication", ["clopidogrel", "xarelto 2.5mg", "brilique"]),
        (
            "procedure",
            ["pta", "ptca", "pci", "cabg", "coronaire bypass", "vasculopathie"],
        ),
    ],
    "SC": [("sex", ["vrouw", "patiënte"])],
}

# ICD-10 Structured Overrides
STRUCTURED_MAPPING = {
    "C": (
        "CHF",
        [
            "diagnose_i501",
            "diagnose_i502",
            "diagnose_i503",
            "diagnose_i504",
            "diagnose_i508",
            "diagnose_i509",
        ],
    ),
    "H": (
        "HTN",
        [
            "diagnose_i10",
            "diagnose_i11",
            "diagnose_i12",
            "diagnose_i13",
            "diagnose_i15",
        ],
    ),
    "D": (
        "DM",
        [
            "diagnose_e10",
            "diagnose_e11",
            "diagnose_e12",
            "diagnose_e13",
            "diagnose_e14",
        ],
    ),
    "S": ("STROKE", ["diagnose_i69", "diagnose_z8673"]),
}


# =========================
# EXECUTION
# =========================
def apply_structured_overrides(df, structured_mapping):
    df_copy = df.copy()

    mapping = {
        "C": "C_LLM",
        "H": "H_LLM",
        "A2": "A2_LLM",
        "D": "D_LLM",
        "S": "S_LLM",
        "V": "V_LLM",
        "A": "A_LLM",
        "SC": "SC_LLM",
    }

    for component, pred_col in mapping.items():
        if component in structured_mapping:
            icd_cols = structured_mapping[component][1]
            df_copy[pred_col] = np.where(
                df_copy[icd_cols].sum(axis=1) > 0, 1, df_copy[pred_col]
            )

    return df_copy


def generate_question_terms(config, templates):
    final_dict = {}
    for factor, items_list in config.items():
        factor_questions = {}
        for q_type, terms in items_list:
            for term in terms:
                # Create the question using the template
                question = templates[q_type].format(item=term)
                # Keep the original structure: { "Question": [Keyword] }
                factor_questions[question] = [term]
        final_dict[factor] = factor_questions
    return final_dict


if __name__ == "__main__":
    df = pd.read_feather(f"../code/ChadsVasc score extraction/Jessa/{FILE}")

    for retrieval_mode in ["lexical"]:
        # df = df.sample(n=20, random_state=41)

        rag_engine = ClinicalRAG(
            MODEL_PATH,
            "UMCU/mirrorbert_medroberta.nl_meantoken_sbert",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            generate_question_terms(RISK_FACTOR_CONFIG, TEMPLATES),
            CLINICAL_PROMPT_TEMPLATE,
            retrieval_mode,
        )

        llm_results = [rag_engine.process_row(row) for row in df.itertuples()]

        df_out = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(llm_results)], axis=1
        )

        print(f"Retrieval mode: {retrieval_mode}")
        # Metrics using only LLM predictions
        print("=== Metrics: Unstructured data only ===")
        print(rag_engine.chadsvasc_metrics(df_out))

        df_combined = apply_structured_overrides(df_out, STRUCTURED_MAPPING)

        # Metrics after adjusting LLM with structured ICD-10 data
        print("=== Metrics: Unstructured data + Structured ICD-10 ===")
        print(rag_engine.chadsvasc_metrics(df_combined))

        output_path = f"../code/ChadsVasc score extraction/Contextual LLM/Gemma/{FILE}_{retrieval_mode}.feather"
        df_combined.to_feather(output_path)
