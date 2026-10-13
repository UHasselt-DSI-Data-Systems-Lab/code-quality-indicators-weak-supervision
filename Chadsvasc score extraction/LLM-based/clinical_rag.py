import re
import torch
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from llama_cpp import Llama
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, GemmaTokenizerFast
import copy


class ClinicalRAG:
    def __init__(self, model_path, embed_name, reranker_name, question_terms, prompt, retrieval_mode = "lexical", chunk_size=2):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=16, n_gpu_layers=-1)
        self.embed_model = SentenceTransformer(embed_name)
        self.reranker = CrossEncoder(reranker_name)
        self.relevant_sections = ["BESLUIT", "ANAMNESE", "ANTECEDENTEN", "RISICOFACTOREN", "MEDICATIE", "ONTSLAG_MEDICATIE"]
        self.chunk_size = chunk_size
        self.question_terms = question_terms
        self.prompt = prompt
        self.overlap = 0
        self.retrieval_mode = retrieval_mode
        
        # if "geitje" in model_path.lower():
        #     tokenizer_id = "BramVanroy/GEITje-7B-ultra"
        #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # elif "gemma" in model_path.lower():
        #     tokenizer_id = "google/gemma-4-31B-it"
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #             tokenizer_id, 
        #             trust_remote_code=True,
        #             use_fast=True 
        #         )
        # else:
        #     tokenizer_id = "mistralai/Mistral-7B-Instruct-v0.2" 
        #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # print(tokenizer_id)
        
    def _extract_age(self, sections):
        text = (sections.get("BESLUIT") or "").lower()
        patterns = [r"\b(\d{2})-jarige\b", r"\b(\d{2})\s*jarige\b", r"\b(\d{2})\s*jaar oud\b"]
        for p in patterns:
            match = re.search(p, text)
            if match: return int(match.group(1))
        return None

        
    def _chunk_sections(self, sections):
        chunks = []
    
        for sec in self.relevant_sections:
            text = sections.get(sec) or ""
            if not text.strip():
                continue
    
            # Basic cleanup
            text = text.replace("\n", " ").replace("\r", " ")
            text = re.sub(r"\s+", " ", text).strip()
    
            sentences = sent_tokenize(text, language="dutch")
    
            i = 0
            while i < len(sentences):
                chunk_sents = sentences[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_sents).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                i += self.chunk_size - self.overlap
    
        return chunks
    
    def _get_context(self, chunks, question, terms):
        # 1. LEXICAL MODE
        if self.retrieval_mode == "lexical":
            pattern = re.compile(rf'\b({"|".join(map(re.escape, terms))})\b', re.IGNORECASE)
            hits = [c for c in chunks if pattern.search(c)]
            return " ".join(hits[:5]) if hits else None

        # 2. SEMANTIC MODE (Retrieve Top-K + Rerank)
        elif self.retrieval_mode == "semantic":
            q_emb = self.embed_model.encode(question, convert_to_tensor=True)
            c_emb = self.embed_model.encode(chunks, convert_to_tensor=True)
            scores = cos_sim(q_emb, c_emb)[0]
            top_idx = torch.topk(scores, k=min(10, len(chunks))).indices.tolist()
            top_chunks = [chunks[i] for i in top_idx]
            
            # Rerank
            pairs = [[question, c] for c in top_chunks]
            rerank_scores = self.reranker.predict(pairs)
            ranked_chunks = [c for _, c in sorted(zip(rerank_scores, top_chunks), reverse=True)]
            return " ".join(ranked_chunks[:5])

            
        elif self.retrieval_mode == "hybrid":
            # STAGE 1: Lexical Filter (Keyword Match)
            # Narrow down the search space to chunks containing specific terms
            pattern = re.compile(rf'\b({"|".join(map(re.escape, terms))})\b', re.IGNORECASE)
            lexical_hits = [c for c in chunks if pattern.search(c)]
            
            if not lexical_hits:
                return None
            
            # STAGE 2: Similarity Search (Top K)
            # From the lexical hits, find the most semantically similar to the question
            lex_embeddings = self.embed_model.encode(lexical_hits, convert_to_tensor=True)
            q_emb = self.embed_model.encode(question, convert_to_tensor=True)
            
            # Calculate similarity and take top K (e.g., top 10)
            scores = cos_sim(q_emb, lex_embeddings)[0]
            top_k_val = min(10, len(lexical_hits))
            top_idx = torch.topk(scores, k=top_k_val).indices.tolist()
            semantic_candidates = [lexical_hits[i] for i in top_idx]
    
            # STAGE 3: Reranker (Cross-Encoder)
            # Deep analysis of the semantic candidates to find the final context
            pairs = [[question, c] for c in semantic_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort and take top 3 or 5 for the LLM
            final_chunks = [c for _, c in sorted(zip(rerank_scores, semantic_candidates), 
                                                 key=lambda x: x[0], 
                                                 reverse=True)]
            
        return " ".join(final_chunks[:3])

    def _ask_llm(self, question, context, prompt_template):
        if isinstance(prompt_template, list):
            msg_list = copy.deepcopy(prompt_template)
            
            for msg in msg_list:
                if msg["role"] == "user":
                    msg["content"] = msg["content"].format(context=context, question=question)
            
            final_prompt = self.tokenizer.apply_chat_template(
                msg_list, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            final_prompt = prompt_template.format(context=context, question=question)

        output = self.llm(
            final_prompt, 
            max_tokens=4, 
            temperature=0.0,
            stop=["</s>", "<|im_end|>", "<end_of_turn>", "\n"] 
        )
        
        return output["choices"][0]["text"].strip().lower().startswith("ja")
        
        
        prompt = prompt_template.format(context=context, question=question)
        
        # # print(question)
        # # print(context)

        # output = self.llm(
        #     prompt, 
        #     max_tokens=2, 
        #     temperature=0.0, 
        # )
        
        # # print(output["choices"][0]["text"].strip().lower())
        # # print("\n\n")

        
        return output["choices"][0]["text"].strip().lower().startswith("ja")

    def process_row(self, row):
        results = {}
        sections = row.sections
        
        # Deterministic Age
        age = self._extract_age(sections)
        results["A_LLM"] = 1 if age and 65 <= age <= 74 else 0
        results["A2_LLM"] = 1 if age and age > 74 else 0

        # Clinical Reasoning
        chunks = self._chunk_sections(sections)
        for factor, q_dict in self.question_terms.items():
            if factor in ["A", "A2"]: continue
            is_positive = 0
            for question, terms in q_dict.items():
                context = self._get_context(chunks, question, terms)
                if context and self._ask_llm(question, context, self.prompt):
                    is_positive = 1
                    break
            results[f"{factor}_LLM"] = is_positive
        return results
        
        
        
    def chadsvasc_metrics(self, df, structured_mapping=None):
        """
        Calculate precision, recall, F1 for each CHADS-VASc component and overall macro/micro F1.
        Can optionally adjust LLM predictions using structured ICD-10 data.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing ground truth and LLM prediction columns.
            structured_mapping (dict, optional): Dict of component -> list of ICD-10 columns.
        
        Returns:
            pd.DataFrame: Component-level metrics + macro/micro F1 in the last row.
        """
        
        # Mapping: component -> (true_col, pred_col)
        mapping = {
            'C': ('CHF', 'C_LLM'),
            'H': ('HTN', 'H_LLM'),
            'A2': ('AGE_75', 'A2_LLM'),
            'D': ('DM', 'D_LLM'),
            'S': ('STROKE', 'S_LLM'),
            'V': ('VASCULAR', 'V_LLM'),
            'A': ('AGE_65_74', 'A_LLM'),
            'Sc': ('SEX_FEMALE', 'SC_LLM')
        }
        
        # Extract all expected ground truth columns from the mapping
        required_true_cols = [true_col for true_col, pred_col in mapping.values()]
        
        # Check if these columns exist in the current dataframe
        missing_cols = [col for col in required_true_cols if col not in df.columns]
        
        if missing_cols:
            print("WARNING: EVALUATION NOT POSSIBLE")
            print(f"Missing ground truth columns: {missing_cols}")
            print("Reason: You are likely using a training set without manual annotations.")
            return None
            
        df_copy = df.copy()
        

        results = []
        
        for component, (true_col, pred_col) in mapping.items():
            y_true = df_copy[true_col].fillna(0)
            y_pred = df_copy[pred_col].fillna(0)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'Component': component,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })
        
        metrics_df = pd.DataFrame(results)
        
        all_y_true = df_copy[[true for true, pred in mapping.values()]].fillna(0)
        all_y_pred = df_copy[[pred for true, pred in mapping.values()]].fillna(0)
        
        macro_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(all_y_true, all_y_pred, average='micro', zero_division=0)
        
        overall_row = pd.DataFrame([{
            'Component': 'Overall Macro/Micro',
            'Precision': None,
            'Recall': None,
            'F1': f"Macro: {macro_f1:.3f}, Micro: {micro_f1:.3f}"
        }])
        
        metrics_df = pd.concat([metrics_df, overall_row], ignore_index=True)
        
        return metrics_df
        
