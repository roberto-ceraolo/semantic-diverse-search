import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_metric
import evaluate
from typing import List, Tuple, Dict, Any
from openai import OpenAI
import json
import logging
import random
from utils import Config, setup_logging
from pair_finder import SentencePairFinder

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.bertscore = evaluate.load('bertscore')
        if self.config.toggle_GPT:
            self.client = OpenAI(api_key=config.openai_api_key)

    def evaluate_pairs(self, pairs: List[Tuple], lang: str) -> Dict[str, float]:
        if len(pairs) > self.config.num_eval_samples:
            pairs = random.sample(pairs, self.config.num_eval_samples)
        
        bleu_scores = []
        bert_scores = []
        embedding_sim_scores = []
        lexical_div_scores = []
        gpt_scores = []

        for pair in pairs:
            if lang == "other":  # Cross-lingual pairs
                sent1, sent2, embedding_sim = pair
            else:
                sent1, sent2, embedding_sim, lexical_div = pair
            
            
            bert_score = self.bertscore.compute(predictions=[sent2], references=[sent1], lang=lang)
            bert_scores.append(bert_score['f1'][0])
            
            embedding_sim_scores.append(embedding_sim)
            if lang != "other":
                lexical_div_scores.append(lexical_div)
                bleu = sentence_bleu([sent1.split()], sent2.split(), smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu)


        if self.config.toggle_GPT:
            gpt_scores = self.gpt_evaluate_pairs(pairs)

        results = {
            "avg_bert": np.mean(bert_scores),
            "avg_embedding_sim": np.mean(embedding_sim_scores),
        }

        if lang != "other":
            results["avg_lexical_div"] = np.mean(lexical_div_scores)
            results["avg_bleu"] = np.mean(bleu_scores)

        if self.config.toggle_GPT:
            results["avg_gpt_score"] = np.mean(gpt_scores)

        return results

    def gpt_evaluate_pairs(self, pairs: List[Tuple]) -> List[float]:
        prompt = self._prepare_gpt_prompt(pairs)
        system_prompt = "You are an expert linguist specializing in semantic analysis."
        response_format = {"type": "json_object"}
        
        response = self._get_gpt_response(prompt, system_prompt, response_format)
        ratings = self._parse_gpt_output(response)
        
        return ratings["similarity_ratings"]["ratings"]

    def _prepare_gpt_prompt(self, pairs: List[Tuple]) -> str:
        prompt = """Your task is to evaluate the semantic similarity of pairs of sentences pertaining to the legal world. Some pairs may be in the same language, while others may be cross-lingual (e.g., one sentence in English and one in German). For each pair, follow these steps:

Rate the semantic similarity of the sentences on a scale of 1 to 5, where:
1 = Completely different meaning
2 = Mostly different meaning
3 = Somewhat similar meaning
4 = Very similar meaning
5 = Identical or nearly identical meaning

Provide your response as a JSON object with the following structure:
    {
        "similarity_ratings": {
            "ratings": [Rating_pair_1, Rating_pair_2, ...]
        }
     }

Remember, we're interested in semantic similarity, not just lexical overlap. Two sentences can use different words but convey the same meaning, or use similar words but have subtly different meanings.
Please evaluate the following pairs:"""

        for i, pair in enumerate(pairs, 1):
            prompt += f"\nPair {i}:\n"
            prompt += f"Sentence 1: {pair[0]}\n"
            prompt += f"Sentence 2: {pair[1]}\n"

        return prompt

    def _get_gpt_response(self, prompt: str, system_prompt: str, response_format: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.config.gpt_model,
            messages=messages,
            temperature=0.4,
            seed=42,
            response_format=response_format
        )
        return response.choices[0].message.content

    def _parse_gpt_output(self, output: str) -> Dict[str, Any]:
        try:
            parsed_data = json.loads(output)
            return parsed_data
        except json.JSONDecodeError:
            logging.error(f"Error: GPT's response was not valid JSON. Raw response:\n{output}")
            return None

    def log_results(self, results: Dict[str, Dict[str, float]]) -> None:
        for lang, scores in results.items():
            log_message = f"Evaluation Results for {lang.upper()} pairs:\n"
            for metric, value in scores.items():
                log_message += f"{metric}: {value:.4f}\n"
            log_message += "\n"
            logging.info(log_message)
            print(log_message)  # Also print to console
    
    def log_training_stats(self, stats: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        for lang, lang_stats in stats.items():
            log_message = f"Training Stats for {lang.upper()} pairs:\n"
            for category, category_stats in lang_stats.items():
                log_message += f"{category.capitalize()}:\n"
                for stat, value in category_stats.items():
                    log_message += f"  {stat}: {value:.4f}\n"
            log_message += "\n"
            logging.info(log_message)
            print(log_message)
