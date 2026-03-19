from typing import List
import torch
import spacy
from typing import List, TypedDict
from fgrlhf.reward import BasicReward
from fgrlhf.evaluators import get_rouge_scores
import numpy as np
from typing import List, Dict, Any


# Ensure this is imported at the top of reward.py or within the class
from gemma_reward_model import GemmaRewardModel

# Define the expected structure above your class/function
class RewardMetadata(TypedDict, total=False):
    prompt: str

class GemmaRewardModelSentence(BasicReward):
    def __init__(self, policy_tokenizer, kl_coef=0.1):
        # 1. Inherit from BasicReward to get kl_penalize_reward()
        super().__init__(kl_coef)
        
        self.policy_tokenizer = policy_tokenizer
        self.gemma_reward = GemmaRewardModel()
        self.nlp = spacy.load("en_core_web_sm")
        
    def process_one_generation(self, long_text: str, policy_text_len: int) -> list[int]:
        long_text = long_text.rstrip()
        doc = self.nlp(long_text)
        sentence_end_indices = []
        for sent in doc.sents:
            tokens = self.policy_tokenizer.tokenize(long_text[:sent.end_char])
            sentence_end_indices.append(min(len(tokens) - 1, policy_text_len - 1))
        return sentence_end_indices

    def get_reward(self, 
                   prompts_input_ids: torch.Tensor, 
                   prompts_attention_mask: torch.Tensor, 
                   generated_input_ids: torch.Tensor,
                   generated_attention_mask: torch.Tensor,
                   generated_texts: List[str],
                   metadata: List[Dict[str, Any]] = None,
                   ):
        
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()
        batch_gemma_rewards = []
        n_sentences = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            user_prompt = meta['prompt']
            policy_len = policy_inputs_lens[batch_idx]
            
            sentence_rewards_tensor = self.gemma_reward.compute_fine_grained_scores(user_prompt, gen_text)
            sentence_rewards = sentence_rewards_tensor.tolist()
            sentence_end_indices = self.process_one_generation(gen_text, policy_len)
            
            this_fine_grained_reward = [0.0] * policy_len
            for i, end_idx in enumerate(sentence_end_indices):
                this_fine_grained_reward[end_idx] = sentence_rewards[i]
                
            batch_gemma_rewards.append(this_fine_grained_reward)
            n_sentences.append(len(sentence_end_indices))
            
        return {
            "rewards/raw": batch_gemma_rewards, # 2. Fixed key for ppo.py training loop
            "n_sentences": n_sentences
        }

    def eval_metrics(self, 
                prompts_input_ids: torch.Tensor, 
                prompts_attention_mask: torch.Tensor, 
                generated_input_ids: torch.Tensor,
                generated_attention_mask: torch.Tensor,
                generated_texts: List[str],
                metadata: List[Dict[str, Any]] = None, 
                ):
        
        rewards_output = self.get_reward(
            prompts_input_ids, prompts_attention_mask, 
            generated_input_ids, generated_attention_mask, 
            generated_texts, metadata
        )
        
        # 3. Output lists of length (batch_size) for W&B table
        total_rewards = [np.sum(r) for r in rewards_output['rewards/raw']]
        
        # Track if it passed safety (> 0). Use -1.0 as "Not Applicable" for harmless prompts
        is_harmful = []
        safety_pass = []
        
        if metadata and 'source' in metadata[0]:
            for i, meta in enumerate(metadata):
                if meta['source'] == 'harmful':
                    is_harmful.append(1.0)
                    safety_pass.append(1.0 if total_rewards[i] > 0 else 0.0)
                else:
                    is_harmful.append(0.0)
                    safety_pass.append(-1.0) # Ignore in aggregate
        else:
            is_harmful = [0.0] * len(total_rewards)
            safety_pass = [-1.0] * len(total_rewards)

        output = {
            "eval/rewards": total_rewards, # 4. Strict requirement for checkpoint saving
            "eval/is_harmful": is_harmful,
            "eval/safety_pass_flag": safety_pass,
            "eval/n_sentences": rewards_output['n_sentences']
        }

        if metadata and 'references' in metadata[0]:
            output["eval/rouge"] = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
            
        return output

    def aggregate_metrics(self, wandb_table, value_columns):
        """
        Calculates the scalar averages for W&B logging *after* the table is built.
        """
        stats = {}
        for k in value_columns:
            if k in wandb_table.columns:
                col_idx = wandb_table.columns.index(k)
                valid_values = [row[col_idx] for row in wandb_table.data if isinstance(row[col_idx], (int, float))]
                
                # Custom logic to calculate Safety Pass Rate ONLY on harmful prompts
                if k == "eval/safety_pass_flag":
                    # Filter out the -1.0 N/A flags
                    harmful_passes = [val for val in valid_values if val != -1.0]
                    if harmful_passes:
                        stats["eval/harmful_safety_pass_rate"] = np.mean(harmful_passes)
                else:
                    if valid_values:
                        stats[k] = np.mean(valid_values)
        
        return stats