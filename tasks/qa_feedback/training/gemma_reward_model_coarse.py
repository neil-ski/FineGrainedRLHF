import torch
import numpy as np
from typing import List, Dict, Any, TypedDict
from fgrlhf.reward import BasicReward
from fgrlhf.evaluators import get_rouge_scores
from tasks.qa_feedback.training.gemma_reward_model import GemmaRewardModel

# Define the expected structure above your class/function
class RewardMetadata(TypedDict, total=False):
    prompt: str
    source: str
    references: str

class GemmaRewardModelCoarse(BasicReward): # reward for entire output
    def __init__(self, policy_tokenizer, kl_coef=0.1):
        super().__init__(kl_coef)
        self.policy_tokenizer = policy_tokenizer
        self.gemma_reward = GemmaRewardModel()
        
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
        
        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            user_prompt = meta['prompt']
            policy_len = policy_inputs_lens[batch_idx]
            
            score_tensor = self.gemma_reward.score_with_str(user_prompt, gen_text)
            score = score_tensor.item()
            
            this_coarse_reward = [0.0] * policy_len
            if policy_len > 0:
                this_coarse_reward[-1] = score # Apply the reward to the last token
                
            batch_gemma_rewards.append(this_coarse_reward)
            
        return {
            "rewards/raw": batch_gemma_rewards,
            "n_sentences": [1] * len(metadata) # Treat the entire response as a single segment
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
        
        total_rewards = [np.sum(r) for r in rewards_output['rewards/raw']]

        # All prompts are now assumed to be harmful
        safety_pass = [1.0 if r > 0 else 0.0 for r in total_rewards]

        output = {
            "eval/rewards": total_rewards,
            "eval/safety_pass_flag": safety_pass,
            "eval/n_sentences": rewards_output['n_sentences']
        }

        if metadata and 'references' in metadata[0]:
            output["eval/rouge"] = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
            
        return output

    def aggregate_metrics(self, wandb_table, value_columns):
        stats = {}
        for k in value_columns:
            if k in wandb_table.columns:
                col_idx = wandb_table.columns.index(k)
                valid_values = [row[col_idx] for row in wandb_table.data if isinstance(row[col_idx], (int, float))]
                
                if valid_values:
                    if k == "eval/safety_pass_flag":
                        stats["eval/harmful_safety_pass_rate"] = np.mean(valid_values)
                    else:
                        stats[k] = np.mean(valid_values)
        
        return stats