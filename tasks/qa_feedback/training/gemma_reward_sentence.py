from typing import List, Dict, Any
import torch
import spacy

# Ensure this is imported at the top of reward.py or within the class
from gemma_reward_model import GemmaRewardModel

# Define the expected structure above your class/function
class RewardMetadata(TypedDict, total=False):
    prompt: str

class GemmaRewardModelSentence:
    def __init__(self, policy_tokenizer):
        """
        Initializes the Gemma fine-grained sentence reward model.
        
        Args:
            policy_tokenizer: The tokenizer used by the policy model (e.g., T5). 
                              Used to map character boundaries back to token boundaries.
        """
        self.policy_tokenizer = policy_tokenizer
        
        # Initialize the underlying ShieldGemma model
        self.gemma_reward = GemmaRewardModel()
        
        # Load spacy for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")
        
    def process_one_generation(self, long_text: str, policy_text_len: int) -> list[int]:
        """
        Splits the text into sentences using Spacy and finds the policy token index 
        corresponding to the end of each sentence.
        """
        # We strip trailing whitespace to perfectly match the logic inside 
        # GemmaRewardModel.compute_fine_grained_scores()
        long_text = long_text.rstrip()
        doc = self.nlp(long_text)
        
        sentence_end_indices: list[int] = []
        
        for sent in doc.sents:
            # Tokenize the prefix up to the end of the current sentence using the policy model's tokenizer
            tokens = self.policy_tokenizer.tokenize(long_text[:sent.end_char])
            token_count = len(tokens)
            
            # Record the index of the last token in the sentence (0-indexed).
            # We cap it at policy_text_len - 1 to prevent out-of-bounds indexing in case of length mismatches.
            sentence_end_indices.append(min(token_count - 1, policy_text_len - 1))
            
        return sentence_end_indices

    def get_reward(self, 
                   prompts_input_ids: torch.Tensor, 
                   prompts_attention_mask: torch.Tensor, 
                   generated_input_ids: torch.Tensor, # (B, output_len)
                   generated_attention_mask: torch.Tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata: List[RewardMetadata],
                   ) -> dict:
        
        # Get the actual unpadded sequence length for each generated text in the batch
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()
        
        batch_gemma_rewards: list[list[float]] = []
        n_sentences: list[int] = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            user_prompt = meta['prompt']
            policy_len = policy_inputs_lens[batch_idx]
            
            # 1. Get the scalar rewards per sentence prefix from Gemma
            # Returns a tensor of shape (num_sentences,)
            sentence_rewards_tensor = self.gemma_reward.compute_fine_grained_scores(user_prompt, gen_text)
            sentence_rewards = sentence_rewards_tensor.tolist()
            
            # 2. Map sentences boundaries to policy tokenizer indices
            sentence_end_indices = self.process_one_generation(gen_text, policy_len)
            
            # Safety check: Spacy should yield the exact same number of sentences in both methods
            assert len(sentence_rewards) == len(sentence_end_indices), "Mismatch between Gemma reward count and policy token indices count"
            
            # 3. Place rewards into fine-grained list (aligned with policy model tokens)
            this_fine_grained_reward = [0.0] * policy_len
            
            for i, end_idx in enumerate(sentence_end_indices):
                # Note: compute_fine_grained_scores computes the safety score of the cumulatively generated prefix.
                # If you want step-wise *marginal* rewards instead of cumulative prefix rewards, 
                # you can change this to: reward = sentence_rewards[i] - (sentence_rewards[i-1] if i > 0 else 0)
                this_fine_grained_reward[end_idx] = sentence_rewards[i]
                
            batch_gemma_rewards.append(this_fine_grained_reward)
            n_sentences.append(len(sentence_end_indices))
            
        return {
            "gemma_rewards": batch_gemma_rewards,
            "rewards": batch_gemma_rewards, # Included so it acts natively as a drop-in replacement
            "n_sentences": n_sentences
        }
