import torch
from tasks.qa_feedback.training.gemma_reward_model import GemmaRewardModel
from fgrlhf.utils import mask_pad
from fgrlhf.value import MLP

# TODO make gemma safety reward value estimator for use as baseline in PPO 
class GemmaValue:

    def __init__(self,
                 model_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = False,
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        if model is not None:
            self.model = model
        else:
            self.gemma_reward = GemmaRewardModel()
            self.model = self.gemma_reward.base_model
            
        self.linear = MLP(self.model.config.hidden_size, 1)
        self.linear.to(self.model.device)
        self.linear.to(self.model.dtype)
        
        # freeze all parameters except the last layer
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        input_ids = torch.cat([prompts_input_ids, generated_input_ids], dim=1)
        attention_mask = torch.cat([prompts_attention_mask, generated_attention_mask], dim=1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        input_len = prompts_input_ids.size(1)
        output_len = generated_input_ids.size(1)
        
        # Get the hidden states matching the generated steps. 
        # Hidden state at step t is used to predict token t+1 and estimate value from step t onward.
        hidden_states = outputs.hidden_states[-1][:, input_len - 1 : input_len - 1 + output_len, :]

        logits = self.linear(hidden_states).squeeze(-1) # (B, output_len)
        logits = logits.to(torch.float32)
        
        results = {
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
        }

        return results