# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Refactored for CPU/MPS compatibility
from typing import List, Optional, Tuple

import torch
from torch import nn
import warnings

import transformers
# パッチを当てる前の、標準のLlamaAttentionに戻す
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel

# もしパッチが当たっていても、これで標準の実装に上書きし直します
transformers.models.llama.modeling_llama.LlamaAttention.forward = LlamaAttention.forward
transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = LlamaModel._prepare_decoder_attention_mask

print("[DEBUG] Flash Attention monkey patch has been disabled for inference.")
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Try to import Flash Attention - only available on CUDA
FLASH_ATTN_AVAILABLE = False
try:
    from einops import rearrange
    # * some version is changed to flash_attn_varlen_qkvpacked_func, so need to check
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    except:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Flash Attention is not available: {e}\n"
        "This is expected on CPU/MPS devices. Standard attention will be used instead.\n"
        "Flash Attention is a CUDA-only optimization for training on NVIDIA GPUs."
    )

def forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel
    
    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    offset = 0
    if past_key_value is not None:
        offset = past_key_value[0].shape[-2]
        kv_seq_len += offset
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states,
                                                    key_states,
                                                    cos,
                                                    sin,
                                                    offset=offset)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"
    assert past_key_value is None, "past_key_value is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3) # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask


    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                device=qkv.device)
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                    indices, bsz, q_len),
                        'b s (h d) -> b s h d', h=nheads)
    return self.o_proj(rearrange(output,
                                    'b s h d -> b s (h d)')), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    """
    Replace LLaMA attention with Flash Attention for CUDA devices.

    Refactored for CPU/MPS compatibility:
    - Only applies Flash Attention if available (CUDA GPUs)
    - Falls back to standard attention on CPU/MPS
    - Warns user about the configuration
    """
    if not FLASH_ATTN_AVAILABLE:
        warnings.warn(
            "Flash Attention replacement skipped - not available on this device.\n"
            "Using standard PyTorch attention. This is normal for CPU/MPS devices.\n"
            "Note: Training on CPU/MPS will be significantly slower than on CUDA GPUs."
        )
        return

    if not torch.cuda.is_available():
        warnings.warn(
            "Flash Attention is available but CUDA is not detected.\n"
            "Skipping Flash Attention replacement. Using standard attention."
        )
        return

    # Apply Flash Attention optimization
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    print("[INFO] Flash Attention successfully applied for CUDA training.")
