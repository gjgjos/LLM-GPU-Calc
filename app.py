import streamlit as st
from transformers import AutoConfig
import requests

# Constants
DEFAULT_GPU_UTILIZATION = 0.9
NON_TORCH_MEMORY_MB = 1024

PARAMETER_DATA_TYPE_SIZE = {
    "float32": 4,
    "float16/bfloat16": 2,
    "float8/int8": 1,
    "int4": 0.5
}
KV_DATA_TYPE_SIZE = {
    "float16": 2,
    "float8": 1,
}


def get_model_parameters(model_name):
    # Hugging Face API URL
    url = f"https://huggingface.co/api/models/{model_name}"

    # API 요청
    response = requests.get(url)

    if response.status_code == 200:
        model_data = response.json()
        num_params = model_data['safetensors']['total']
        
        return num_params
    else:
        raise Exception(f"Failed to fetch model details: {response.status_code}")

def get_config_value(config, possible_keys, default=None):
    for key in possible_keys:
        if hasattr(config, key):
            return getattr(config, key)
    return default

def calculate_required_gpu_memory(model_name, concurrent_users, seq_len, gpu_utilization, data_type, kv_data_type, huggingface_key=None):
    # Download model config
    try:
        if huggingface_key:
            config = AutoConfig.from_pretrained(model_name, use_auth_token=huggingface_key, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        return None, f"Error loading model config: {e}"

    # Extract model parameters
    num_params = get_model_parameters(model_name)
    
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head", "n_heads", "num_heads"])
    if num_attention_heads is None:
        return None, "Number of attention heads not found in model config."

    num_key_value_heads = get_config_value(config, ["num_key_value_heads", "n_head_kv", "num_kv_heads", "multi_query_group_num", "kv_n_head"], num_attention_heads)

    hidden_size = get_config_value(config, ["hidden_size", "dim", "n_embd"])
    if hidden_size is None:
        return None, "Hidden size not found in model config."

    head_dim = get_config_value(config, ["head_dim"], hidden_size // num_attention_heads)


    num_layers = get_config_value(config, ["num_layers", "n_layer", "n_layers", "num_hidden_layers"])
    if num_layers is None:
        return None, "Number of layers not found in model config."
    
    intermediate_size = config.intermediate_size

    # Calculate components
    model_weight = num_params * PARAMETER_DATA_TYPE_SIZE[data_type]
    pytorch_activation_peak_memory = seq_len * concurrent_users * (18 * hidden_size + 4 * intermediate_size)
    kv_cache_memory_per_batch = (
        2 * num_key_value_heads * head_dim * num_layers * KV_DATA_TYPE_SIZE[kv_data_type] * seq_len
    )

    total_kv_cache_memory = kv_cache_memory_per_batch * concurrent_users

    required_gpu_memory = (
        model_weight + NON_TORCH_MEMORY_MB * 1024**2 + pytorch_activation_peak_memory + total_kv_cache_memory
    ) / gpu_utilization # bytes

    # Convert Bytes to GB
    required_gpu_memory_gb = required_gpu_memory / (1024**3)

    explanation = {
        "num_params": num_params,
        "param_data_type": data_type,
        "concurrent_users": concurrent_users,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "kv_data_type": kv_data_type,
        "gpu_utilization": gpu_utilization,
    }

    process_steps = f"""
    ##### Calculation Process:
    1. **Model Weight**: {model_weight / (1024**3):.3f} GB  
       Computed as: `num_params x size of parameter data type`
    2. **Non-Torch Memory**: {NON_TORCH_MEMORY_MB / 1024:.3f} GB  
       Computed as a constant value of 1 GB
    3. **PyTorch Activation Peak Memory**: {pytorch_activation_peak_memory / (1024**3):.3f} GB  
       Computed as: `concurrent_users x seq_len x (18 x hidden_size + 4 x intermediate_size)`
    4. **KV Cache Memory Per Batch**: {kv_cache_memory_per_batch / (1024**3):.3f} GB  
       Computed as: `2 x num_key_value_heads x head_dim x num_layers x size of KV data type x seq_len`
    5. **Total KV Cache Memory**: {total_kv_cache_memory / (1024**3):.3f} GB  
       Computed as: `kv_cache_memory_per_batch x concurrent_users`
    6. **Final GPU Memory Requirement**: {required_gpu_memory_gb:.3f} GB  
       Computed as: `(Model Weight + Non-Torch Memory + PyTorch Activation Peak Memory + Total KV Cache Memory) ÷ GPU Utilization`
    """

    return required_gpu_memory_gb, explanation, process_steps

# Streamlit UI
st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Input Parameters")
    # User inputs
    model_name = st.text_input("Model Name (e.g., Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-V3)")
    data_type = st.selectbox("Parameter Data Type", list(PARAMETER_DATA_TYPE_SIZE.keys()), index=1)
    concurrent_users = st.number_input("Number of Concurrent Users (=batch size)", min_value=1, value=10, step=1)
    seq_len = st.number_input("Sequence Length (input+output token)", min_value=1, value=4096, step=1)
    gpu_utilization = st.slider("GPU Utilization (default: 0.9)", min_value=0.1, max_value=1.0, value=DEFAULT_GPU_UTILIZATION)
    kv_data_type = st.selectbox("KV Cache Data Type", list(KV_DATA_TYPE_SIZE.keys()), index=0)
    huggingface_key = st.text_input("Hugging Face Token (if required)", type="password")
    calculate_button = st.button("Calculate Required GPU Memory")

with st.container():

    # Streamlit UI
    st.title("GPU Inference Memory Calculator for LLM Serving")

    st.markdown("""
    This tool estimates the GPU memory requirements for LLM inference based on the vLLM serving framework, which leverages FlashAttention for efficiency. 

    **Note:** Results may vary depending on the specific GPU hardware and configuration, and some deviation from actual usage is expected.
    """)

    # Calculate
    if calculate_button:
        if not model_name:
            st.error("Please enter a valid model name.")
        else:
            required_memory, explanation, process_steps = calculate_required_gpu_memory(
                model_name, concurrent_users, seq_len, gpu_utilization, data_type, kv_data_type, huggingface_key
            )
            if required_memory is None:
                st.error(process_steps)
            else:
                st.markdown("---")   
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(33, 195, 84, 0.1);
                        padding: 10px;
                        border-radius: 5px;
                        color: rgb(23, 114, 51);
                        font-size: 24px;
                        font-weight: bold;
                    ">
                        Estimated Required GPU Memory: {required_memory:.3f} GB
                    </div>
                    """,
                    unsafe_allow_html=True
                )        
                st.text("")
                st.markdown("##### Calculation Details")
                st.json(explanation)
                st.markdown(process_steps)