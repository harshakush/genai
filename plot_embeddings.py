import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load model and tokenizer
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Token embeddings
token_embeddings = model.base_model.embed_tokens.weight.detach().cpu()  # shape: [vocab_size, hidden_dim]
vocab_size, hidden_dim = token_embeddings.shape
head_dim = hidden_dim // 2  # RoPE applies to even dims in pairs

# Simulate RoPE effect: apply rotation for positions 0 to 4
def apply_rope(embeddings, position):
    dim = embeddings.shape[-1]
    half_dim = dim // 2
    theta = 10000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
    angle = position * theta
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    emb_even, emb_odd = embeddings[:, 0::2], embeddings[:, 1::2]
    emb_rotated_even = emb_even * cos - emb_odd * sin
    emb_rotated_odd = emb_even * sin + emb_odd * cos
    rotated = torch.empty_like(embeddings)
    rotated[:, 0::2] = emb_rotated_even
    rotated[:, 1::2] = emb_rotated_odd
    return rotated

# Combine embeddings across positions (simulate RoPE)
augmented_embeddings = []
positions_to_simulate = [0, 1, 2, 3, 4]
for pos in positions_to_simulate:
    rope_emb = apply_rope(token_embeddings, pos)
    augmented_embeddings.append(rope_emb)

# Stack them: shape → [num_positions * vocab_size, hidden_dim]
stacked_embeddings = torch.cat(augmented_embeddings, dim=0)

# PCA to 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(stacked_embeddings.numpy())

# Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], s=1, alpha=0.2)

# Highlight tokens
highlight_words = ["Ginny", "Dumbledore", "Harry", "Hermione", "Snape"]
for word in highlight_words:
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for pos_idx, pos in enumerate(positions_to_simulate):
        for subtoken, token_id in zip(tokens, token_ids):
            index = pos_idx * vocab_size + token_id
            if token_id != tokenizer.unk_token_id:
                x, y, z = embeddings_3d[index]
                ax.scatter(x, y, z, color='red', s=40)
                ax.text(x, y, z, f"{subtoken}@{pos}", fontsize=9, color='black')

ax.set_title("Token Embeddings + Simulated RoPE (3D PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.tight_layout()
plt.show()
