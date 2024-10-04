import torch
import math

# set default device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def naive_attention(Q, K, V, dropout_p = 0.0):
    assert Q.shape == K.shape == V.shape
    assert dropout_p == 0.0, "dropout not implemented"
    N, d = Q.shape
    scale_factor = math.sqrt(d)
    bias = torch.zeros(N, N)
    mask = torch.ones(N, N, dtype=torch.bool).tril(diagonal=0)
    bias = bias.masked_fill(mask.logical_not(), float("-inf"))
    attn = (Q @ K.T) / scale_factor
    attn += bias
    attn = torch.softmax(attn, dim = -1)
    O = attn @ V
    return O

def torch_flash_attention(Q,K,V, dropout_p = 0.0):
    assert Q.shape == K.shape == V.shape
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, dropout_p = dropout_p, is_causal = True)


def time_wrapper(func, *args, **kwargs):
    import time
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    # sync 
    torch.cuda.synchronize()
    return end - start



def test_attention():
    N = 128
    d = 64
    Q = torch.randn(N, d, device=device)
    K = torch.randn(N, d, device=device)
    V = torch.randn(N, d, device=device)
    O1 = naive_attention(Q, K, V)
    O2 = torch_flash_attention(Q, K, V)
    assert torch.allclose(O1, O2)

def benchmark_attention():
    N = 2048
    d = 2048
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    print("Naive attention: ", time_wrapper(naive_attention, Q, K, V))
    print("Torch attention: ", time_wrapper(torch_flash_attention, Q, K, V))

benchmark_attention()