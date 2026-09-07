import numpy as np

def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_liveness_score(variacao_profundidade, media_saturacao):
    score_profundidade = limitar((variacao_profundidade - 8.0) / 14.0)
    score_saturacao = limitar((190.0 - media_saturacao) / 80.0)
    return limitar((score_profundidade * 0.65) + (score_saturacao * 0.35))


def normalizar_intervalo(valor, minimo, maximo):
    return limitar((valor - minimo) / (maximo - minimo))

def softmax(valores):
    valores = np.asarray(valores, dtype=np.float32)
    valores = valores - np.max(valores)
    exp = np.exp(valores)
    soma = np.sum(exp)
    return exp / soma if soma else exp


def media_deque(valores):
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0


