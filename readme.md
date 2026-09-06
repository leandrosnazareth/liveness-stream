# Real-Time Multi-Face Liveness Detection com Docker e NVIDIA CUDA

Este projeto e uma aplicacao de Inteligencia Artificial para deteccao de vivacidade
(Liveness Detection / Anti-Spoofing) em tempo real. A aplicacao captura o feed da
webcam pelo navegador, envia frames para uma API FastAPI, detecta multiplos rostos
com InsightFace e classifica cada rosto como `REAL` ou `SPOOF / FOTO`.

## Funcionalidades

- Deteccao multi-rosto em tempo real.
- Bounding boxes coloridas por face detectada.
- Analise independente por rosto.
- Uso de InsightFace com ONNXRuntime GPU.
- Execucao em container Docker com suporte a NVIDIA CUDA/cuDNN.

## Tecnologias

- Python 3.10
- FastAPI e Uvicorn
- OpenCV
- InsightFace
- ONNXRuntime GPU
- Docker Desktop com WSL2
- NVIDIA CUDA 11.8 + cuDNN 8

## Estrutura

```text
liveness-stream/
|-- app.py            # API FastAPI, inferencia e interface web
|-- dockerfile        # Imagem Docker com CUDA/cuDNN
|-- requirements.txt  # Dependencias Python
`-- readme.md         # Documentacao do projeto
```

## Pre-requisitos

1. Docker Desktop instalado e usando WSL2.
2. Driver NVIDIA atualizado no Windows.
3. Suporte a GPU habilitado no Docker.
4. WSL atualizado:

```powershell
wsl --update
```

5. Validar se a GPU aparece no host:

```powershell
nvidia-smi
```

## Como Subir o Container

Execute os comandos abaixo no PowerShell dentro da pasta do projeto:

```powershell
docker stop ia-liveness-api-gpu
docker build -t ia-liveness-api .
docker run -d --rm --gpus device=0 -p 8080:8000 --name ia-liveness-api-gpu ia-liveness-api
```

Depois acesse:

```text
http://localhost:8080
```

Se o container ainda nao existir, o primeiro comando pode retornar erro dizendo que
nao encontrou `ia-liveness-api-gpu`. Nesse caso, continue com o `docker build` e o
`docker run`.

## Verificar Uso da GPU

Confira se o container esta rodando:

```powershell
docker ps
```

Confira se o processo Python aparece na RTX:

```powershell
docker exec ia-liveness-api-gpu nvidia-smi
```

Confira os logs da aplicacao:

```powershell
docker logs --tail 80 ia-liveness-api-gpu
```

Os logs devem mostrar algo parecido com:

```text
ONNXRuntime providers disponiveis: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Se aparecer apenas `CPUExecutionProvider`, a inferencia nao esta usando a GPU.

## Observacoes Sobre GPU

O modelo de face roda via InsightFace + ONNXRuntime. Por isso, quem precisa mostrar
`CUDAExecutionProvider` e o ONNXRuntime, nao o PyTorch.

A CPU ainda pode ficar em uso alto porque algumas etapas continuam no processador:

- captura/envio dos frames pelo navegador;
- decodificacao e codificacao JPEG;
- partes do OpenCV;
- FastAPI/Uvicorn;
- trafego HTTP frequente para `/predict`.

Isso e normal. A validacao correta e ver `CUDAExecutionProvider` nos logs e o
processo Python consumindo memoria da GPU no `nvidia-smi`.

## Logica de Decisao

1. Variacao de profundidade no eixo Z: usa landmarks 3D para comparar diferencas de
   profundidade entre regioes da face.
2. Analise cromatica HSV: calcula saturacao media no recorte do rosto para ajudar a
   identificar reflexos e artefatos de tela.

## Autor

Desenvolvido e mantido por Leandro Nazareth.

LinkedIn: https://www.linkedin.com/in/leandrosnazareth/
