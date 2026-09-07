# Real-Time Multi-Face Liveness Detection com Docker e NVIDIA CUDA

Este projeto e uma aplicacao de Inteligencia Artificial para deteccao de vivacidade
(Liveness Detection / Anti-Spoofing) em tempo real. A aplicacao captura o feed da
webcam pelo navegador, envia frames para uma API FastAPI, detecta multiplos rostos
com InsightFace e classifica cada rosto com uma decisao principal binaria de PAD
(Presentation Attack Detection): `REAL` ou `SPOOF`.

## Funcionalidades

- Deteccao multi-rosto em tempo real.
- Bounding boxes coloridas por face detectada.
- Analise independente por rosto.
- Analise temporal por face para reduzir oscilacoes entre frames.
- Fusao de evidencias de profundidade, textura, cor, movimento e sinais de ataque.
- Modelo dedicado MiniFASNet V2 ONNX para anti-spoofing visual.
- Ensemble de crops do modelo PAD em diferentes escalas.
- Desafio ativo simples para coletar evidencias de piscada e movimento.
- Log CSV de calibracao com scores por face.
- Scores separados para deteccao facial e anti-spoofing.
- Uso de InsightFace com ONNXRuntime GPU.
- Execucao em container Docker com suporte a NVIDIA CUDA/cuDNN.

## Tecnologias

- Python 3.10
- FastAPI e Uvicorn
- OpenCV
- InsightFace
- ONNXRuntime GPU
- MiniFASNet V2 ONNX para PAD
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

A deteccao facial e o liveness sao tratados como etapas diferentes. O score do
detector de face indica apenas se existe um rosto na imagem; ele nao e usado como
prova suficiente de pessoa real.

A classificacao final combina:

- qualidade da deteccao facial;
- profundidade estimada por landmarks 3D;
- textura e nitidez do recorte facial;
- distribuicao de cor, saturacao e brilho;
- sinais provaveis de foto impressa;
- suporte plano ao redor da face, como folha, cartaz ou impressao;
- sinais provaveis de tela ou monitor;
- movimento temporal, incluindo deslocamento, escala, profundidade e olhos;
- paralaxe 3D temporal;
- resposta ao desafio ativo;
- estabilidade dos ultimos frames da mesma face.

Para retornar `REAL`, o sistema exige evidencias suficientes de profundidade,
textura, cor e movimento natural ao longo de varios frames. Quando as evidencias
sao fracas, conflitantes ou ainda insuficientes, o resultado principal passa a ser
`SPOOF`, evitando falso `REAL 100%` apenas por haver um rosto detectado.

O tipo especifico da apresentacao fica em um campo separado de diagnostico:
`tipo_apresentacao`, podendo indicar `FOTO`, `TELA`, `SPOOF`, `INCERTO` ou
`PRESENCA_FISICA`.

## Scores Retornados

Cada face pode retornar campos como:

```json
{
  "label": "SPOOF",
  "tipo_apresentacao": "FOTO",
  "confianca": 0.91,
  "face_score": 0.99,
  "anti_spoofing_score": 0.12,
  "evidencias": {
    "face_detectada": 0.99,
    "profundidade": 0.18,
    "textura_natural": 0.24,
    "cor_natural": 0.42,
    "movimento_natural": 0.05,
    "paralaxe_3d": 0.04,
    "desafio_ativo_score": 0.12,
    "anti_spoofing": 0.12,
    "foto_score": 0.91,
    "tela_score": 0.22,
    "suporte_plano": 0.88,
    "pad_model_live": 0.08,
    "pad_model_print": 0.89,
    "pad_model_replay": 0.03,
    "pad_model_attack": 0.92
  },
  "desafio_ativo": {
    "codigo": "PISQUE",
    "texto": "Pisque",
    "segundos_restantes": 7
  },
  "desafio_ok": false
}
```

## Ajustes de Liveness/PAD

Os limiares podem ser ajustados por variaveis de ambiente no `docker run`:

```powershell
docker run -d --rm --gpus device=0 -p 8080:8000 `
  --name ia-liveness-api-gpu `
  -e TEMPORAL_WINDOW_SIZE=12 `
  -e TEMPORAL_MIN_FRAMES=5 `
  -e TEMPORAL_STABILITY_DELTA=0.18 `
  -e PAD_REAL_THRESHOLD=0.72 `
  -e PAD_SPOOF_THRESHOLD=0.42 `
  -e PAD_MIN_MOTION_SCORE=0.18 `
  -e PAD_STRONG_ATTACK_SCORE=0.62 `
  -e PAD_FLAT_SUPPORT_SCORE=0.48 `
  -e PAD_MIN_REAL_FACE_SCALE=0.23 `
  -e PAD_LARGE_REAL_FACE_SCALE=0.30 `
  -e PAD_MODEL_CROP_SCALES=2.0,2.7,3.4 `
  -e PAD_MODEL_REAL_THRESHOLD=0.72 `
  -e PAD_MODEL_ATTACK_THRESHOLD=0.55 `
  -e ACTIVE_CHALLENGE_MIN_SCORE=0.45 `
  -e CALIBRATION_LOG_ENABLED=true `
  -e ACTIVE_CHALLENGE_ENABLED=true `
  ia-liveness-api
```

Valores mais altos em `PAD_REAL_THRESHOLD` e `PAD_MIN_MOTION_SCORE` deixam a
decisao `REAL` mais conservadora. Valores menores em `PAD_STRONG_ATTACK_SCORE`
fazem o sistema marcar `FOTO` ou `TELA` com menos evidencia acumulada.
Valores menores em `PAD_FLAT_SUPPORT_SCORE` deixam a deteccao de foto impressa
em folha/cartaz mais sensivel.
`PAD_LARGE_REAL_FACE_SCALE` permite recuperar um rosto fisico grande parcialmente
ocluido, evitando que uma folha na frente da boca transforme automaticamente a
face real em `SPOOF`.

## Modelo Anti-Spoofing Dedicado

Durante o build, o Docker baixa o modelo `minifasnet_v2.onnx` para:

```text
/app/models/minifasnet_v2.onnx
```

Esse modelo recebe um crop BGR da face em `80x80` com margem de `2.7x` e retorna
tres probabilidades:

- `pad_model_live`: probabilidade de face real;
- `pad_model_print`: probabilidade de ataque por foto impressa;
- `pad_model_replay`: probabilidade de replay/tela.

Quando o modelo esta carregado, a classificacao `REAL` tambem exige
`pad_model_live >= PAD_MODEL_REAL_THRESHOLD` e
`pad_model_attack < PAD_MODEL_ATTACK_THRESHOLD`.

Por padrao, o mesmo modelo e executado com crops em multiplas escalas
(`PAD_MODEL_CROP_SCALES=2.0,2.7,3.4`) e a media das probabilidades e usada na
decisao. Tambem e possivel informar varios modelos separados por virgula em
`PAD_MODEL_PATHS`.

## Calibracao

A cada face processada, a aplicacao grava um CSV com as evidencias usadas na
decisao:

```text
/app/logs/liveness_calibration.csv
```

Para consultar as ultimas linhas pela API:

```text
GET /calibration?limit=20
```

Esse log ajuda a calibrar os limiares comparando exemplos reais, fotos impressas,
telas e replays.

## Desafio Ativo

Quando `ACTIVE_CHALLENGE_ENABLED=true`, a API alterna pequenos desafios, como:

- piscar;
- virar a cabeca;
- aproximar o rosto.

O desafio atual aparece no painel e tambem e retornado no JSON em
`desafio_ativo`. O resultado por face aparece em `desafio_ok` e no score
`desafio_ativo_score`.

## Autor

Desenvolvido e mantido por Leandro Nazareth.

LinkedIn: https://www.linkedin.com/in/leandrosnazareth/
