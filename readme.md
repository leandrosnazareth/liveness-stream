# Real-Time Multi-Face Liveness Detection (Anti-Spoofing) with Docker and NVIDIA CUDA

Este projeto consiste em uma aplicação de Inteligência Artificial para Detecção de Vivacidade (Liveness Detection / Anti-Spoofing) operando em tempo real. O sistema é capaz de capturar o feed de vídeo da webcam do usuário através do navegador, processar múltiplos rostos simultaneamente utilizando aceleração por hardware (GPU dedicada) dentro de um container Docker e diferenciar se o rosto pertence a uma pessoa real ou a uma fraude (foto impressa ou tela de dispositivo).

## Funcionalidades

- Detecção Multi-Rosto: Identifica e rastreia múltiplos rostos simultaneamente na cena.
- Bounding Boxes em Tempo Real: Desenha quadros delimitadores coloridos diretamente ao redor de cada rosto detectado.
- Análise de Liveness Independente: Avalia individualmente cada face, retornando a etiqueta REAL (borda verde) ou SPOOF / FOTO (borda vermelha).
- Abordagem Híbrida Inteligente: Combina redes neurais profundas de mapeamento geométrico 3D (para validar profundidade volumétrica da face) com análise de textura nos canais de cores HSV (para identificar reflexos e saturação artificial de telas).
- Processamento na GPU via Docker: Arquitetura otimizada para rodar em servidores isolados utilizando drivers NVIDIA CUDA.

## Tecnologias e Infraestrutura Utilizadas

- Back-end / Engine de IA: Python 3.10, FastAPI, Uvicorn, OpenCV e PyTorch.
- Modelos de Visão Computacional: InsightFace (Engine Neural profunda para análise facial 3D).
- Infraestrutura e Deploy: Docker Desktop (WSL2) utilizando a imagem oficial de runtime da NVIDIA Corporation.
- Hardware de Validação Host: Laptop com GPU Dedicada NVIDIA GeForce RTX 3050 Ti (4GB VRAM).

## Estrutura do Projeto

liveness-stream/
├── app.py              # Backend FastAPI com lógica de IA e Interface Web (HTML5/JS)
├── dockerfile          # Configuração do ambiente isolado Ubuntu com suporte a CUDA 11.8
└── requirements.txt    # Dependências estruturais do ecossistema Python

## Pre-requisitos no Sistema Hospedeiro (Host)

1. Docker Desktop instalado com suporte a instâncias WSL2 ativado.
2. NVIDIA Drivers atualizados no seu Windows.
3. NVIDIA Container Toolkit instalado e configurado (o ecossistema do WSL2 geralmente mapeia os drivers automaticamente se o Docker Desktop estiver atualizado).
4. Certificar-se de que o subsistema Linux está atualizado executando no PowerShell:
   wsl --update

## Como Compilar e Executar o Projeto

Siga os passos abaixo utilizando o terminal PowerShell do Windows dentro do diretório do projeto:

1. Limpar Resíduos e Portas Presas
Caso algum container antigo ou processo fantasma esteja utilizando a porta do servidor, execute o comando de limpeza:
docker stop $(docker ps -q) 2>$null

2. Construir a Imagem Docker
Compile a imagem isolada do container com todas as dependências de Deep Learning:
docker build -t ia-liveness-api .

3. Iniciar o Container acoplado à GPU NVIDIA (RTX 3050 Ti)
Execute o container mapeando a porta interna 8000 para a porta local 8080 do seu Windows, instruindo o ecossistema a carregar as dependências de aceleração gráfica direto no índice padrão da sua GPU dedicada (device=0):
docker run --rm -p 8080:8000 --gpus device=0 ia-liveness-api

Nota: Na primeira execução, a inteligência neural do InsightFace fará o download automatizado dos pesos oficiais do modelo buffalo_l para dentro da imagem. O servidor estará pronto assim que exibir a mensagem: [*] Modelo Multi-Face Neural carregado com sucesso!.

4. Acessar a Aplicação
Abra o seu navegador de preferência (Google Chrome, Microsoft Edge, etc.) e acesse o endereço local:
http://localhost:8080

Conceda permissão de acesso à webcam quando solicitado pelo navegador e inicie os testes práticos de validação biométrica.

## Logica de Tomada de Decisao do Algoritmo

1. Validação Volumétrica (Z-Axis Variance): Rostos humanos reais possuem curvas acentuadas tridimensionais (a ponta do nariz fica muito mais próxima da lente da câmera do que as orelhas). Fotos e telas planas possuem vetor de variação geométrica tendendo a zero. O modelo calcula essa discrepância nos tensores gráficos.
2. Filtro Cromático Dinâmico (Saturate HSV Noise): Telas de smartphones ou monitores emitem luz própria polarizada, gerando micro-padrões de ruído (Moiré) e picos artificiais de saturação cromática na pele. O script analisa o recorte local da face para identificar anomalias térmicas e de iluminação reflexiva.

## Autor

Desenvolvido e mantido por:
Leandro Nazareth

LinkedIn: https://www.linkedin.com/in/leandrosnazareth/
