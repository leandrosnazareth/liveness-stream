# Liveness Stream

Captura facial com FastAPI, InsightFace, OpenCV e PAD ONNX, com decisao conservadora
e historico isolado por sessao. O resultado indica vivacidade; a verificacao do
titular e a autorizacao financeira continuam sendo responsabilidades do backend bancario.

**O modelo distribuido ainda exige validacao independente de seu contrato e de suas
amostras de referencia. Sem isso, `/health` retorna 503 e nenhuma sessao e iniciada.**
As alteracoes de seguranca e os testes sinteticos nao demonstram a precisao contra
telas reais nem constituem certificacao para uso bancario.

## Politica implementada

- `EM_ANALISE`: coleta em andamento, sem permissao de aprovacao.
- `APROVADO`: PAD, qualidade, tempo minimo, continuidade e dois desafios satisfeitos.
- `REJEITADO`: ataque, repeticao de pixels, troca de face, multiplas faces ou sequencia invalida.
- `INCONCLUSIVO`: qualidade insuficiente, falha de modelo, interrupcao ou expiracao.

Os tres ultimos estados encerram a tentativa. Nao ha retorno automatico de uma
rejeicao para aprovacao, nem excecoes por tamanho do rosto ou oclusao. Um ataque
no frame atual tem precedencia sobre todo o historico. Ausencia de PAD nunca e
substituida por heuristicas. O score exposto e **nao calibrado**, sem piso artificial
de confianca. Os limiares iniciais sao parametros de engenharia, nao metas de seguranca comprovadas.

Geometria estimada por landmarks nao mede profundidade fisica. Movimento,
piscadas e desafios complementam o PAD; videos de tela e injeções digitais precisam
de avaliacao propria. Os hashes detectam pixels exatamente repetidos dentro da
sessao; nao autenticam o sensor e nao detectam toda recodificacao ou alteracao do video.

## Contrato e validacao do PAD

Para cada caminho em `PAD_MODEL_PATHS`, monte tres artefatos revisados:

```text
/app/models/minifasnet_v2.onnx
/app/models/minifasnet_v2.onnx.json
/app/models/reference.npz
```

O JSON define hash do modelo, nomes dos tensores, ordem das classes, canais,
divisor dos pixels, escala e modo de recorte, logits/probabilidades e a referencia.
Veja [o exemplo de formato](docs/pad-contract.example.json). **Os valores do exemplo
nao sao um contrato validado** e precisam ser substituidos pelos valores comprovados.

Ha uma divergencia a resolver: o [cartao do ONNX](https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx)
descreve `live` no indice zero e divisao por 255; o
[teste upstream](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/test.py)
usa o indice um, e a conversao de arrays em
[functional.py](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/data_io/functional.py)
retorna floats sem essa divisao. Verifique se a exportacao mudou os pesos ou a
semantica antes de escolher o contrato. O codigo nao adivinha esse mapeamento.

Gere `reference.npz` com uma implementacao upstream revisada, independente deste
adaptador, usando imagens rotuladas cuja procedencia esteja registrada. Inclua:

| Array | Formato | Conteudo |
|---|---|---|
| `frames` | `uint8 [N,H,W,3]` | Imagens BGR com mesma resolucao |
| `boxes` | `[N,4]` | Coordenadas x1,y1,x2,y2 no frame |
| `inputs` | `float32 [N,3,80,80]` | Tensores pre-processados pela referencia |
| `probabilities` | `[N,3]` | Probabilidades na ordem canonica live, print, replay |
| `labels` | string `[N]` | Rotulos live, print, replay; todos devem estar presentes |

Registre o SHA256 do NPZ no JSON, alem da revisao upstream e da procedencia das
amostras em `reference_source`. As referencias devem ter classificacao correta e
incluir recortes nao quadrados, proximos das bordas e variacao de cores para
exercitar o contrato. Use mais amostras do que o minimo estrutural de tres.
Nao gere as saidas esperadas com o proprio adaptador sob teste.

Valide, no ambiente de inferencia:

```powershell
python -m liveness_app.validate_pad /app/models/minifasnet_v2.onnx
```

Na inicializacao, todos os modelos configurados precisam passar os hashes, formas,
tipos e comparacoes numericas com a referencia. Falha em qualquer modelo bloqueia
o conjunto inteiro. Falha durante inferencia desabilita novas aprovacoes ate uma
reinicializacao com validacao bem-sucedida. O ensemble usa o menor live e o maior
ataque; nao dilui discordancias por media. Cada modelo usa apenas o recorte do seu contrato.

## Execucao

O Dockerfile original utiliza Python 3.10, CUDA e ONNXRuntime GPU e baixa o ONNX
com verificacao SHA256. O modelo sozinho nao habilita aprovacao.

```powershell
docker build -t ia-liveness-api .
# Configure a chave forte no ambiente, sem a colocar no frontend ou no repositorio.
# A pasta abaixo deve conter o ONNX, seu JSON validado e a referencia NPZ.
docker run --rm --gpus device=0 -p 127.0.0.1:8080:8000 `
  -e LIVENESS_SERVICE_API_KEY `
  --mount 'type=bind,source=C:\modelos-validados,target=/app/models,readonly' `
  ia-liveness-api
```

`LIVENESS_SERVICE_API_KEY` deve ter pelo menos 32 caracteres aleatorios. Ela autentica
o **backend confiavel** na emissao e consumo de sessoes. Use TLS, protecao de rede,
gestao de segredos e limites por conta/transacao na integracao. Nao entregue essa
chave ao navegador ou ao aplicativo cliente.

O armazenamento de sessoes e em memoria, limitado e protegido contra processamento
concorrente da mesma tentativa. **Execute um unico processo/worker e uma replica.**
Reinicializacao invalida todas as sessoes. Escalabilidade distribuida exige um store
compartilhado com reserva de frames e consumo atomicos; afinidade de carga sozinha
nao substitui esses controles. A inferencia e serializada para proteger os modelos.

## API e integracao

1. O backend autenticado cria `POST /sessions`, com `X-API-Key` e JSON
   `{"subject_id":"id-interno","transaction_id":"id-imutavel-da-transacao"}`.
   Esses identificadores devem ser definidos e autorizados pelo backend bancario.
2. A resposta entrega `session_id`, `capture_token`, `next_sequence`, `expires_in`
   e o primeiro desafio. Encaminhe apenas as credenciais dessa captura ao cliente.
3. O cliente envia JPEG em `POST /predict?modo=metadata` (ou `imagem`), multipart
   campo `file`, com `Authorization: Bearer <capture_token>`, `X-Session-ID` e
   `X-Frame-Sequence` com inteiros consecutivos a partir de 1.
4. A resposta contem `decisao.estado`, `decisao.motivo`, `encerrada` e o desafio
   atual. Apenas a etapa atual e revelada. Novos frames nao podem reutilizar
   evidencias anteriores a emissao da etapa. As direcoes usam imagem nao espelhada.
5. Apos aprovacao, o backend chama `POST /sessions/{id}/consume`, com `X-API-Key`
   e os mesmos `subject_id`/`transaction_id`. O resultado so pode ser consumido
   uma vez, dentro da validade. Token de captura nao permite consumir resultados.

O resultado retorna `autoriza_transacao: false` deliberadamente: o banco ainda
deve verificar o titular contra cadastro confiavel, confirmar a intencao e os
dados da operacao, autenticar o dispositivo e aplicar a politica antifraude.
O embedding atual apenas verifica continuidade em relacao a primeira face da
sessao; nao comprova a identidade do titular. O limiar de continuidade tambem
precisa ser validado com a populacao e os dispositivos alvo.

Erros de autenticacao retornam 401; sessao encerrada/sequencia invalida, 409;
expiracao/interrupcao, 410; upload excessivo, 413; JPEG invalido, 422; capacidade
ou cadencia, 429; indisponibilidade, 503. Respostas nao devem ser armazenadas em cache.
Uma falha de transporte exige nova tentativa controlada pelo backend: nao reenviar
automaticamente uma captura cujo processamento possa ter ocorrido.

`GET /health` retorna 200 somente com modelos obrigatorios validados; consulte
`ready`, `provider_ativo` e `pad_model_loaded`. `GET /calibration` requer a chave
do servico. Logs sao opcionais, desativados por padrao, sem imagens, embeddings ou
tokens. Sua retencao e rotacao precisam ser configuradas no ambiente de operacao.

## Demonstracao no navegador

Para habilitar o botao de demonstracao em `http://localhost:8080`, adicione
`-e LIVENESS_DEMO_MODE=true` ao comando de execucao. Isso habilita `/demo/sessions`
sem autenticacao, exclusivamente para desenvolvimento local. Essas sessoes nunca
podem ter resultados consumidos pelo backend. Os modelos validados continuam
obrigatorios. Desabilite a demonstracao em qualquer ambiente bancario.

O navegador cria uma sessao por tentativa, envia tokens apenas em headers, usa
canvas separado para captura, JPEG 0.90 e para a camera ao concluir ou falhar.
Nao reinicia tentativas automaticamente. A pagina raiz e uma demonstracao;
o aplicativo integrado deve obter sua sessao do backend autenticado.

## Parametros relevantes

| Variavel | Padrao | Finalidade |
|---|---|---|
| `PAD_MODEL_REAL_THRESHOLD` | 0.90 | Live minimo em todos os frames da janela |
| `PAD_MODEL_ATTACK_THRESHOLD` | 0.55 | Veto de ataque no frame atual |
| `TEMPORAL_MIN_FRAMES` / `TEMPORAL_WINDOW_SIZE` | 5 / 12 | Janela de evidencias |
| `TEMPORAL_MIN_SECONDS` | 1.0 | Duracao minima real, medida pelo servidor |
| `SESSION_TTL_SECONDS` | 45 | Validade total, inclusive para consumo |
| `SESSION_IDLE_SECONDS` | 3.0 | Intervalo maximo entre frames / processamento |
| `SESSION_MIN_FRAME_SECONDS` | 0.08 | Intervalo minimo entre frames |
| `SESSION_MAX_FRAMES` / `SESSION_MAX_COUNT` | 300 / 1000 | Limites de recursos |
| `ACTIVE_CHALLENGE_SECONDS` | 15 | Prazo por etapa |
| `IDENTITY_MIN_COSINE` | 0.65 | Continuidade com a face inicial |
| `MAX_UPLOAD_BYTES` | 2097152 | Limite do JPEG |
| `MAX_IMAGE_PIXELS` | 2073600 | Limite verificado antes da decodificacao |

O fluxo anterior sem sessao deixou de ser suportado. Foram removidos os atalhos
e parametros `PAD_REAL_THRESHOLD`, `PAD_SPOOF_THRESHOLD`,
`TEMPORAL_REAL_THRESHOLD`, `TEMPORAL_SPOOF_THRESHOLD`,
`PAD_LARGE_REAL_FACE_SCALE`, `PAD_PARTIAL_REAL_FACE_SCALE`,
`PAD_MODEL_CROP_SCALES` e `ACTIVE_CHALLENGE_ENABLED`. Desafios sao obrigatorios.
O CSV mudou de formato; use um novo `CALIBRATION_LOG_PATH` ao ativar o log.

## Testes e limites da validacao

```powershell
docker build -f dockerfile.test -t liveness-stream-tests .
docker run --rm liveness-stream-tests
node --check liveness_app/web/capture.js
```

Alternativa local com Python 3.10: instale `requirements-test.txt` e execute
`python -m pytest -q`. Os testes exercitam as regras e a API com inferencia simulada,
incluindo um caminho completo de aprovacao, consumo, isolamento, ataque forte,
PAD indisponivel, continuidade, validade e contratos invalidos. Nao baixam pesos
faciais nem simulam uma certificacao de resistencia a ataques.

Antes de liberar uso financeiro, faltam referencias reais do ONNX, homologacao
GPU/cameras, avaliacao de APCER/BPCER por tipo de ataque e dispositivo, testes de
injecao digital e integracao com autenticacao do titular e dispositivo. Inclua
telas proximas preenchendo o quadro, sem bordas visiveis, e separe pessoas,
videos e dispositivos entre desenvolvimento e teste. Verifique tambem a licenca
comercial dos pesos InsightFace usados; o projeto upstream descreve restricoes
para [modelos pre-treinados](https://github.com/deepinsight/insightface).

## Autor

Leandro Nazareth — [LinkedIn](https://www.linkedin.com/in/leandrosnazareth/)
