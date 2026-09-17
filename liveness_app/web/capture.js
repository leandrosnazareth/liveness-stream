"use strict";
const byId = id => document.getElementById(id);
const video = byId("video");
const canvas = byId("canvas");
const ctx = canvas.getContext("2d");
// The upload canvas never contains labels or other UI overlays.
const capture = document.createElement("canvas");
capture.width = 640;
capture.height = 480;
const captureCtx = capture.getContext("2d");
let session = null;
let sequence = 1;
let mode = "metadata";
let busy = false;
let faces = [];
let camera = null;
let starting = false;
let expiresAt = 0;
let lastVideoTime = -1;

const reasons = {
    MODELOS_NAO_VALIDADOS: "Modelos ainda nao validados. Captura indisponivel.",
    DEMO_DESABILITADA: "A demonstracao esta desabilitada neste ambiente.",
    VIVACIDADE_VERIFICADA: "Verificacao de vivacidade concluida.",
    QUALIDADE_INSUFICIENTE: "Ajuste a iluminacao e mantenha o rosto inteiro visivel.",
    DESAFIO_PENDENTE: "Siga a instrucao exibida.",
    COLETANDO_EVIDENCIAS: "Analisando a captura...",
    MULTIPLAS_FACES: "Mantenha apenas uma pessoa diante da camera.",
    FRAME_REPETIDO: "A imagem parou de atualizar. Inicie uma nova captura.",
    FACE_PERDIDA: "O rosto saiu do enquadramento. Inicie uma nova captura.",
};
function message(text) { byId("contador").textContent = text; }
function stop() {
    session = null;
    faces = [];
    if (camera) camera.getTracks().forEach(track => track.stop());
    camera = null;
    video.srcObject = null;
    byId("iniciar").disabled = false;
    byId("status-camera").textContent = "parada";
    byId("output-img").removeAttribute("src");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
async function readResponse(response) {
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "CAPTURA_INVALIDA");
    return data;
}
byId("iniciar").addEventListener("click", async () => {
    if (busy || starting) return;
    starting = true;
    stop();
    byId("iniciar").disabled = true;
    try {
        camera = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480, facingMode: "user"}, audio: false});
        video.srcObject = camera;
        await video.play();
        const data = await readResponse(await fetch("/demo/sessions", {method: "POST"}));
        session = data;
        sequence = data.next_sequence;
        expiresAt = performance.now() + data.expires_in * 1000;
        lastVideoTime = -1;
        byId("status-camera").textContent = "ativa";
        byId("desafio-ativo").textContent = data.desafio_ativo.texto;
        message("Siga as instrucoes. Mantenha o rosto inteiro no enquadramento.");
    } catch (error) {
        stop();
        message(reasons[error.message] || "Nao foi possivel iniciar a captura.");
    } finally { starting = false; }
});
for (const value of ["metadata", "imagem"]) {
    byId("modo-" + value).addEventListener("click", () => {
        mode = value;
        canvas.style.display = value === "metadata" ? "block" : "none";
        byId("output-img").style.display = value === "imagem" ? "block" : "none";
        for (const other of ["metadata", "imagem"]) byId("modo-" + other).classList.toggle("ativo", other === value);
    });
}
function render() {
    if (session && performance.now() >= expiresAt && !busy) {
        stop();
        message("O tempo de captura terminou. Inicie uma nova tentativa.");
    }
    if (camera && video.readyState >= 2 && mode === "metadata") {
        ctx.drawImage(video, 0, 0, 640, 480);
        for (const face of faces) {
            const [x1, y1, x2, y2] = face.bbox;
            ctx.strokeStyle = face.estado === "APROVADO" ? "#22c55e" : "#facc15";
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }
    }
    requestAnimationFrame(render);
}
requestAnimationFrame(render);

setInterval(async () => {
    if (!session || busy || video.readyState < 2 || video.currentTime === lastVideoTime) return;
    busy = true;
    byId("status-fila").textContent = "processando";
    const current = session;
    const started = performance.now();
    try {
        lastVideoTime = video.currentTime;
        captureCtx.drawImage(video, 0, 0, 640, 480);
        const blob = await new Promise(resolve => capture.toBlob(resolve, "image/jpeg", 0.90));
        if (!blob) throw new Error("FRAME_INVALIDO");
        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        let data;
        try {
            data = await readResponse(await fetch(`/predict?modo=${mode}`, {
                method: "POST", body: form, signal: controller.signal,
                headers: {Authorization: `Bearer ${current.capture_token}`,
                          "X-Session-ID": current.session_id, "X-Frame-Sequence": String(sequence)}
            }));
        } finally { clearTimeout(timeout); }
        sequence += 1;
        faces = data.faces || [];
        const result = data.decisao;
        message(reasons[result.motivo] || (result.estado === "EM_ANALISE" ? "Analisando..." : "Captura nao aprovada. Inicie uma nova tentativa."));
        byId("status-rostos").textContent = data.quantidade_rostos;
        byId("provider-ativo").textContent = data.provider_ativo;
        byId("desafio-ativo").textContent = `${data.desafio_ativo.texto} (${data.desafio_ativo.segundos_restantes}s)`;
        byId("desafio-status").textContent = data.desafio_ativo.codigo === "CONCLUIDO" ? "concluido" : "pendente";
        byId("frames-temporais").textContent = `${result.frames_analisados || 0}/${data.analise_temporal.min_frames}`;
        byId("estabilidade-temporal").textContent = result.estado;
        byId("latencia-media").textContent = `${Math.round(performance.now() - started)} ms`;
        byId("tempo-decode").textContent = `${data.metricas.decode_ms} ms`;
        byId("tempo-total-backend").textContent = `${data.metricas.total_ms} ms`;
        byId("resolucao-frame").textContent = `${data.resolucao.largura}x${data.resolucao.altura}`;
        const evidence = faces[0]?.evidencias || {};
        for (const [id, key] of Object.entries({"score-face":"face_detectada", "score-profundidade":"profundidade",
            "score-textura":"textura_natural", "score-suporte-plano":"suporte_plano", "score-escala-face":"escala_face",
            "score-modelo-live":"pad_model_live", "score-modelo-print":"pad_model_print", "score-modelo-replay":"pad_model_replay"})) {
            byId(id).textContent = Number.isFinite(evidence[key]) ? evidence[key].toFixed(3) : "--";
        }
        byId("score-movimento").textContent = (result.movimento_natural || 0).toFixed(3);
        byId("score-antispoofing").textContent = result.score === null ? "--" : result.score.toFixed(3);
        if (data.imagem_processada) byId("output-img").src = data.imagem_processada;
        if (data.encerrada) stop();
    } catch (error) {
        stop(); // No automatic retries or reuse after a lost/ambiguous response.
        message(reasons[error.message] || "Captura interrompida. Inicie uma nova tentativa.");
    } finally {
        busy = false;
        byId("status-fila").textContent = "livre";
    }
}, 150);
window.addEventListener("pagehide", stop);
