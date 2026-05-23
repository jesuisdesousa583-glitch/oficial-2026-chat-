/**
 * Baileys WhatsApp Sidecar for LegalFlow / Espírito Santo AI
 * - Connects to WhatsApp via QR (Baileys)
 * - Exposes minimal HTTP API for the FastAPI backend
 * - Forwards inbound messages to the backend webhook
 *
 * FIX 2026-01: WhatsApp now routes new/unknown contacts via @lid (anonymous
 * linked-device identifiers). Previously we derived a phone from the lid jid
 * and rebuilt ${digits}@s.whatsapp.net to reply — which created NEW fake
 * conversations (the "+8955..." phantom number bug). Now we always forward
 * the original remoteJid to the backend and reply back to that exact jid.
 */
const path = require("path");
const express = require("express");
const QRCode = require("qrcode");
const axios = require("axios");
const pino = require("pino");
const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  Browsers,
  downloadMediaMessage,
  jidNormalizedUser,
  isJidUser,
  isLidUser,
} = require("@whiskeysockets/baileys");

const PORT = parseInt(process.env.BAILEYS_PORT || "8002", 10);
const AUTH_DIR = process.env.BAILEYS_AUTH_DIR || path.join(__dirname, "auth_info");
const INTERNAL_TOKEN = process.env.BAILEYS_INTERNAL_TOKEN || "legalflow-baileys-2026";
const BACKEND_WEBHOOK =
  process.env.BACKEND_WEBHOOK ||
  "http://localhost:8001/api/whatsapp/webhook/baileys";

const logger = pino({ level: "warn" });

let sock = null;
let qrRaw = null; // string from baileys
let qrDataUri = null; // base64 image
let qrGeneratedAt = null; // timestamp ms (Date.now()) — para mostrar idade do QR
let connectionState = "close"; // 'close' | 'connecting' | 'open' | 'conflicted'
let lastError = null;
let me = null;
let reconnectAttempts = 0;
const MAX_BACKOFF_MS = 60_000; // max 1min between tries

// phone(digits) -> last-seen original remoteJid (for reply routing).
// Essential to handle @lid contacts: replying to a rebuilt
// `${digits}@s.whatsapp.net` creates a NEW fake chat instead of continuing
// the original conversation. We reply to the exact jid we received from.
const jidRouteCache = new Map();

function jidToPhone(jid) {
  if (!jid) return "";
  return String(jid).split("@")[0].split(":")[0];
}

/**
 * Extract the best phone identifier for a given message.
 * For @lid messages (new WhatsApp anonymous routing), the real phone jid
 * may live in senderPn / participantPn / keyAlt. We fall back to the lid
 * digits ONLY for display — routing always uses the original jid.
 */
function resolvePhoneAndJid(msg) {
  const key = msg.key || {};
  const remoteJid = key.remoteJid || "";
  // Prefer a verified phone jid when present (newer Baileys fields)
  const senderPn = msg.senderPn || key.senderPn || null;
  const participantPn = msg.participantPn || key.participantPn || null;
  const altJid = key.remoteJidAlt || null;
  let phoneJid = null;
  for (const cand of [senderPn, participantPn, altJid]) {
    if (cand && typeof cand === "string" && cand.endsWith("@s.whatsapp.net")) {
      phoneJid = cand;
      break;
    }
  }
  // Phone for display/storage: prefer the real phone jid, else lid digits
  const displayJid = phoneJid || remoteJid;
  const phone = jidToPhone(displayJid);
  return {
    phone,
    remoteJid, // where to REPLY (original routing, never rebuilt)
    phoneJid, // real phone jid if resolvable
    isLid: remoteJid.endsWith("@lid"),
  };
}

async function startSock() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion().catch(() => ({
    version: [2, 3000, 1015901307],
  }));

  sock = makeWASocket({
    version,
    auth: state,
    printQRInTerminal: false,
    browser: Browsers.appropriate("LegalFlow"),
    logger,
    syncFullHistory: false,
    markOnlineOnConnect: false,
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", async (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      qrRaw = qr;
      try {
        qrDataUri = await QRCode.toDataURL(qr, { margin: 1, scale: 6 });
      } catch (e) {
        qrDataUri = null;
      }
      qrGeneratedAt = Date.now();
      connectionState = "connecting";
    }
    if (connection) {
      connectionState = connection;
    }
    if (connection === "open") {
      qrRaw = null;
      qrDataUri = null;
      qrGeneratedAt = null;
      lastError = null;
      reconnectAttempts = 0; // reset backoff on success
      me = sock.user || null;
      console.log("[baileys] connected as", me?.id);
    }
    if (connection === "close") {
      const code = lastDisconnect?.error?.output?.statusCode;
      lastError = lastDisconnect?.error?.message || null;
      const loggedOut = code === DisconnectReason.loggedOut;
      // Detecta "stream replaced" (code 440 / conflict:replaced)
      const replaced = code === 440 || /replaced|conflict/i.test(lastError || "");
      // Detecta "QR refs attempts ended" — Baileys gerou 5 QRs sem scan; sessao morta
      const qrEnded = /QR refs attempts ended|QR refs/i.test(lastError || "");
      console.log("[baileys] connection closed", { code, loggedOut, replaced, qrEnded, attempts: reconnectAttempts });
      if (loggedOut || qrEnded) {
        // qrEnded é tratado igual loggedOut: limpa auth e re-inicia fresco
        console.log("[baileys] resetting session (logged out or QR expired)");
        try {
          const fs = require("fs");
          fs.rmSync(AUTH_DIR, { recursive: true, force: true });
          fs.mkdirSync(AUTH_DIR, { recursive: true });
        } catch (_) {}
        qrRaw = null;
        qrDataUri = null;
        qrGeneratedAt = null;
        me = null;
        reconnectAttempts = 0;
        setTimeout(startSock, 1500);
      } else if (replaced) {
        console.warn("[baileys] SESSION CONFLICT: outra instancia Baileys esta logada. Pausando reconexao.");
        connectionState = "conflicted";
        lastError = "Sessão em uso em outro servidor. Desconecte o outro Baileys ou clique em Logout aqui e escaneie o QR novamente.";
        me = null;
      } else {
        reconnectAttempts += 1;
        const delay = Math.min(MAX_BACKOFF_MS, 3000 * Math.pow(2, Math.min(reconnectAttempts - 1, 5)));
        console.log(`[baileys] reconectando em ${Math.round(delay/1000)}s (tentativa ${reconnectAttempts})`);
        setTimeout(startSock, delay);
      }
    }
  });

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;
    for (const msg of messages) {
      try {
        if (!msg.message) continue;
        if (msg.key?.fromMe) continue;
        const remote = msg.key?.remoteJid || "";
        if (remote.endsWith("@g.us")) continue; // ignore groups
        if (remote.endsWith("@broadcast")) continue;
        if (remote === "status@broadcast") continue;

        const { phone, remoteJid, phoneJid, isLid } = resolvePhoneAndJid(msg);
        if (!phone) continue;

        // Cache routing so /send-text can reply to the EXACT original jid
        // (prevents creating phantom chats with lid-derived fake numbers)
        jidRouteCache.set(phone, remoteJid);
        if (phoneJid) jidRouteCache.set(jidToPhone(phoneJid), remoteJid);

        const m = msg.message;
        let text = "";
        let audioBase64 = null;
        let audioMime = null;
        let imageBase64 = null;
        let imageMime = null;
        let imageCaption = null;
        if (m.conversation) text = m.conversation;
        else if (m.extendedTextMessage?.text) text = m.extendedTextMessage.text;
        else if (m.imageMessage) {
          imageCaption = m.imageMessage.caption || null;
          text = imageCaption || "[Imagem recebida]";
          try {
            const buf = await downloadMediaMessage(msg, "buffer", {}, { logger });
            imageBase64 = Buffer.from(buf).toString("base64");
            imageMime = m.imageMessage.mimetype || "image/jpeg";
          } catch (e) {
            console.warn("[baileys] image download failed:", e.message);
          }
        }
        else if (m.audioMessage) {
          try {
            const buf = await downloadMediaMessage(msg, "buffer", {}, { logger });
            audioBase64 = Buffer.from(buf).toString("base64");
            audioMime = m.audioMessage.mimetype || "audio/ogg; codecs=opus";
            text = "[Áudio recebido]";
          } catch (e) {
            console.warn("[baileys] audio download failed:", e.message);
            text = "[Áudio recebido - falha ao baixar]";
          }
        }
        else if (m.documentMessage) {
          // tambem baixa documentos (PDFs, etc) se forem imagens convertidas
          const docMime = m.documentMessage.mimetype || "";
          if (docMime.startsWith("image/")) {
            try {
              const buf = await downloadMediaMessage(msg, "buffer", {}, { logger });
              imageBase64 = Buffer.from(buf).toString("base64");
              imageMime = docMime;
              imageCaption = m.documentMessage.caption || m.documentMessage.fileName || null;
              text = imageCaption || "[Documento-imagem recebido]";
            } catch (e) {
              text = "[Documento recebido]";
            }
          } else {
            text = `[Documento: ${m.documentMessage.fileName || "arquivo"}]`;
          }
        }
        else if (m.videoMessage) text = "[Vídeo recebido]";
        else if (m.stickerMessage) text = "[Sticker]";
        else continue;

        const payload = {
          token: INTERNAL_TOKEN,
          phone,
          jid: remoteJid,
          phone_jid: phoneJid || null,
          is_lid: isLid,
          name: msg.pushName || phone,
          text,
          audio_base64: audioBase64,
          audio_mime: audioMime,
          image_base64: imageBase64,
          image_mime: imageMime,
          image_caption: imageCaption,
          message_id: msg.key.id,
          timestamp: msg.messageTimestamp,
        };
        try {
          await axios.post(BACKEND_WEBHOOK, payload, { timeout: 20000, maxBodyLength: 50 * 1024 * 1024 });
        } catch (err) {
          console.warn("[baileys] forward to backend failed:", err.message);
        }
      } catch (e) {
        console.error("[baileys] msg handler error", e);
      }
    }
  });
}

// HTTP API
const app = express();
app.use(express.json({ limit: "4mb" }));

function authMiddleware(req, res, next) {
  const token = req.header("x-internal-token");
  if (token !== INTERNAL_TOKEN) {
    return res.status(401).json({ ok: false, error: "unauthorized" });
  }
  next();
}

app.get("/health", (_req, res) => res.json({ ok: true, service: "baileys" }));

app.get("/status", authMiddleware, (_req, res) => {
  res.json({
    ok: true,
    connected: connectionState === "open",
    state: connectionState,
    me: me ? { id: me.id, name: me.name } : null,
    has_qr: !!qrDataUri,
    last_error: lastError,
  });
});

app.get("/qr", authMiddleware, (_req, res) => {
  const qrAgeMs = qrGeneratedAt ? Date.now() - qrGeneratedAt : null;
  res.json({
    ok: true,
    connected: connectionState === "open",
    state: connectionState,
    qr: qrDataUri,
    qr_raw: qrRaw,
    qr_age_s: qrAgeMs !== null ? Math.round(qrAgeMs / 1000) : null,
    qr_expires_in_s: qrAgeMs !== null ? Math.max(0, 60 - Math.round(qrAgeMs / 1000)) : null,
  });
});

/**
 * POST /restart — força nova sessão (apaga auth_info + reinicia).
 * Útil quando QR expirou várias vezes ou Baileys ficou travado.
 * IDÊNTICO a /logout mas com nome mais claro pra UX.
 */
app.post("/restart", authMiddleware, async (_req, res) => {
  try {
    if (sock) {
      try { await sock.end(); } catch (_) {}
    }
    const fs = require("fs");
    try {
      fs.rmSync(AUTH_DIR, { recursive: true, force: true });
      fs.mkdirSync(AUTH_DIR, { recursive: true });
    } catch (_) {}
    qrRaw = null;
    qrDataUri = null;
    qrGeneratedAt = null;
    connectionState = "close";
    lastError = null;
    me = null;
    reconnectAttempts = 0;
    jidRouteCache.clear();
    setTimeout(startSock, 1000);
    res.json({ ok: true, message: "Sessão zerada. Em ~5s aparece um novo QR." });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/**
 * Resolve the best JID to send to.
 * Priority:
 *   1) explicit jid passed by caller (from the webhook payload) — e.g. @lid
 *   2) cached jid from the last inbound message for this phone
 *   3) fallback ${digits}@s.whatsapp.net (only valid for real phone numbers)
 */
function resolveSendJid({ jid, phone }) {
  if (jid && typeof jid === "string" && jid.includes("@")) return jid;
  const digits = String(phone || "").replace(/\D/g, "");
  if (!digits) return null;
  const cached = jidRouteCache.get(digits);
  if (cached) return cached;
  return `${digits}@s.whatsapp.net`;
}

app.post("/send-text", authMiddleware, async (req, res) => {
  if (!sock || connectionState !== "open") {
    return res.status(503).json({ ok: false, error: "not-connected", state: connectionState });
  }
  const { phone, text, jid: explicitJid } = req.body || {};
  if ((!phone && !explicitJid) || !text) {
    return res.status(400).json({ ok: false, error: "phone|jid and text required" });
  }
  try {
    const jid = resolveSendJid({ jid: explicitJid, phone });
    if (!jid) {
      return res.status(400).json({ ok: false, error: "could not resolve target jid" });
    }
    const r = await sock.sendMessage(jid, { text });
    res.json({ ok: true, message_id: r?.key?.id, jid });
  } catch (e) {
    console.error("[baileys] send error", e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post("/send-audio", authMiddleware, async (req, res) => {
  if (!sock || connectionState !== "open") {
    return res.status(503).json({ ok: false, error: "not-connected", state: connectionState });
  }
  const { phone, audio_base64, mime, jid: explicitJid } = req.body || {};
  if ((!phone && !explicitJid) || !audio_base64) {
    return res.status(400).json({ ok: false, error: "phone|jid and audio_base64 required" });
  }
  try {
    const jid = resolveSendJid({ jid: explicitJid, phone });
    if (!jid) {
      return res.status(400).json({ ok: false, error: "could not resolve target jid" });
    }
    let buf = Buffer.from(audio_base64, "base64");
    // FIX MOBILE: WhatsApp NO CELULAR só toca voice notes em OGG/OPUS.
    // MP3 com ptt:true funciona no WhatsApp Web/Desktop mas não toca no Android/iOS.
    // Convertemos qualquer formato pra OGG/OPUS antes de enviar como PTT.
    try {
      const { spawn } = require("child_process");
      const oggBuf = await new Promise((resolve, reject) => {
        const ff = spawn("ffmpeg", [
          "-i", "pipe:0",
          "-c:a", "libopus",
          "-b:a", "32k",       // bitrate típico de voice note WhatsApp
          "-ar", "48000",      // sample rate exigido pelo OPUS
          "-ac", "1",          // mono
          "-vn",
          "-f", "ogg",
          "pipe:1",
        ], { stdio: ["pipe", "pipe", "pipe"] });
        const chunks = [];
        let stderr = "";
        ff.stdout.on("data", (c) => chunks.push(c));
        ff.stderr.on("data", (c) => stderr += c.toString());
        ff.on("error", (e) => reject(e));
        ff.on("close", (code) => {
          if (code === 0) resolve(Buffer.concat(chunks));
          else reject(new Error(`ffmpeg exited ${code}: ${stderr.slice(-300)}`));
        });
        ff.stdin.write(buf);
        ff.stdin.end();
      });
      buf = oggBuf;
      console.log(`[baileys] audio converted: in=${audio_base64.length}b base64 -> out=${oggBuf.length}b ogg/opus`);
    } catch (convErr) {
      console.warn("[baileys] ffmpeg conversion failed, sending original buffer:", convErr.message);
      // Cai pro envio original (vai tocar no desktop mas pode falhar no mobile)
    }
    const r = await sock.sendMessage(jid, {
      audio: buf,
      mimetype: "audio/ogg; codecs=opus",
      ptt: true,
    });
    res.json({ ok: true, message_id: r?.key?.id, jid });
  } catch (e) {
    console.error("[baileys] send-audio error", e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post("/logout", authMiddleware, async (_req, res) => {
  try {
    if (sock) {
      try {
        await sock.logout();
      } catch (_) {}
    }
    const fs = require("fs");
    try {
      fs.rmSync(AUTH_DIR, { recursive: true, force: true });
      fs.mkdirSync(AUTH_DIR, { recursive: true });
    } catch (_) {}
    qrRaw = null;
    qrDataUri = null;
    connectionState = "close";
    me = null;
    jidRouteCache.clear();
    setTimeout(startSock, 1500);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[baileys] HTTP API listening on :${PORT}`);
  startSock().catch((e) => {
    console.error("[baileys] start failed:", e);
    setTimeout(startSock, 5000);
  });
});

process.on("uncaughtException", (e) => console.error("uncaught:", e));
process.on("unhandledRejection", (e) => console.error("unhandled:", e));
