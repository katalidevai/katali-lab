/* Katali-GGUF local server — Apache-2.0
 *
 * A loopback-only HTTP endpoint for applications that prefer a process boundary
 * over the DLL. It is a thin wrapper over the SDK: the same model object, the
 * same session semantics, the same cancellation and the same statistics, so the
 * CLI, the SDK and the server cannot disagree about answers.
 *
 * Bind: 127.0.0.1 only (never a remote interface); the port is a CLI option.
 * Generation is serialized: one model, one session, one generation in flight.
 * A second generation request while one is running is answered with a structured
 * "busy" error rather than queueing.
 *
 * Protocol: HTTP/1.1 with a flat JSON body. Streaming responses are newline
 * delimited JSON (NDJSON), one object per line, always terminated by a "done",
 * "cancelled" or "error" line so a client can never wait forever.
 *
 *   GET  /health                 -> {"ok":true,"version":..,"abi":..}
 *   POST /load    {"model":path,"threads":n,"ctx":n}
 *   GET  /status                 -> model info + session stats
 *   POST /generate {"prompt":..,"max_tokens":n,"temperature_milli":n,
 *                   "top_k":n,"top_p_milli":n,"seed":n,"think":n,
 *                   "id":"..","stream":true|false}
 *   POST /reset                  -> drop KV state
 *   POST /cancel  {"id":".."}    -> cancel the generation with that id (or the
 *                                  current one when id is absent)
 *   POST /shutdown               -> answer, then exit
 *
 * Disconnect safety: if a client closes the socket during a streamed generation
 * the write fails, the server cancels that request, and the worker is released.
 */
#include "katali_gguf_server.h"
#include "katali_gguf_sdk.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SRV_MAX_BODY (64 * 1024)
#define SRV_MAX_LINE 4096

typedef struct ServerState {
    katali_model   *model;
    katali_session *session;
    char            model_path[1024];
    volatile LONG   shutdown;
    volatile LONG   busy;            /* a generation is in flight */
    volatile LONG   cancel_request;  /* set by /cancel */
    volatile LONG   generations;     /* generations served since start */
    volatile LONG   model_loads;     /* 1: the model is opened once at startup */
    double          uptime_start_s;
    double          last_total_s;    /* wall time of the most recent generation */
    int             last_tokens;
    char            current_id[64];
    katali_session *active_session;  /* the worker's session, for /cancel */
    CRITICAL_SECTION id_lock;
} ServerState;

/* A generation runs on its own thread so the accept loop stays responsive:
 * that is what lets POST /cancel reach a generation in flight, and what lets a
 * client disconnect cancel the request instead of leaving the worker stuck. */
typedef struct GenJob {
    SOCKET          c;
    char            id[64];
    char            body[SRV_MAX_BODY];
} GenJob;

static ServerState g_srv;

/* --- small helpers ------------------------------------------------------ */

static void json_escape(const char *s, char *out, size_t cap) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)s; p && *p && o + 7 < cap; p++) {
        switch (*p) {
            case '"':  out[o++] = '\\'; out[o++] = '"';  break;
            case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
            case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
            case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
            case '\t': out[o++] = '\\'; out[o++] = 't';  break;
            default:
                if (*p < 0x20) { o += (size_t)snprintf(out + o, cap - o, "\\u%04x", *p); }
                else out[o++] = (char)*p;
        }
    }
    if (cap) out[o < cap ? o : cap - 1] = '\0';
}

/* Extract a flat top-level string field. Supports the escapes above; returns 0
 * when the key is absent. Deliberately small: the request format is flat. */
static int json_get_str(const char *body, const char *key, char *out, size_t cap) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    size_t o = 0;
    while (*p && *p != '"' && o + 1 < cap) {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': out[o++] = '\n'; break;
                case 't': out[o++] = '\t'; break;
                case 'r': out[o++] = '\r'; break;
                case '0': out[o++] = '\0'; break;
                default:  out[o++] = *p;   break;
            }
            p++;
        } else {
            out[o++] = *p++;
        }
    }
    out[o] = '\0';
    return 1;
}

static int json_get_int(const char *body, const char *key, int dflt) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return dflt;
    p = strchr(p + strlen(pat), ':');
    if (!p) return dflt;
    return atoi(p + 1);
}

static int json_get_bool(const char *body, const char *key, int dflt) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return dflt;
    p = strchr(p + strlen(pat), ':');
    if (!p) return dflt;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return atoi(p) ? 1 : 0;
}

/* --- HTTP plumbing ------------------------------------------------------ */

static void send_all(SOCKET c, const char *p, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        int k = send(c, p + sent, (int)(n - sent), 0);
        if (k <= 0) return;
        sent += (size_t)k;
    }
}

static void send_headers(SOCKET c, int code, const char *ctype, size_t body_len) {
    char h[256];
    const char *why = code == 200 ? "OK" : code == 400 ? "Bad Request"
                    : code == 404 ? "Not Found" : code == 409 ? "Conflict"
                    : "Internal Server Error";
    int n = snprintf(h, sizeof(h),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %llu\r\n"
                     "Connection: close\r\n\r\n",
                     code, why, ctype, (unsigned long long)body_len);
    send_all(c, h, (size_t)n);
}

static void send_json(SOCKET c, int code, const char *json) {
    send_headers(c, code, "application/json", strlen(json));
    send_all(c, json, strlen(json));
}

/* Streamed responses carry no Content-Length (the body ends when the socket
 * closes), so a client reads NDJSON until EOF instead of expecting zero bytes. */
static void send_stream_headers(SOCKET c, const char *ctype) {
    char h[192];
    int n = snprintf(h, sizeof(h),
                     "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nConnection: close\r\n\r\n",
                     ctype);
    send_all(c, h, (size_t)n);
}

/* Structured error: {"ok":false,"error":{"code":"...","message":"..."}} */
static void send_error(SOCKET c, int http, const char *code, const char *msg) {
    char esc[512];
    char body[768];
    json_escape(msg, esc, sizeof(esc));
    snprintf(body, sizeof(body),
             "{\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}\n", code, esc);
    send_json(c, http, body);
}

/* Read the whole request (headers + Content-Length body). Returns 0 on success. */
static int read_request(SOCKET c, char *buf, size_t cap, size_t *out_len) {
    size_t have = 0;
    long body_len = 0;
    char *hdr_end = NULL;
    while (have + 1 < cap) {
        int k = recv(c, buf + have, (int)(cap - have - 1), 0);
        if (k <= 0) return -1;
        have += (size_t)k;
        buf[have] = '\0';
        if (!hdr_end) {
            hdr_end = strstr(buf, "\r\n\r\n");
            if (hdr_end) {
                const char *cl = strstr(buf, "Content-Length:");
                if (!cl) cl = strstr(buf, "content-length:");
                body_len = cl ? atol(cl + 15) : 0;
                if (body_len < 0 || (size_t)body_len > SRV_MAX_BODY) return -2;
            }
        }
        if (hdr_end) {
            size_t hdr = (size_t)(hdr_end - buf) + 4;
            if (have >= hdr + (size_t)body_len) break;
        }
    }
    *out_len = have;
    return 0;
}

/* --- streaming ---------------------------------------------------------- */

typedef struct StreamWr {
    SOCKET c;
    char   id[64];
    int    chunks;
    int    client_ok;
} StreamWr;

static int32_t stream_cb(const char *utf8, size_t len, void *user) {
    StreamWr *w = (StreamWr *)user;
    /* The SDK hands over at most one complete UTF-8 character per call. */
    char esc[64];
    char line[256];
    if (len >= 32) len = 31;
    char tmp[32];
    memcpy(tmp, utf8, len);
    tmp[len] = '\0';
    json_escape(tmp, esc, sizeof(esc));
    int n = snprintf(line, sizeof(line),
                     "{\"id\":\"%s\",\"type\":\"chunk\",\"text\":\"%s\"}\n", w->id, esc);
    send_all(w->c, line, (size_t)n);
    w->chunks++;
    if (InterlockedCompareExchange(&g_srv.cancel_request, 0, 0) != 0) {
        w->client_ok = 0;      /* cancelled: stop without marking a disconnect */
        return 1;
    }
    return 0;
}


/* --- endpoint handlers -------------------------------------------------- */

static void handle_health(SOCKET c) {
    char b[256];
    snprintf(b, sizeof(b),
             "{\"ok\":true,\"version\":\"%s\",\"abi\":%u,\"model_loaded\":%s}\n",
             katali_sdk_version_string(), katali_sdk_abi_version(),
             g_srv.model ? "true" : "false");
    send_json(c, 200, b);
}

static void handle_load(SOCKET c, const char *body) {
    char path[1024];
    char esc[1024];
    if (!json_get_str(body, "model", path, sizeof(path)) || !path[0]) {
        send_error(c, 400, "invalid_request", "load needs a \"model\" path");
        return;
    }
    if (InterlockedCompareExchange(&g_srv.busy, 1, 0) != 0) {
        send_error(c, 409, "busy", "a generation is running");
        return;
    }
    if (g_srv.session) { katali_sdk_session_destroy(g_srv.session); g_srv.session = NULL; }
    if (g_srv.model)   { katali_sdk_model_close(g_srv.model);       g_srv.model = NULL; }

    katali_model_options mo;
    memset(&mo, 0, sizeof(mo));
    mo.struct_size = sizeof(mo);
    mo.model_path = path;
    mo.n_threads = json_get_int(body, "threads", 0);
    mo.context_length = json_get_int(body, "ctx", 0);
    g_srv.model = katali_sdk_model_open(&mo);
    if (!g_srv.model) {
        InterlockedExchange(&g_srv.busy, 0);
        send_error(c, 500, "model_open_failed", katali_sdk_last_error());
        return;
    }
    katali_gen_options go;
    memset(&go, 0, sizeof(go));
    go.struct_size = sizeof(go);
    go.max_tokens = 256;
    g_srv.session = katali_sdk_session_create(g_srv.model, &go);
    InterlockedExchange(&g_srv.busy, 0);
    if (!g_srv.session) {
        send_error(c, 500, "session_failed", katali_sdk_last_error());
        return;
    }
    snprintf(g_srv.model_path, sizeof(g_srv.model_path), "%s", path);

    katali_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    katali_sdk_model_info(g_srv.model, &info);
    katali_sdk_stats st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    katali_sdk_session_stats(g_srv.session, &st);
    json_escape(path, esc, sizeof(esc));
    char b[1400];
    snprintf(b, sizeof(b),
             "{\"ok\":true,\"model\":{\"path\":\"%s\",\"arch\":\"%s\",\"layers\":%d,"
             "\"hidden\":%d,\"heads\":%d,\"kv_heads\":%d,\"head_dim\":%d,\"vocab\":%d,"
             "\"ctx\":%d,\"kv_cap\":%d,\"tied\":%s},\"load_s\":%.3f,"
             "\"resident_bytes\":%llu}\n",
             esc, info.arch, info.n_layers, info.hidden, info.n_heads, info.n_kv_heads,
             info.head_dim, info.vocab, info.ctx_len, info.kv_cap,
             info.tied ? "true" : "false", st.load_s,
             (unsigned long long)info.resident_bytes);
    send_json(c, 200, b);
}

static double now_wall(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static void server_note_generation(double total_s, int tokens) {
    EnterCriticalSection(&g_srv.id_lock);
    g_srv.last_total_s = total_s;
    g_srv.last_tokens = tokens;
    LeaveCriticalSection(&g_srv.id_lock);
    InterlockedIncrement(&g_srv.generations);
}

static void handle_status(SOCKET c) {
    char b[1200];
    if (!g_srv.model) {
        snprintf(b, sizeof(b), "{\"ok\":true,\"model_loaded\":false,\"busy\":%s}\n",
                 g_srv.busy ? "true" : "false");
        send_json(c, 200, b);
        return;
    }
    katali_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    katali_sdk_model_info(g_srv.model, &info);
    katali_sdk_stats st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    if (g_srv.session) katali_sdk_session_stats(g_srv.session, &st);
    double last_total = 0.0;
    int last_tokens = 0;
    EnterCriticalSection(&g_srv.id_lock);
    last_total = g_srv.last_total_s;
    last_tokens = g_srv.last_tokens;
    LeaveCriticalSection(&g_srv.id_lock);
    snprintf(b, sizeof(b),
             "{\"ok\":true,\"model_loaded\":true,\"busy\":%s,\"arch\":\"%s\","
             "\"layers\":%d,\"vocab\":%d,\"ctx\":%d,\"load_s\":%.3f,"
             "\"model_loads\":%d,\"generations\":%d,\"last_total_s\":%.3f,"
             "\"last_tokens\":%d,\"uptime_s\":%.1f,"
             "\"last_prefill_s\":%.3f,\"last_ttft_s\":%.3f,\"last_decode_tps\":%.2f,"
             "\"resident_bytes\":%llu}\n",
             g_srv.busy ? "true" : "false", info.arch, info.n_layers, info.vocab,
             info.ctx_len, st.load_s, (int)g_srv.model_loads, (int)g_srv.generations,
             last_total, last_tokens,
             g_srv.uptime_start_s > 0.0 ? now_wall() - g_srv.uptime_start_s : 0.0,
             st.last_prefill_s, st.last_ttft_s,
             st.last_decode_tps, (unsigned long long)info.resident_bytes);
    send_json(c, 200, b);
}

static void handle_reset(SOCKET c) {
    if (!g_srv.session) { send_error(c, 409, "no_model", "load a model first"); return; }
    int32_t rc = katali_sdk_session_reset(g_srv.session);
    if (rc != KATALI_SDK_OK) {
        send_error(c, 409, rc == KATALI_SDK_ERR_BUSY ? "busy" : "reset_failed",
                   katali_sdk_last_error());
        return;
    }
    send_json(c, 200, "{\"ok\":true,\"type\":\"reset\"}\n");
}

static void handle_cancel(SOCKET c, const char *body) {
    if (!g_srv.session) { send_error(c, 409, "no_model", "load a model first"); return; }
    char id[64];
    int have_id = json_get_str(body, "id", id, sizeof(id)) && id[0];
    int matches = 1;
    if (have_id) {
        EnterCriticalSection(&g_srv.id_lock);
        matches = strcmp(g_srv.current_id, id) == 0;
        LeaveCriticalSection(&g_srv.id_lock);
    }
    if (!matches) {
        char b[256];
        char esc[64];
        json_escape(id, esc, sizeof(esc));
        snprintf(b, sizeof(b),
                 "{\"ok\":false,\"error\":{\"code\":\"unknown_id\",\"message\":\"no generation with id %s\"}}\n",
                 esc);
        send_json(c, 404, b);
        return;
    }
    InterlockedExchange(&g_srv.cancel_request, 1);
    EnterCriticalSection(&g_srv.id_lock);
    katali_session *active = g_srv.active_session;
    LeaveCriticalSection(&g_srv.id_lock);
    if (active) katali_sdk_cancel(active);
    send_json(c, 200, "{\"ok\":true,\"type\":\"cancel\"}\n");
}


/* Generation runs on a worker thread (see GenJob): the accept loop keeps serving,
 * so /cancel can interrupt a stream and a client disconnect frees the worker. */
static void handle_generate(SOCKET c, const char *body);

static DWORD WINAPI generate_thread(LPVOID p) {
    GenJob *j = (GenJob *)p;
    handle_generate(j->c, j->body);
    closesocket(j->c);
    EnterCriticalSection(&g_srv.id_lock);
    g_srv.active_session = NULL;
    g_srv.current_id[0] = '\0';
    LeaveCriticalSection(&g_srv.id_lock);
    InterlockedExchange(&g_srv.busy, 0);
    free(j);
    return 0;
}

/* Validate, claim the single generation slot, and hand the socket to a worker.
 * Returns 1 when the caller still owns the socket (error answered inline). */
static int start_generate(SOCKET c, const char *body) {
    if (!g_srv.model || !g_srv.session) {
        send_error(c, 409, "no_model", "load a model first");
        return 1;
    }
    char prompt[64];
    if (!json_get_str(body, "prompt", prompt, sizeof(prompt))) {
        send_error(c, 400, "invalid_request", "generate needs a \"prompt\"");
        return 1;
    }
    if (InterlockedCompareExchange(&g_srv.busy, 1, 0) != 0) {
        send_error(c, 409, "busy", "another generation is running");
        return 1;
    }
    GenJob *j = (GenJob *)calloc(1, sizeof(GenJob));
    if (!j) {
        InterlockedExchange(&g_srv.busy, 0);
        send_error(c, 500, "internal", "out of memory for the request");
        return 1;
    }
    j->c = c;
    snprintf(j->body, sizeof(j->body), "%s", body);
    if (!json_get_str(body, "id", j->id, sizeof(j->id)) || !j->id[0])
        snprintf(j->id, sizeof(j->id), "g1");
    EnterCriticalSection(&g_srv.id_lock);
    snprintf(g_srv.current_id, sizeof(g_srv.current_id), "%s", j->id);
    LeaveCriticalSection(&g_srv.id_lock);
    InterlockedExchange(&g_srv.cancel_request, 0);

    HANDLE th = CreateThread(NULL, 0, generate_thread, j, 0, NULL);
    if (!th) {
        free(j);
        InterlockedExchange(&g_srv.busy, 0);
        send_error(c, 500, "internal", "could not start the generation thread");
        return 1;
    }
    CloseHandle(th);
    return 0;
}

static void handle_generate(SOCKET c, const char *body) {
    char prompt[8192];
    if (!json_get_str(body, "prompt", prompt, sizeof(prompt))) {
        send_error(c, 400, "invalid_request", "generate needs a \"prompt\"");
        return;
    }
    char id[64];
    if (!json_get_str(body, "id", id, sizeof(id)) || !id[0]) snprintf(id, sizeof(id), "g1");
    int stream = json_get_bool(body, "stream", 0);

    katali_gen_options go;
    memset(&go, 0, sizeof(go));
    go.struct_size = sizeof(go);
    go.max_tokens = json_get_int(body, "max_tokens", 0);
    go.temperature_milli = json_get_int(body, "temperature_milli", 0);
    go.top_k = json_get_int(body, "top_k", 0);
    go.top_p_milli = json_get_int(body, "top_p_milli", 0);
    go.seed = json_get_int(body, "seed", 0);
    go.think = json_get_int(body, "think", 0);

    /* A session per request: independent KV state, the model stays loaded. */
    katali_session *s = katali_sdk_session_create(g_srv.model, &go);
    if (!s) {
        send_error(c, 500, "session_failed", katali_sdk_last_error());
        return;
    }
    EnterCriticalSection(&g_srv.id_lock);
    g_srv.active_session = s;      /* so POST /cancel can reach this generation */
    LeaveCriticalSection(&g_srv.id_lock);

    if (stream) {
        /* NDJSON: one object per line, always terminated by done/cancelled/error. */
        double t0 = now_wall();
        send_stream_headers(c, "application/x-ndjson");
        StreamWr w;
        memset(&w, 0, sizeof(w));
        w.c = c;
        w.client_ok = 1;
        snprintf(w.id, sizeof(w.id), "%s", id);
        int32_t rc = katali_sdk_generate_stream(s, prompt, stream_cb, &w);
        double total = now_wall() - t0;
        server_note_generation(total, (int)(rc > 0 ? rc : 0));
        char line[320];
        int n;
        if (rc == KATALI_SDK_ERR_CANCELLED) {
            n = snprintf(line, sizeof(line),
                         "{\"id\":\"%s\",\"type\":\"cancelled\",\"chunks\":%d,\"total_s\":%.3f}\n",
                         id, w.chunks, total);
        } else if (rc < 0) {
            n = snprintf(line, sizeof(line),
                         "{\"id\":\"%s\",\"type\":\"error\",\"message\":\"generation failed\",\"total_s\":%.3f}\n",
                         id, total);
        } else {
            n = snprintf(line, sizeof(line),
                         "{\"id\":\"%s\",\"type\":\"done\",\"tokens\":%d,\"chunks\":%d,\"total_s\":%.3f}\n",
                         id, (int)rc, w.chunks, total);
        }
        send_all(c, line, (size_t)n);
    } else {
        double t0 = now_wall();
        size_t used = 0;
        char *ans = katali_sdk_generate_alloc(s, prompt, &used);
        double total = now_wall() - t0;
        if (!ans) {
            send_error(c, 500, "generate_failed", katali_sdk_last_error());
        } else {
            server_note_generation(total, 0);
            size_t need = used * 6 + 512;
            char *esc = (char *)malloc(need);
            if (esc) {
                json_escape(ans, esc, need);
                katali_sdk_stats st;
                memset(&st, 0, sizeof(st));
                st.struct_size = sizeof(st);
                katali_sdk_session_stats(s, &st);
                size_t rn = strlen(esc) + 1024;
                char *resp = (char *)malloc(rn);
                if (resp) {
                    snprintf(resp, rn,
                             "{\"ok\":true,\"id\":\"%s\",\"tokens\":%d,\"total_s\":%.3f,\"text\":\"%s\","
                             "\"stats\":{\"prefill_s\":%.3f,\"ttft_s\":%.3f,"
                             "\"decode_s\":%.3f,\"decode_tps\":%.2f,\"prompt_tokens\":%d,"
                             "\"batched_prefill\":%s,\"resident_bytes\":%llu}}\n",
                             id, st.last_generated_tokens, total, esc, st.last_prefill_s,
                             st.last_ttft_s, st.last_decode_s, st.last_decode_tps,
                             st.last_prompt_tokens,
                             st.last_batched_prefill ? "true" : "false",
                             (unsigned long long)st.resident_bytes);
                    send_json(c, 200, resp);
                    free(resp);
                } else {
                    send_error(c, 500, "internal", "out of memory building the response");
                }
                free(esc);
            } else {
                send_error(c, 500, "internal", "out of memory escaping the answer");
            }
            katali_sdk_string_free(ans);
        }
    }

    katali_sdk_session_destroy(s);
}


/* --- server lifecycle --------------------------------------------------- */

/* Returns 1 when the caller still owns the socket, 0 when a worker owns it. */
static int dispatch(SOCKET c, const char *method, const char *path, const char *body) {
    if (!strcmp(method, "GET") && !strcmp(path, "/health"))    { handle_health(c); return 1; }
    if (!strcmp(method, "GET") && !strcmp(path, "/status"))    { handle_status(c); return 1; }
    if (!strcmp(method, "POST") && !strcmp(path, "/load"))     { handle_load(c, body); return 1; }
    if (!strcmp(method, "POST") && !strcmp(path, "/generate")) return start_generate(c, body);
    if (!strcmp(method, "POST") && !strcmp(path, "/reset"))    { handle_reset(c); return 1; }
    if (!strcmp(method, "POST") && !strcmp(path, "/cancel"))   { handle_cancel(c, body); return 1; }
    if (!strcmp(method, "POST") && !strcmp(path, "/shutdown")) {
        send_json(c, 200, "{\"ok\":true,\"type\":\"shutdown\"}\n");
        InterlockedExchange(&g_srv.shutdown, 1);
        return 1;
    }
    send_error(c, 404, "not_found", "unknown endpoint; see the SDK/API reference");
    return 1;
}

/* Startup load shares the model_options path with POST /load so the two cannot
 * diverge in behaviour. */
static void preload(const char *model_path, int threads) {
    if (!model_path || !model_path[0]) return;
    katali_model_options mo;
    memset(&mo, 0, sizeof(mo));
    mo.struct_size = sizeof(mo);
    mo.model_path = model_path;
    mo.n_threads = threads;
    g_srv.model = katali_sdk_model_open(&mo);
    if (!g_srv.model) {
        printf("[server] model open failed: %s\n", katali_sdk_last_error());
        fflush(stdout);
        return;
    }
    katali_gen_options go;
    memset(&go, 0, sizeof(go));
    go.struct_size = sizeof(go);
    go.max_tokens = 256;
    g_srv.session = katali_sdk_session_create(g_srv.model, &go);
    snprintf(g_srv.model_path, sizeof(g_srv.model_path), "%s", model_path);
    printf("[server] model loaded: %s\n", model_path);
    fflush(stdout);
}

int katali_gguf_serve(const char *model_path, int port, int threads, char *err, size_t err_cap) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        snprintf(err, err_cap, "WSAStartup failed");
        return -1;
    }
    memset(&g_srv, 0, sizeof(g_srv));
    g_srv.uptime_start_s = now_wall();
    g_srv.model_loads = 1;   /* the model is opened once, at startup or on /load */
    InitializeCriticalSection(&g_srv.id_lock);

    SOCKET ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls == INVALID_SOCKET) {
        snprintf(err, err_cap, "socket() failed: %d", WSAGetLastError());
        DeleteCriticalSection(&g_srv.id_lock);
        WSACleanup();
        return -1;
    }
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   /* 127.0.0.1 only */
    if (bind(ls, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        snprintf(err, err_cap, "bind 127.0.0.1:%d failed: %d", port, WSAGetLastError());
        closesocket(ls);
        DeleteCriticalSection(&g_srv.id_lock);
        WSACleanup();
        return -1;
    }
    if (listen(ls, 8) != 0) {
        snprintf(err, err_cap, "listen failed: %d", WSAGetLastError());
        closesocket(ls);
        DeleteCriticalSection(&g_srv.id_lock);
        WSACleanup();
        return -1;
    }
    printf("[server] listening on http://127.0.0.1:%d (loopback only)\n", port);
    printf("[server] GET /health /status; POST /load /generate /reset /cancel /shutdown\n");
    fflush(stdout);

    preload(model_path, threads);

    while (!g_srv.shutdown) {
        SOCKET cl = accept(ls, NULL, NULL);
        if (cl == INVALID_SOCKET) continue;

        static char buf[SRV_MAX_BODY + 8192];
        size_t len = 0;
        if (read_request(cl, buf, sizeof(buf), &len) != 0) {
            send_error(cl, 400, "bad_request", "malformed or oversized request");
            closesocket(cl);
            continue;
        }
        char method[16] = "", path[128] = "";
        if (sscanf(buf, "%15s %127s", method, path) != 2) {
            send_error(cl, 400, "bad_request", "malformed request line");
            closesocket(cl);
            continue;
        }
        char *body = strstr(buf, "\r\n\r\n");
        body = body ? body + 4 : (char *)"";
        if (dispatch(cl, method, path, body)) closesocket(cl);
    }

    /* If a generation is still running, cancel it and let the worker unwind
     * before tearing the model down. */
    if (InterlockedCompareExchange(&g_srv.busy, 0, 0) != 0) {
        InterlockedExchange(&g_srv.cancel_request, 1);
        EnterCriticalSection(&g_srv.id_lock);
        katali_session *active = g_srv.active_session;
        LeaveCriticalSection(&g_srv.id_lock);
        if (active) katali_sdk_cancel(active);
        for (int i = 0; i < 100 && InterlockedCompareExchange(&g_srv.busy, 0, 0) != 0; i++)
            Sleep(50);
    }

    if (g_srv.session) katali_sdk_session_destroy(g_srv.session);
    if (g_srv.model)   katali_sdk_model_close(g_srv.model);
    DeleteCriticalSection(&g_srv.id_lock);
    closesocket(ls);
    WSACleanup();
    printf("[server] stopped\n");
    fflush(stdout);
    return 0;
}


