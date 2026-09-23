/* Katali-GGUF local server — Apache-2.0
 *
 * Loopback-only HTTP endpoint, implemented over the public SDK so the server,
 * the DLL and the CLI share one generation path. */
#ifndef KATALI_GGUF_SERVER_H
#define KATALI_GGUF_SERVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Serve until POST /shutdown. `model_path` may be NULL to start empty and use
 * POST /load. Binds 127.0.0.1:port. Returns 0 on a clean shutdown, -1 with a
 * reason in `err`. */
int katali_gguf_serve(const char *model_path, int port, int threads, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_SERVER_H */

