#ifndef KATALI_PLATFORM_H
#define KATALI_PLATFORM_H
#include <stdint.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KataliMap {
    const uint8_t *data;
    uint64_t       size;
    void          *os_file;
    void          *os_map;
    int            mapped; /* 1 = OS mmap, 0 = heap */
} KataliMap;

KATALI_API int  katali_map_file(const char *path, KataliMap *out);
KATALI_API void katali_unmap(KataliMap *m);
KATALI_API double katali_time_s(void);
KATALI_API uint64_t katali_ram_avail_bytes(void);

#ifdef __cplusplus
}
#endif
#endif
