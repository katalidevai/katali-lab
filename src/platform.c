#define _POSIX_C_SOURCE 200809L
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif

int katali_map_file(const char *path, KataliMap *out) {
    memset(out, 0, sizeof(*out));
#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hf == INVALID_HANDLE_VALUE) return KATALI_ERR_IO;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return KATALI_ERR_IO; }
    if (sz.QuadPart <= 0) { CloseHandle(hf); return KATALI_ERR_IO; }
    HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); return KATALI_ERR_IO; }
    void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hm); CloseHandle(hf); return KATALI_ERR_IO; }
    out->data = (const uint8_t *)p;
    out->size = (uint64_t)sz.QuadPart;
    out->os_file = (void *)hf;
    out->os_map = (void *)hm;
    out->mapped = 1;
    return KATALI_OK;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return KATALI_ERR_IO;
    struct stat st;
    if (fstat(fd, &st) || st.st_size <= 0) { close(fd); return KATALI_ERR_IO; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); return KATALI_ERR_IO; }
    out->data = (const uint8_t *)p;
    out->size = (uint64_t)st.st_size;
    out->os_file = (void *)(intptr_t)fd;
    out->mapped = 1;
    return KATALI_OK;
#endif
}

void katali_unmap(KataliMap *m) {
    if (!m) return;
#ifdef _WIN32
    if (m->data && m->mapped) UnmapViewOfFile((void *)m->data);
    if (m->os_map) CloseHandle((HANDLE)m->os_map);
    if (m->os_file) CloseHandle((HANDLE)m->os_file);
#else
    if (m->data && m->mapped) munmap((void *)m->data, (size_t)m->size);
    if (m->os_file) close((int)(intptr_t)m->os_file);
#endif
    memset(m, 0, sizeof(*m));
}

double katali_time_s(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init;
    LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

uint64_t katali_ram_avail_bytes(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return (uint64_t)ms.ullAvailPhys;
#else
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psz = sysconf(_SC_PAGESIZE);
    if (pages < 0 || psz < 0) return 0;
    return (uint64_t)pages * (uint64_t)psz;
#endif
}
