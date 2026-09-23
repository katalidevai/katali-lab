/* katali-lab — shared types — Apache-2.0 */
#ifndef KATALI_LAB_H
#define KATALI_LAB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KATALI_OK            0
#define KATALI_ERR          -1
#define KATALI_ERR_IO       -2
#define KATALI_ERR_FORMAT   -3
#define KATALI_ERR_NOMEM    -4
#define KATALI_ERR_SUPPORT  -5

#ifndef KATALI_API
#define KATALI_API
#endif

#ifdef __cplusplus
}
#endif
#endif
