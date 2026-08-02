#ifndef MAIN_UTILS_H
#define MAIN_UTILS_H

#include "filter.h"

double get_time_ms(void);

// Один непрогретый прогон отбрасывается, чтобы не мерить холодный кэш/страницы
// затем `repeat` прогонов дают min/mean/median
void benchmark_filter(const IplImage *src, IplImage *dst, const Filter *f, int repeat,
                      double *out_min, double *out_mean, double *out_median);

#endif
