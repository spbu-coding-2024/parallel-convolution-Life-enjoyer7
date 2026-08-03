#ifndef MAIN_UTILS_H
#define MAIN_UTILS_H

#include "filter.h"

// Сигнатура и последовательной applyFilter, и любой из параллельных стратегий.
typedef void (*FilterFn)(const IplImage *, IplImage *, const Filter *);

// Один непрогретый прогон отбрасывается, чтобы не мерить холодный кэш/страницы;
// затем `repeat` прогонов дают min/mean/median вместо шумного одиночного замера.
void benchmark_filter(FilterFn fn, const IplImage *src, IplImage *dst, const Filter *f, int repeat,
                       double *out_min, double *out_mean, double *out_median);

#endif
