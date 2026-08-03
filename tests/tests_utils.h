#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H

#include "../src/filter.h"

// Сигнатура и последовательной applyFilter, и любой из параллельных стратегий.
typedef void (*FilterFn)(const IplImage *, IplImage *, const Filter *);

// Общий набор тестовых изображений (используется тестами 1-4 и 6).
extern const char *imagePaths[];
extern const char *imageNames[];

// Все 6 параллельных стратегий с именами - для итерации в TEST 5.
typedef struct
{
    const char *name;
    FilterFn fn;
} NamedStrategy;
extern const NamedStrategy parallelStrategies[6];

// Последовательная applyFilter + все 6 параллельных стратегий - для property-тестов
// TEST 5, где каждое свойство свёртки проверяется отдельно на каждой реализации.
extern const NamedStrategy allStrategies[7];

// Прогрев + усреднение по нескольким прогонам вместо одного шумного замера.
// fn - последовательная applyFilter либо любая из параллельных стратегий.
double timeFilterMs(FilterFn fn, const IplImage *src, IplImage *dst, const Filter *f);
double timeTwoFiltersMs(FilterFn fn, const IplImage *srcA, IplImage *dstA, const Filter *fA,
                         const IplImage *srcB, IplImage *dstB, const Filter *fB);

int imagesEqual(const IplImage *a, const IplImage *b);
int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance, int *max_diff_out);
int imageIsBlack(const IplImage *img);

// Генераторы случайных данных для property-тестов (TEST 5).
IplImage *createRandomImage(int w, int h);
int randomOddSize(int maxSize);
int randomImageDim(void);
Filter randomConvexFilter(int w, int h);
Filter composeKernels(const Filter *a, const Filter *b);

// Эталонная свёртка через cv::filter2D (для сверки в TEST 6).
IplImage *referenceApplyFilter(const IplImage *src, const Filter *f);

#endif
