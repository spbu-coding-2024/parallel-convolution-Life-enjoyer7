#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H

#include "../src/filter.h"

// Общий набор тестовых изображений (используется тестами 1-4 и 6).
extern const char *imagePaths[];
extern const char *imageNames[];

double get_time_ms(void);

// Прогрев + усреднение по нескольким прогонам вместо одного шумного замера.
double timeFilterMs(const IplImage *src, IplImage *dst, const Filter *f);
double timeTwoFiltersMs(const IplImage *srcA, IplImage *dstA, const Filter *fA,
                         const IplImage *srcB, IplImage *dstB, const Filter *fB);

int imagesEqual(const IplImage *a, const IplImage *b);
int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance, int *max_diff_out);

// Генераторы случайных данных для property-тестов (TEST 5).
IplImage *createRandomImage(int w, int h);
int randomOddSize(int maxSize);
int randomImageDim(void);
Filter randomConvexFilter(int w, int h);
Filter composeKernels(const Filter *a, const Filter *b);

// Эталонная свёртка через cv::filter2D (для сверки в TEST 6).
IplImage *referenceApplyFilter(const IplImage *src, const Filter *f);

#endif
