#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// C++ Mat API - нужен только для referenceApplyFilter (сверка с OpenCV как эталоном).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "tests_utils.h"

const char *imagePaths[15] = {
    "images/bugatti_1200x600.jpg",
    "images/bugatti_1536x2048.jpg",
    "images/bugatti_3275x4096.jpg",
    "images/ferari_320x320.jpg",
    "images/ferrari_2560x1440.jpg",
    "images/ford_1080x1080.jpg",
    "images/lambo_236x236.jpg",
    "images/lambo_1080x1349.jpg",
    "images/maseratti_2048x2048.jpg",
    "images/mustang_736x736.jpg",
    "images/sportcar_474x503.jpg",
    "images/sportcar_736x981.jpg",
    "images/sportcar_3823x4237.jpg",
    "images/bugatti_big_size.jpg",
    "images/car_1200x687.jpeg"};

const char *imageNames[15] = {
    "bugatti_1200x600.jpg",
    "bugatti_1536x2048.jpg",
    "bugatti_3275x4096.jpg",
    "ferari_320x320.jpg",
    "ferrari_2560x1440.jpg",
    "ford_1080x1080.jpg",
    "lambo_236x236.jpg",
    "lambo_1080x1349.jpg",
    "maseratti_2048x2048.jpg",
    "mustang_736x736.jpg",
    "sportcar_474x503.jpg",
    "sportcar_736x981.jpg",
    "sportcar_3823x4237.jpg",
    "bugatti_big_size.jpg",
    "car_1200x687.jpeg"};

const NamedStrategy parallelStrategies[6] = {
    {"pixelwise", applyFilterParallelPixelwise},
    {"rows", applyFilterParallelByRows},
    {"cols", applyFilterParallelByCols},
    {"blocks32", applyFilterParallelByBlocks32},
    {"blocks64", applyFilterParallelByBlocks64},
    {"blocks128", applyFilterParallelByBlocks128},
};

const NamedStrategy allStrategies[7] = {
    {"sequential", applyFilter},
    {"pixelwise", applyFilterParallelPixelwise},
    {"rows", applyFilterParallelByRows},
    {"cols", applyFilterParallelByCols},
    {"blocks32", applyFilterParallelByBlocks32},
    {"blocks64", applyFilterParallelByBlocks64},
    {"blocks128", applyFilterParallelByBlocks128},
};

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#define BENCH_REPEAT 3

// Один непрогретый прогон отбрасывается, дальше усредняем по BENCH_REPEAT прогонам,
// т.к. один-единственный замер слишком шумный для сравнения между картинками/фильтрами/стратегиями.
double timeFilterMs(FilterFn fn, const IplImage *src, IplImage *dst, const Filter *f)
{
    fn(src, dst, f);

    double total = 0.0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        double start = get_time_ms();
        fn(src, dst, f);
        double end = get_time_ms();
        total += (end - start);
    }
    return total / BENCH_REPEAT;
}

// То же самое, но для пары фильтров, применяемых одной и той же стратегией одна за другой
// (композиция или сравнение "оригинал vs padded" - время меряется суммарно на обе).
double timeTwoFiltersMs(FilterFn fn, const IplImage *srcA, IplImage *dstA, const Filter *fA,
                        const IplImage *srcB, IplImage *dstB, const Filter *fB)
{
    fn(srcA, dstA, fA);
    fn(srcB, dstB, fB);

    double total = 0.0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        double start = get_time_ms();
        fn(srcA, dstA, fA);
        fn(srcB, dstB, fB);
        double end = get_time_ms();
        total += (end - start);
    }
    return total / BENCH_REPEAT;
}

int imagesEqual(const IplImage *a, const IplImage *b)
{
    if (a->width != b->width || a->height != b->height || a->nChannels != b->nChannels)
        return 0;

    int step = a->widthStep;
    int channels = a->nChannels;
    const unsigned char *data_a = (const unsigned char *)a->imageData;
    const unsigned char *data_b = (const unsigned char *)b->imageData;

    for (int y = 0; y < a->height; y++)
    {
        for (int x = 0; x < a->width; x++)
        {
            const unsigned char *pa = data_a + y * step + x * channels;
            const unsigned char *pb = data_b + y * step + x * channels;
            if (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2])
                return 0;
        }
    }
    return 1;
}

// Сравнение с допуском: возвращает 1, если максимальное отклонение канала
// не превышает tolerance. max_diff_out (может быть NULL) получает фактический максимум.
int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance, int *max_diff_out)
{
    if (a->width != b->width || a->height != b->height || a->nChannels != b->nChannels)
        return 0;

    int step = a->widthStep;
    int channels = a->nChannels;
    const unsigned char *da = (const unsigned char *)a->imageData;
    const unsigned char *db = (const unsigned char *)b->imageData;
    int max_diff = 0;
    int ok = 1;

    for (int y = 0; y < a->height; y++)
    {
        for (int x = 0; x < a->width; x++)
        {
            const unsigned char *pa = da + y * step + x * channels;
            const unsigned char *pb = db + y * step + x * channels;
            for (int c = 0; c < 3; c++)
            {
                int diff = abs((int)pa[c] - (int)pb[c]);
                if (diff > max_diff)
                    max_diff = diff;
                if (diff > tolerance)
                    ok = 0;
            }
        }
    }

    if (max_diff_out)
        *max_diff_out = max_diff;
    return ok;
}

// Возвращает 1, если все пиксели изображения чёрные (0,0,0).
int imageIsBlack(const IplImage *img)
{
    int step = img->widthStep;
    int channels = img->nChannels;
    const unsigned char *data = (const unsigned char *)img->imageData;

    for (int y = 0; y < img->height; y++)
    {
        for (int x = 0; x < img->width; x++)
        {
            const unsigned char *pixel = data + y * step + x * channels;
            if (pixel[0] || pixel[1] || pixel[2])
                return 0;
        }
    }
    return 1;
}

IplImage *createRandomImage(int w, int h)
{
    IplImage *img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
    int step = img->widthStep;
    int channels = img->nChannels;
    unsigned char *data = (unsigned char *)img->imageData;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            unsigned char *p = data + y * step + x * channels;
            p[0] = (unsigned char)(rand() % 256);
            p[1] = (unsigned char)(rand() % 256);
            p[2] = (unsigned char)(rand() % 256);
        }
    }
    return img;
}

// Случайный нечётный размер от 1 до maxSize включительно

int randomOddSize(int maxSize)
{
    int halfCount = (maxSize + 1) / 2; // сколько нечётных чисел в [1, maxSize]
    return 2 * (rand() % halfCount) + 1;
}

// Небольшая картинка: часть прогонов явно бьёт по крайним случаям (1x1, 2x2, 3x3),
// остальные — случайный размер из достаточно широкого диапазона.
int randomImageDim(void)
{
    int r = rand() % 10;
    if (r == 0)
        return 1;
    if (r == 1)
        return 2;
    if (r == 2)
        return 3;
    return 4 + rand() % 40;
}

// Случайный фильтр-"выпуклая комбинация": неотрицательные веса, сумма == 1,
// factor=1, bias=0. Такой фильтр — взвешенное усреднение соседних пикселей,
// поэтому результат применения гарантированно остаётся в [0,255]
Filter randomConvexFilter(int w, int h)
{
    double *data = (double *)malloc(w * h * sizeof(double));
    double sum = 0.0;
    for (int i = 0; i < w * h; i++)
    {
        data[i] = (double)rand() / RAND_MAX;
        sum += data[i];
    }
    if (sum == 0.0)
    {
        data[0] = 1.0;
        sum = 1.0;
    }
    for (int i = 0; i < w * h; i++)
    {
        data[i] /= sum;
    }

    Filter f = filter_create(w, h, data, 1.0, 0.0);
    free(data);
    return f;
}

// "Объединяет" два фильтра A, B (bias=0) в один фильтр C в вещественной
// арифметике. Это обычная (не циклическая) свёртка ядер A и B: смещения от
// центра у correlate(x - w/2 + fx) складываются линейно, а поскольку A и B
// оба нечётного размера, (wA/2 + wB/2) == (wA+wB-1)/2 и центрирование
// результата совпадает с тем, как applyFilter центрирует фильтр размера
// (wA+wB-1) x (hA+hB-1).
Filter composeKernels(const Filter *a, const Filter *b)
{
    int cw = a->width + b->width - 1;
    int ch = a->height + b->height - 1;
    double *data = (double *)calloc(cw * ch, sizeof(double));

    for (int ay = 0; ay < a->height; ay++)
        for (int ax = 0; ax < a->width; ax++)
            for (int by = 0; by < b->height; by++)
                for (int bx = 0; bx < b->width; bx++)
                    data[(ay + by) * cw + (ax + bx)] += a->matrix[ay][ax] * b->matrix[by][bx];

    Filter c = filter_create(cw, ch, data, 1.0, 0.0);
    free(data);
    return c;
}

// cv::filter2D сам не поддерживает BORDER_WRAP ("BORDER_WRAP is not supported"
// в документации OpenCV), поэтому картинка вручную дополняется по кругу через
// cv::copyMakeBorder(..., BORDER_WRAP) на половину размера фильтра с каждой
// стороны (все фильтры здесь нечётного размера => отступ симметричный), после
// чего вырезается центральная область того же размера, что и исходное изображение.
// Итоговый clamp+truncate делаем сами по double, а не через встроенную конвертацию
// OpenCV в 8U - у неё round(), а у applyFilter/параллельных стратегий - truncate().
IplImage *referenceApplyFilter(const IplImage *src, const Filter *f)
{
    cv::Mat srcMat = cv::cvarrToMat(src);

    cv::Mat kernel(f->height, f->width, CV_64F);
    for (int y = 0; y < f->height; y++)
        for (int x = 0; x < f->width; x++)
            kernel.at<double>(y, x) = f->matrix[y][x] * f->factor;

    int padW = f->width / 2;
    int padH = f->height / 2;

    cv::Mat padded;
    cv::copyMakeBorder(srcMat, padded, padH, padH, padW, padW, cv::BORDER_WRAP);

    cv::Mat filtered64;
    cv::filter2D(padded, filtered64, CV_64F, kernel, cv::Point(-1, -1), f->bias, cv::BORDER_CONSTANT);

    cv::Mat cropped = filtered64(cv::Rect(padW, padH, src->width, src->height));

    IplImage *result = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
    int step = result->widthStep;
    int channels = result->nChannels;
    unsigned char *dst_data = (unsigned char *)result->imageData;

    for (int y = 0; y < src->height; y++)
    {
        for (int x = 0; x < src->width; x++)
        {
            const cv::Vec3d &px = cropped.at<cv::Vec3d>(y, x);
            unsigned char *out = dst_data + y * step + x * channels;
            for (int c = 0; c < 3; c++)
            {
                int v = (int)px[c];
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                out[c] = (unsigned char)v;
            }
        }
    }

    return result;
}
