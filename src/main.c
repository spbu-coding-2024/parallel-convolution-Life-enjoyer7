#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "filter.h"
#include "main_utils.h"

#ifndef CV_LOAD_IMAGE_COLOR
#define CV_LOAD_IMAGE_COLOR 1
#endif

#define NUM_FILTERS 15
#define DEFAULT_REPEAT 10

int main(int argc, char *argv[])
{
    int filterId = -1;
    int strategyId = -1;
    char *load_path = NULL;
    char *save_path = NULL;
    int repeat = DEFAULT_REPEAT;

    Filter filters[NUM_FILTERS];
    filters[0] = filter_blur3x3();
    filters[1] = filter_blur5x5();
    filters[2] = filter_gaussian3x3();
    filters[3] = filter_gaussian5x5();
    filters[4] = filter_motionblur();
    filters[5] = filter_findedges1();
    filters[6] = filter_findedges2();
    filters[7] = filter_findedges3();
    filters[8] = filter_findedges4();
    filters[9] = filter_sharpen1();
    filters[10] = filter_sharpen2();
    filters[11] = filter_sharpen3();
    filters[12] = filter_emboss1();
    filters[13] = filter_emboss2();
    filters[14] = filter_identity();

    // Только параллельные стратегии (без последовательной)
    FilterFn strategies[6] = {
        applyFilterParallelPixelwise,  // 0 - попиксельно
        applyFilterParallelByRows,     // 1 - по строкам
        applyFilterParallelByCols,     // 2 - по столбцам
        applyFilterParallelByBlocks32, // 3 - блоки 32x32
        applyFilterParallelByBlocks64, // 4 - блоки 64x64
        applyFilterParallelByBlocks128 // 5 - блоки 128x128
    };

    static struct option long_options[] = {
        {"filter", required_argument, 0, 'f'},
        {"tactic", required_argument, 0, 't'},
        {"src", required_argument, 0, 's'},
        {"out", required_argument, 0, 'o'},
        {"repeat", required_argument, 0, 'r'},
        {0, 0, 0, 0}};

    int c;
    int option_index = 0;

    while ((c = getopt_long(argc, argv, "f:t:s:o:r:", long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 'f':
            filterId = atoi(optarg);
            break;
        case 't':
            strategyId = atoi(optarg);
            break;
        case 's':
            load_path = optarg;
            break;
        case 'o':
            save_path = optarg;
            break;
        case 'r':
            repeat = atoi(optarg);
            break;
        default:
            printf("Unknown option: %c\n", c);
            return 1;
        }
    }

    if (repeat < 1)
    {
        printf("Error: --repeat must be >= 1\n");
        return 1;
    }

    if (filterId < 0 || filterId >= NUM_FILTERS)
    {
        printf("Error: Invalid filter ID (0-14)\n");
        return 1;
    }

    if (strategyId < 0 || strategyId >= 6)
    {
        printf("Error: Invalid strategy ID (0-5)\n");
        printf("  0 - pixelwise\n");
        printf("  1 - by rows\n");
        printf("  2 - by cols\n");
        printf("  3 - blocks 32x32\n");
        printf("  4 - blocks 64x64\n");
        printf("  5 - blocks 128x128\n");
        return 1;
    }

    if (!load_path)
    {
        printf("Error: Source image required (-s)\n");
        return 1;
    }

    if (!save_path)
    {
        printf("Error: Output path required (-o)\n");
        return 1;
    }

    printf("Loading image from: '%s'\n", load_path);
    IplImage *image = cvLoadImage(load_path, CV_LOAD_IMAGE_COLOR);
    if (!image)
    {
        printf("Error: Failed to load image\n");
        return 1;
    }

    IplImage *result = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 3);

    printf("Applying filter %d with strategy %d (%d timed runs after warm-up)...\n", filterId, strategyId, repeat);

    double min_ms, mean_ms, median_ms;
    benchmark_filter(strategies[strategyId], image, result, &filters[filterId], repeat, &min_ms, &mean_ms, &median_ms);

    printf("Saving to: '%s'\n", save_path);
    cvSaveImage(save_path, result);
    printf("Time spent applying the filter: min=%8.2f ms  mean=%8.2f ms  median=%8.2f ms\n",
           min_ms, mean_ms, median_ms);

    cvReleaseImage(&image);
    cvReleaseImage(&result);
    for (int i = 0; i < NUM_FILTERS; i++)
    {
        filter_free(&filters[i]);
    }

    return 0;
}
