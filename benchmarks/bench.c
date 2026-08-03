// Сбор сырых данных производительности в CSV для make benchmark.
// Не проверяет корректность (это делают tests/tests.c) - только время.
#include <stdio.h>
#include <stdlib.h>
#include "../src/filter.h"
#include "../src/main_utils.h"
#include "../tests/tests_utils.h"

typedef struct
{
    const char *name;
    Filter (*factory)(void);
} NamedFilterFactory;

int main(int argc, char *argv[])
{
    int repeat = 5;
    if (argc > 1)
    {
        repeat = atoi(argv[1]);
        if (repeat < 1)
        {
            fprintf(stderr, "repeat must be >= 1\n");
            return 1;
        }
    }

    NamedFilterFactory filterFactories[15] = {
        {"blur3x3", filter_blur3x3},
        {"blur5x5", filter_blur5x5},
        {"gaussian3x3", filter_gaussian3x3},
        {"gaussian5x5", filter_gaussian5x5},
        {"motionblur", filter_motionblur},
        {"findedges1", filter_findedges1},
        {"findedges2", filter_findedges2},
        {"findedges3", filter_findedges3},
        {"findedges4", filter_findedges4},
        {"sharpen1", filter_sharpen1},
        {"sharpen2", filter_sharpen2},
        {"sharpen3", filter_sharpen3},
        {"emboss1", filter_emboss1},
        {"emboss2", filter_emboss2},
        {"identity", filter_identity},
    };

    printf("image,width,height,filter,filter_w,filter_h,strategy,repeat,min_ms,mean_ms,median_ms\n");

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            fprintf(stderr, "Warning: failed to load %s, skipping\n", imagePaths[i]);
            continue;
        }

        for (int j = 0; j < 15; j++)
        {
            Filter f = filterFactories[j].factory();

            for (int s = 0; s < 7; s++)
            {
                IplImage *dst = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

                double min_ms, mean_ms, median_ms;
                benchmark_filter(allStrategies[s].fn, img, dst, &f, repeat, &min_ms, &mean_ms, &median_ms);

                printf("%s,%d,%d,%s,%d,%d,%s,%d,%.4f,%.4f,%.4f\n",
                       imageNames[i], img->width, img->height,
                       filterFactories[j].name, f.width, f.height,
                       allStrategies[s].name, repeat, min_ms, mean_ms, median_ms);
                fflush(stdout);

                cvReleaseImage(&dst);
            }

            filter_free(&f);
        }

        cvReleaseImage(&img);
        fprintf(stderr, "done: %s\n", imageNames[i]);
    }

    return 0;
}
