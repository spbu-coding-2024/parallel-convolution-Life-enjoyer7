#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tests_utils.h"

void testIdentityFilter(void)
{
    printf("\n");
    printf("                                        TEST 1: IDENTITY FILTER                         \n");

    Filter identity = filter_identity();
    double times[15] = {0};
    int valid_count = 0;

    for (int j = 0; j < 15; j++)
    {
        IplImage *img = cvLoadImage(imagePaths[j], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %s\n", imagePaths[j]);
            continue;
        }

        IplImage *result = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        times[valid_count] = timeFilterMs(img, result, &identity);

        // Проверка: identity фильтр не должен изменять изображение
        int eq = imagesEqual(img, result);
        if (!eq)
        {
            printf("Results do not match original for image %d!\n", j);
        }
        assert(eq);

        cvReleaseImage(&result);
        cvReleaseImage(&img);
        valid_count++;
    }

    printf("\n                                        PERFORMANCE RESULTS\n\n");
    double total = 0;
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[i], imageNames[i]);
        total += times[i];
    }
    printf("\n  Total: %8.2f ms\n", total);
    printf("\n                                        TEST 1 PASSED\n");
    filter_free(&identity);
}

void testShiftComposition(void)
{
    printf("\n");
    printf("                                        TEST 2: SHIFT COMPOSITION                         \n");

    double times[3][15] = {0}; // 3 композиции: RightLeft, UpDown, Diag
    int valid_count = 0;

    double kernel_right[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}};
    double kernel_left[3][3] = {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    double kernel_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 1, 0}};
    double kernel_down[3][3] = {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
    double kernel_diag_up[3][3] = {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}};
    double kernel_diag_down[3][3] = {{0, 0, 1}, {0, 0, 0}, {0, 0, 0}};

    Filter shiftRight = filter_create(3, 3, &kernel_right[0][0], 1.0, 0.0);
    Filter shiftLeft = filter_create(3, 3, &kernel_left[0][0], 1.0, 0.0);
    Filter shiftUp = filter_create(3, 3, &kernel_up[0][0], 1.0, 0.0);
    Filter shiftDown = filter_create(3, 3, &kernel_down[0][0], 1.0, 0.0);
    Filter shiftDiagUp = filter_create(3, 3, &kernel_diag_up[0][0], 1.0, 0.0);
    Filter shiftDiagDown = filter_create(3, 3, &kernel_diag_down[0][0], 1.0, 0.0);

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        // Right-Left
        IplImage *temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        times[0][valid_count] = timeTwoFiltersMs(img, temp, &shiftRight, temp, final, &shiftLeft);
        int eq = imagesEqual(img, final);
        if (!eq)
            printf("Right-Left composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        // Up-Down
        temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        times[1][valid_count] = timeTwoFiltersMs(img, temp, &shiftUp, temp, final, &shiftDown);
        eq = imagesEqual(img, final);
        if (!eq)
            printf("Up-Down composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        // Diag
        temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        final = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        times[2][valid_count] = timeTwoFiltersMs(img, temp, &shiftDiagUp, temp, final, &shiftDiagDown);
        eq = imagesEqual(img, final);
        if (!eq)
            printf("Diag composition failed for image %d\n", i);
        assert(eq);
        cvReleaseImage(&temp);
        cvReleaseImage(&final);

        cvReleaseImage(&img);
        valid_count++;
    }

    filter_free(&shiftRight);
    filter_free(&shiftLeft);
    filter_free(&shiftUp);
    filter_free(&shiftDown);
    filter_free(&shiftDiagUp);
    filter_free(&shiftDiagDown);

    printf("\n                                        SHIFT COMPOSITION RESULTS\n\n");
    printf("LEFT-RIGHT:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[0][i], imageNames[i]);
    }
    printf("\nUP-DOWN:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[1][i], imageNames[i]);
    }
    printf("\nDIAG:\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[2][i], imageNames[i]);
    }

    printf("\n                                        TEST 2 PASSED\n");
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                                        TEST 3: ZERO PADDING                                         \n");

    double times[5][15] = {0}; // 5 фильтров
    int valid_count = 0;

    Filter original_filters[5];
    original_filters[0] = filter_blur3x3();
    original_filters[1] = filter_gaussian3x3();
    original_filters[2] = filter_findedges1();
    original_filters[3] = filter_sharpen1();
    original_filters[4] = filter_emboss1();

    double kernel_blur_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0.2, 0, 0},
        {0, 0.2, 0.2, 0.2, 0},
        {0, 0, 0.2, 0, 0},
        {0, 0, 0, 0, 0}};

    double kernel_gauss_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 1, 2, 1, 0},
        {0, 2, 4, 2, 0},
        {0, 1, 2, 1, 0},
        {0, 0, 0, 0, 0}};

    double kernel_edges_padded[7][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, 2, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0}};

    double kernel_sharpen_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, -1, -1, -1, 0},
        {0, -1, 9, -1, 0},
        {0, -1, -1, -1, 0},
        {0, 0, 0, 0, 0}};

    double kernel_emboss_padded[5][5] = {
        {0, 0, 0, 0, 0},
        {0, -1, -1, 0, 0},
        {0, -1, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0}};

    Filter padded_filters[5];
    padded_filters[0] = filter_create(5, 5, &kernel_blur_padded[0][0], 1.0, 0.0);
    padded_filters[1] = filter_create(5, 5, &kernel_gauss_padded[0][0], 1.0 / 16.0, 0.0);
    padded_filters[2] = filter_create(7, 7, &kernel_edges_padded[0][0], 1.0, 0.0);
    padded_filters[3] = filter_create(5, 5, &kernel_sharpen_padded[0][0], 1.0, 0.0);
    padded_filters[4] = filter_create(5, 5, &kernel_emboss_padded[0][0], 1.0, 128.0);

    const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1", "sharpen1", "emboss1"};

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        for (int j = 0; j < 5; j++)
        {
            IplImage *result_orig = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            IplImage *result_padded = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);

            times[j][valid_count] = timeTwoFiltersMs(img, result_orig, &original_filters[j],
                                                     img, result_padded, &padded_filters[j]);

            int eq = imagesEqual(result_orig, result_padded);
            if (!eq)
            {
                printf("ERROR: Filter %s on image %d: results do not match!\n", filter_names[j], i);
            }
            assert(eq);

            cvReleaseImage(&result_orig);
            cvReleaseImage(&result_padded);
        }

        cvReleaseImage(&img);
        valid_count++;
    }

    for (int j = 0; j < 5; j++)
    {
        filter_free(&padded_filters[j]);
        filter_free(&original_filters[j]);
    }

    printf("\n                                        ZERO PADDING RESULTS\n\n");
    for (int j = 0; j < 5; j++)
    {
        printf("%s:\n", filter_names[j]);
        for (int i = 0; i < valid_count; i++)
        {
            printf("  %8.2f ms  for  %s\n", times[j][i], imageNames[i]);
        }
        printf("\n");
    }

    printf("\n                                        TEST 3 PASSED\n");
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                                        TEST 4: ZERO FILTER                                         \n");

    double times[15] = {0};
    int valid_count = 0;

    double kernel_zero[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Filter zero_filter = filter_create(3, 3, &kernel_zero[0][0], 1.0, 0.0);

    for (int i = 0; i < 15; i++)
    {
        IplImage *img = cvLoadImage(imagePaths[i], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %d\n", i);
            continue;
        }

        IplImage *result = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        times[valid_count] = timeFilterMs(img, result, &zero_filter);

        // Проверка: все пиксели должны быть чёрными (0,0,0)
        int step = result->widthStep;
        int channels = result->nChannels;
        const unsigned char *data = (const unsigned char *)result->imageData;
        int all_black = 1;
        for (int y = 0; y < result->height && all_black; y++)
        {
            for (int x = 0; x < result->width; x++)
            {
                const unsigned char *pixel = data + y * step + x * channels;
                if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                {
                    all_black = 0;
                    break;
                }
            }
        }
        if (!all_black)
        {
            printf("ERROR: Zero filter on image %d: result is not all black!\n", i);
        }
        assert(all_black);

        cvReleaseImage(&result);
        cvReleaseImage(&img);
        valid_count++;
    }

    filter_free(&zero_filter);

    printf("\n                                        ZERO FILTER RESULTS\n\n");
    for (int i = 0; i < valid_count; i++)
    {
        printf("  %8.2f ms  for  %s\n", times[i], imageNames[i]);
    }

    printf("\n                                        TEST 4 PASSED\n");
}

// TEST 5: property-based тесты на случайных данных

// В отличие от тестов 1-4 (фиксированные 15 реальных фотографий), здесь и
// изображения, и фильтры генерируются случайно, с широким разбросом размеров,
// включая крайние случаи (1x1 картинка, фильтр 1x1, фильтр крупнее картинки).

void testRandomizedProperties(void)
{
    printf("\n");
    printf("                                        TEST 5: RANDOMIZED PROPERTIES (random image/filter sizes)\n");

    unsigned int seed = (unsigned int)time(NULL);
    printf("  random seed: %u \n", seed);
    srand(seed);

    const int TRIALS = 30;
    // Допуск для теста композиции: между двумя последовательными applyFilter
    // результат первого прохода округляется double -> uint8, а однопроходный
    // composed-фильтр этого промежуточного округления не делает. Разница
    // ограничена (сумма весов A и B по отдельности <= 1), эмпирически не
    // превышает 1-2 градации яркости.
    const int COMPOSITION_TOLERANCE = 2;

    Filter identity = filter_identity();
    int max_diff_seen = 0;

    for (int t = 0; t < TRIALS; t++)
    {
        int w = randomImageDim();
        int h = randomImageDim();
        IplImage *img = createRandomImage(w, h);

        // (iv) identity не меняет изображение — при случайном размере картинки
        IplImage *idOut = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double tId = timeFilterMs(img, idOut, &identity);
        printf("  trial %2d | img %2dx%-2d | %-9s %2dx%-2d          | %9.4f ms\n", t, w, h, "identity", 3, 3, tId);
        assert(imagesEqual(img, idOut));
        cvReleaseImage(&idOut);

        // (iv) нулевой фильтр случайного нечётного размера -> чёрное изображение
        int zw = randomOddSize(w);
        int zh = randomOddSize(h);
        double *zeroData = (double *)calloc(zw * zh, sizeof(double));
        Filter zeroFilter = filter_create(zw, zh, zeroData, 1.0, 0.0);
        free(zeroData);

        IplImage *zeroOut = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double tZero = timeFilterMs(img, zeroOut, &zeroFilter);
        printf("  trial %2d | img %2dx%-2d | %-9s %2dx%-2d          | %9.4f ms\n", t, w, h, "zero", zw, zh, tZero);
        {
            int step = zeroOut->widthStep, channels = zeroOut->nChannels;
            const unsigned char *zdata = (const unsigned char *)zeroOut->imageData;
            int all_black = 1;
            for (int y = 0; y < h && all_black; y++)
                for (int x = 0; x < w; x++)
                {
                    const unsigned char *p = zdata + y * step + x * channels;
                    if (p[0] || p[1] || p[2])
                    {
                        all_black = 0;
                        break;
                    }
                }
            assert(all_black);
        }
        cvReleaseImage(&zeroOut);
        filter_free(&zeroFilter);

        // (iii) расширение случайного фильтра кольцом нулей не меняет результат
        int pw = randomOddSize(w);
        int ph = randomOddSize(h);
        double *baseData = (double *)malloc(pw * ph * sizeof(double));
        for (int i = 0; i < pw * ph; i++)
            baseData[i] = (double)(rand() % 5) - 2; // случайные веса в [-2, 2]
        Filter base = filter_create(pw, ph, baseData, 1.0, 0.0);
        free(baseData);

        int padW = pw + 2, padH = ph + 2;
        double *paddedData = (double *)calloc(padW * padH, sizeof(double));
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++)
                paddedData[(y + 1) * padW + (x + 1)] = base.matrix[y][x];
        Filter padded = filter_create(padW, padH, paddedData, 1.0, 0.0);
        free(paddedData);

        IplImage *baseOut = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *paddedOut = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double tPad = timeTwoFiltersMs(img, baseOut, &base, img, paddedOut, &padded);
        printf("  trial %2d | img %2dx%-2d | %-9s %2dx%-2d -> %2dx%-2d | %9.4f ms\n",
               t, w, h, "pad", pw, ph, padW, padH, tPad);
        assert(imagesEqual(baseOut, paddedOut));
        cvReleaseImage(&baseOut);
        cvReleaseImage(&paddedOut);
        filter_free(&base);
        filter_free(&padded);

        // (i) применение двух случайных фильтров подряд == один проход с их композицией..
        int aw = randomOddSize(w), ah = randomOddSize(h);
        int bw = randomOddSize(w), bh = randomOddSize(h);
        Filter fa = randomConvexFilter(aw, ah);
        Filter fb = randomConvexFilter(bw, bh);
        Filter composed = composeKernels(&fa, &fb);

        IplImage *mid = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *sequential = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        IplImage *oneShot = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        double tSeq = timeTwoFiltersMs(img, mid, &fa, mid, sequential, &fb);
        printf("  trial %2d | img %2dx%-2d | %-9s A: %2dx%-2d + B: %2dx%-2d | %9.4f ms\n",
               t, w, h, "seq", aw, ah, bw, bh, tSeq);

        double tComposed = timeFilterMs(img, oneShot, &composed);
        printf("  trial %2d | img %2dx%-2d | %-9s %2dx%-2d          | %9.4f ms\n",
               t, w, h, "composed", composed.width, composed.height, tComposed);

        int diff = 0;
        int ok = imagesApproxEqual(sequential, oneShot, COMPOSITION_TOLERANCE, &diff);
        if (diff > max_diff_seen)
            max_diff_seen = diff;
        if (!ok)
        {
            printf("  MISMATCH: trial %d, img %dx%d, A %dx%d, B %dx%d, max diff %d\n",
                   t, w, h, aw, ah, bw, bh, diff);
            fflush(stdout);
        }
        assert(ok);

        cvReleaseImage(&mid);
        cvReleaseImage(&sequential);
        cvReleaseImage(&oneShot);
        filter_free(&fa);
        filter_free(&fb);
        filter_free(&composed);

        cvReleaseImage(&img);
    }

    filter_free(&identity);

    printf("  max observed diff between sequential and composed-kernel pass: %d (tolerance %d)\n",
           max_diff_seen, COMPOSITION_TOLERANCE);
    printf("\n                                        TEST 5 PASSED (%d random trials)\n", TRIALS);
}

// TEST 6: сверка с эталонной библиотекой (OpenCV cv::filter2D через
// referenceApplyFilter из tests_utils)

void testReferenceLibrary(void)
{
    printf("\n");
    printf("                                        TEST 6: REFERENCE LIBRARY (OpenCV filter2D)\n");

    // Пара маленьких + пара больших картинок - чтобы не гонять cv::filter2D
    const int imageIdx[] = {6, 3, 9, 5, 8};
    const int numImages = 5;

    Filter filters[15];
    const char *filterNames[15] = {
        "blur3x3", "blur5x5", "gaussian3x3", "gaussian5x5", "motionblur",
        "findedges1", "findedges2", "findedges3", "findedges4",
        "sharpen1", "sharpen2", "sharpen3", "emboss1", "emboss2", "identity"};
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

    const int TOLERANCE = 1; // запас на порядок суммирования double, см. referenceApplyFilter
    int max_diff_seen = 0;
    int mismatches = 0;

    for (int i = 0; i < numImages; i++)
    {
        int idx = imageIdx[i];
        IplImage *img = cvLoadImage(imagePaths[idx], 1);
        if (!img)
        {
            printf("ERROR: Failed to load image %s\n", imagePaths[idx]);
            continue;
        }

        for (int j = 0; j < 15; j++)
        {
            IplImage *ours = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
            double t = timeFilterMs(img, ours, &filters[j]);

            IplImage *reference = referenceApplyFilter(img, &filters[j]);

            int diff = 0;
            int ok = imagesApproxEqual(ours, reference, TOLERANCE, &diff);
            if (diff > max_diff_seen)
                max_diff_seen = diff;
            if (!ok)
            {
                mismatches++;
                printf("  MISMATCH: %s / %s, max diff %d\n", imageNames[idx], filterNames[j], diff);
            }
            assert(ok);

            printf("  %-22s | %-12s %2dx%-2d | img %4dx%-4d | %8.4f ms\n",
                   imageNames[idx], filterNames[j], filters[j].width, filters[j].height,
                   img->width, img->height, t);

            cvReleaseImage(&ours);
            cvReleaseImage(&reference);
        }

        cvReleaseImage(&img);
        printf("  %s vs cv::filter2D: OK (all %d filters)\n\n", imageNames[idx], 15);
    }

    for (int j = 0; j < 15; j++)
        filter_free(&filters[j]);

    printf("  max observed diff vs OpenCV: %d (tolerance %d), mismatches: %d\n",
           max_diff_seen, TOLERANCE, mismatches);
    printf("\n                                        TEST 6 PASSED\n");
}

int main(void)
{
    testIdentityFilter();
    testShiftComposition();
    testZeroPadding();
    testZeroFilter();
    testRandomizedProperties();
    testReferenceLibrary();
    return 0;
}
