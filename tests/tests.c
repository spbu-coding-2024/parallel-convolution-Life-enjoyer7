#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include "../src/filter.h"
#include "../src/pipeline.h"
#include "../src/utils.h"

const char *strategy_names[] = {
    "Sequential",    // 0
    "Pixelwise",     // 1
    "By Rows",       // 2
    "By Cols",       // 3
    "Blocks 32x32",  // 4
    "Blocks 64x64",  // 5
    "Blocks 128x128" // 6
};

void testIdentityFilter(void)
{
    printf("\n");
    printf("                        TEST 1: IDENTITY FILTER                                 \n");
    printf("                   (Pipeline vs Sequential Comparison)                         \n");

    Filter identity = filter_identity();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // Оригинальные изображения для сравнения
    IplImage **original_images = (IplImage **)malloc(num_images * sizeof(IplImage *));
    for (int j = 0; j < num_images; j++)
    {
        original_images[j] = cvLoadImage(input_paths[j], 1);
        if (!original_images[j])
        {
            printf("ERROR: Failed to load image %s\n", input_paths[j]);
        }
    }

    // Конфигурации потоков
    int thread_counts[] = {1, 2, 4, 8, 12, 16};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Массив результатов: [стратегия][конфигурация_потоков]
    double results[7][num_configs];

    // Бенчмарк
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        int workers = thread_counts[t_idx];
        printf("\n========================================\n");
        printf("  WORKERS: %d\n", workers);
        printf("========================================\n");

        for (int s = 0; s < 7; s++)
        {
            printf("  %-15s : ", strategy_names[s]);
            fflush(stdout);

            double start = get_time_ms();
            pipeline_run(input_paths, output_paths, num_images, 14, s, workers);
            double elapsed = get_time_ms() - start;

            results[s][t_idx] = elapsed;
            printf("%8.2f ms\n", elapsed);

            // Проверка корректности
            for (int j = 0; j < num_images; j++)
            {
                IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
                if (!pipeline_img)
                    continue;

                int eq = imagesEqual(original_images[j], pipeline_img);
                assert(eq);
                cvReleaseImage(&pipeline_img);
            }
        }
    }

    // Итоговый вывод
    printf("\n");
    printf("================================================================================\n");
    printf("                             SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");

    for (int s = 0; s < 7; s++)
    {
        printf("  %-15s : ", strategy_names[s]);
        for (int t_idx = 0; t_idx < num_configs; t_idx++)
        {
            if (t_idx > 0)
                printf(" | ");
            printf("%5.0f ms", results[s][t_idx]);
        }
        printf("\n");
    }

    printf("\n");
    printf("  Workers:          ");
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        if (t_idx > 0)
            printf(" | ");
        printf("%5d   ", thread_counts[t_idx]);
    }
    printf("\n");

    printf("\n");
    printf("  Processed images: %d\n", num_images);
    printf("\n");
    printf("                          TEST 1 PASSED                                        \n");

    // Очистка
    filter_free(&identity);
    for (int i = 0; i < num_images; i++)
    {
        if (original_images[i])
            cvReleaseImage(&original_images[i]);
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(original_images);
    free(input_paths);
    free(output_paths);
}

void testShiftComposition(void)
{
    printf("\n");
    printf("                        TEST 2: SHIFT COMPOSITION                              \n");
    printf("                   (Pipeline Strategies Comparison)                            \n");

    // Создаём фильтры сдвига
    Filter shiftRight = filter_shift_right();
    Filter shiftLeft = filter_shift_left();
    Filter shiftUp = filter_shift_up();
    Filter shiftDown = filter_shift_down();
    Filter shiftDiagUp = filter_shift_diag_up();
    Filter shiftDiagDown = filter_shift_diag_down();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // Загружаем оригинальные изображения
    IplImage **original_images = (IplImage **)malloc(num_images * sizeof(IplImage *));
    for (int j = 0; j < num_images; j++)
    {
        original_images[j] = cvLoadImage(input_paths[j], 1);
        if (!original_images[j])
        {
            printf("ERROR: Failed to load image %s\n", input_paths[j]);
        }
    }

    // Конфигурации потоков
    int thread_counts[] = {1, 2, 4, 8, 12, 16};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Результаты: [композиция][стратегия][конфигурация]
    double results[3][7][num_configs];
    const char *comp_names[3] = {"Right-Left", "Up-Down", "Diag"};

    int filter_indices[3][2] = {
        {15, 16},
        {17, 18},
        {19, 20}};

    // Бенчмарк
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        int workers = thread_counts[t_idx];
        printf("\n========================================\n");
        printf("  WORKERS: %d\n", workers);
        printf("========================================\n");

        for (int comp = 0; comp < 3; comp++)
        {
            printf("\n  %s composition:\n", comp_names[comp]);

            for (int s = 0; s < 7; s++)
            {
                printf("    %-15s : ", strategy_names[s]);
                fflush(stdout);

                double start = get_time_ms();
                pipeline_run(input_paths, output_paths, num_images, filter_indices[comp][0], s, workers);
                pipeline_run(output_paths, output_paths, num_images, filter_indices[comp][1], s, workers);
                double elapsed = get_time_ms() - start;

                results[comp][s][t_idx] = elapsed;
                printf("%8.2f ms\n", elapsed);

                // Проверка корректности
                for (int j = 0; j < num_images; j++)
                {
                    IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
                    int eq = imagesEqual(original_images[j], pipeline_img);
                    assert(eq);
                    cvReleaseImage(&pipeline_img);
                }
            }
        }
    }

    // Итоговый вывод
    printf("\n");
    printf("================================================================================\n");
    printf("                             SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");

    for (int comp = 0; comp < 3; comp++)
    {
        printf("  %s composition:\n", comp_names[comp]);
        for (int s = 0; s < 7; s++)
        {
            printf("    %-15s : ", strategy_names[s]);
            for (int t_idx = 0; t_idx < num_configs; t_idx++)
            {
                if (t_idx > 0)
                    printf(" | ");
                printf("%5.0f ms", results[comp][s][t_idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("  Workers:          ");
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        if (t_idx > 0)
            printf(" | ");
        printf("%5d   ", thread_counts[t_idx]);
    }
    printf("\n");

    printf("\n");
    printf("  Processed images: %d\n", num_images);
    printf("\n");
    printf("                          TEST 2 PASSED                                        \n");

    // Очистка
    filter_free(&shiftRight);
    filter_free(&shiftLeft);
    filter_free(&shiftUp);
    filter_free(&shiftDown);
    filter_free(&shiftDiagUp);
    filter_free(&shiftDiagDown);

    for (int i = 0; i < num_images; i++)
    {
        if (original_images[i])
            cvReleaseImage(&original_images[i]);
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(original_images);
    free(input_paths);
    free(output_paths);
}

void testZeroPadding(void)
{
    printf("\n");
    printf("                        TEST 3: ZERO PADDING                                    \n");
    printf("                   (Pipeline Strategies Comparison)                            \n");

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    int padded_indices[5] = {21, 22, 23, 24, 25};
    const char *filter_names[5] = {"blur3x3", "gaussian3x3", "findedges1", "sharpen1", "emboss1"};

    // Оригинальные фильтры
    Filter original_filters[5];
    original_filters[0] = filter_blur3x3();
    original_filters[1] = filter_gaussian3x3();
    original_filters[2] = filter_findedges1();
    original_filters[3] = filter_sharpen1();
    original_filters[4] = filter_emboss1();

    // Загружаем оригинальные изображения
    IplImage **original_images = (IplImage **)malloc(num_images * sizeof(IplImage *));
    for (int j = 0; j < num_images; j++)
    {
        original_images[j] = cvLoadImage(input_paths[j], 1);
        if (!original_images[j])
        {
            printf("ERROR: Failed to load image %s\n", input_paths[j]);
        }
    }

    // Конфигурации потоков
    int thread_counts[] = {1, 2, 4, 8, 12, 16};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Результаты: [фильтр][стратегия][конфигурация]
    double results[5][7][num_configs];

    // Бенчмарк
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        int workers = thread_counts[t_idx];
        printf("\n========================================\n");
        printf("  WORKERS: %d\n", workers);
        printf("========================================\n");

        for (int f = 0; f < 5; f++)
        {
            printf("\n  Filter: %s\n", filter_names[f]);

            for (int s = 0; s < 7; s++)
            {
                printf("    %-15s : ", strategy_names[s]);
                fflush(stdout);

                double start = get_time_ms();
                pipeline_run(input_paths, output_paths, num_images, padded_indices[f], s, workers);
                double elapsed = get_time_ms() - start;

                results[f][s][t_idx] = elapsed;
                printf("%8.2f ms\n", elapsed);

                // Проверка корректности
                for (int j = 0; j < num_images; j++)
                {
                    IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
                    if (!pipeline_img)
                        continue;

                    IplImage *original_result = cvCreateImage(cvGetSize(original_images[j]),
                                                              original_images[j]->depth,
                                                              original_images[j]->nChannels);
                    applyFilter(original_images[j], original_result, &original_filters[f]);

                    int eq = imagesEqual(pipeline_img, original_result);
                    assert(eq);

                    cvReleaseImage(&pipeline_img);
                    cvReleaseImage(&original_result);
                }
            }
        }
    }

    // Итоговый вывод
    printf("\n");
    printf("================================================================================\n");
    printf("                             SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");

    for (int f = 0; f < 5; f++)
    {
        printf("  Filter: %s\n", filter_names[f]);
        for (int s = 0; s < 7; s++)
        {
            printf("    %-15s : ", strategy_names[s]);
            for (int t_idx = 0; t_idx < num_configs; t_idx++)
            {
                if (t_idx > 0)
                    printf(" | ");
                printf("%5.0f ms", results[f][s][t_idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("  Workers:          ");
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        if (t_idx > 0)
            printf(" | ");
        printf("%5d   ", thread_counts[t_idx]);
    }
    printf("\n");

    printf("\n");
    printf("  Processed images: %d\n", num_images);
    printf("\n");
    printf("                          TEST 3 PASSED                                        \n");

    // Очистка
    for (int i = 0; i < 5; i++)
    {
        filter_free(&original_filters[i]);
    }
    for (int i = 0; i < num_images; i++)
    {
        if (original_images[i])
            cvReleaseImage(&original_images[i]);
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(original_images);
    free(input_paths);
    free(output_paths);
}

void testZeroFilter(void)
{
    printf("\n");
    printf("                        TEST 4: ZERO FILTER                                     \n");
    printf("                   (Pipeline Strategies Comparison)                            \n");

    Filter zero = filter_zero();

    const char *input_dir = "images";
    const char *output_dir = "new_images";

    const char **input_paths = NULL;
    int num_images = get_image_files(input_dir, &input_paths);

    if (num_images <= 0)
    {
        printf("Error: No images found in %s\n", input_dir);
        return;
    }

    printf("Found %d images in %s\n", num_images, input_dir);

    const char **output_paths = (const char **)malloc(num_images * sizeof(const char *));
    generate_output_paths(input_paths, output_paths, num_images, output_dir);

    // Загружаем оригинальные изображения
    IplImage **original_images = (IplImage **)malloc(num_images * sizeof(IplImage *));
    for (int j = 0; j < num_images; j++)
    {
        original_images[j] = cvLoadImage(input_paths[j], 1);
        if (!original_images[j])
        {
            printf("ERROR: Failed to load image %s\n", input_paths[j]);
        }
    }

    // Конфигурации потоков
    int thread_counts[] = {1, 2, 4, 8, 12, 16};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Результаты: [стратегия][конфигурация]
    double results[7][num_configs];

    // Бенчмарк
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        int workers = thread_counts[t_idx];
        printf("\n========================================\n");
        printf("  WORKERS: %d\n", workers);
        printf("========================================\n");

        for (int s = 0; s < 7; s++)
        {
            printf("  %-15s : ", strategy_names[s]);
            fflush(stdout);

            double start = get_time_ms();
            pipeline_run(input_paths, output_paths, num_images, 26, s, workers);
            double elapsed = get_time_ms() - start;

            results[s][t_idx] = elapsed;
            printf("%8.2f ms\n", elapsed);

            // Проверка корректности (изображение должно быть чёрным)
            for (int j = 0; j < num_images; j++)
            {
                IplImage *pipeline_img = cvLoadImage(output_paths[j], 1);
                if (!pipeline_img)
                    continue;

                int step = pipeline_img->widthStep;
                int channels = pipeline_img->nChannels;
                const unsigned char *data = (const unsigned char *)pipeline_img->imageData;
                int all_black = 1;
                for (int y = 0; y < pipeline_img->height && all_black; y++)
                {
                    for (int x = 0; x < pipeline_img->width; x++)
                    {
                        const unsigned char *pixel = data + y * step + x * channels;
                        if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
                        {
                            all_black = 0;
                            break;
                        }
                    }
                }
                assert(all_black);
                cvReleaseImage(&pipeline_img);
            }
        }
    }

    // Итоговый вывод
    printf("\n");
    printf("================================================================================\n");
    printf("                             SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");

    for (int s = 0; s < 7; s++)
    {
        printf("  %-15s : ", strategy_names[s]);
        for (int t_idx = 0; t_idx < num_configs; t_idx++)
        {
            if (t_idx > 0)
                printf(" | ");
            printf("%5.0f ms", results[s][t_idx]);
        }
        printf("\n");
    }

    printf("\n");
    printf("  Workers:          ");
    for (int t_idx = 0; t_idx < num_configs; t_idx++)
    {
        if (t_idx > 0)
            printf(" | ");
        printf("%5d   ", thread_counts[t_idx]);
    }
    printf("\n");

    printf("\n");
    printf("  Processed images: %d\n", num_images);
    printf("\n");
    printf("                          TEST 4 PASSED                                        \n");

    // Очистка
    filter_free(&zero);
    for (int i = 0; i < num_images; i++)
    {
        if (original_images[i])
            cvReleaseImage(&original_images[i]);
        free((void *)input_paths[i]);
        free((void *)output_paths[i]);
    }
    free(original_images);
    free(input_paths);
    free(output_paths);
}

int main(void)
{
    testIdentityFilter();
    testShiftComposition();
    testZeroPadding();
    testZeroFilter();
    return 0;
}