#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include "filter.h"
#include "signal.h"
#include "timing.h"

#include <pthread.h>
#ifdef __linux__
#include <sched.h>       /* for cpu affinity */
#endif

#define MAXWIDTH     40
#define THRESHOLD    2.0
#define ALIENS_LOW   50000.0
#define ALIENS_HIGH  150000.0


static double avg_power(double *data, int num)
{
    double ss = 0.0;
    for (int i = 0; i < num; i++) {
        ss += data[i] * data[i];
    }
    return ss / num;
}

static double max_of(double *data, int num)
{
    double m = data[0];
    for (int i = 1; i < num; i++) {
        if (data[i] > m) m = data[i];
    }
    return m;
}

static double avg_of(double *data, int num)
{
    double s = 0.0;
    for (int i = 0; i < num; i++) {
        s += data[i];
    }
    return s / num;
}

static void remove_dc(double *data, int num)
{
    double dc = avg_of(data, num);

    printf("Removing DC component of %lf\n", dc);

    for (int i = 0; i < num; i++) {
        data[i] -= dc;
    }
}


typedef struct
{
    signal *sig;              /* shared input signal                    */
    int     filter_order;
    double  bandwidth;
    int     band_lo;          /* inclusive                             */
    int     band_hi;          /* inclusive                             */
    double *band_power;       /* shared result array                   */
    int     tid;              /* thread id (for cpu pinning)           */
    int     num_procs;        /* processors user asked us to use       */
} task_t;


static void *band_worker(void *vptr)
{
    task_t *t = (task_t *)vptr;

#ifdef __linux__
    /* round-robin pinning to keep threads from piling onto one core */
    if (t->num_procs > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(t->tid % t->num_procs, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif

    /* stack array for filter taps */
    double coeff[t->filter_order + 1];

    for (int band = t->band_lo; band <= t->band_hi; band++) {

        generate_band_pass(t->sig->Fs,
                           band * t->bandwidth + 0.0001,
                           (band + 1) * t->bandwidth - 0.0001,
                           t->filter_order,
                           coeff);

        hamming_window(t->filter_order, coeff);

        convolve_and_compute_power(t->sig->num_samples,
                                   t->sig->data,
                                   t->filter_order,
                                   coeff,
                                   &t->band_power[band]);
    }

    return NULL;
}

/* parallelised analyse_signal */
static int analyze_signal_parallel(signal *sig,
                                   int     filter_order,
                                   int     num_bands,
                                   int     num_threads,
                                   int     num_procs,
                                   double *lb,
                                   double *ub)
{
    double Fc        = sig->Fs / 2.0;
    double bandwidth = Fc / num_bands;

    remove_dc(sig->data, sig->num_samples);

    double signal_power = avg_power(sig->data, sig->num_samples);
    printf("signal average power:     %lf\n", signal_power);

    resources rstart;
    get_resources(&rstart, THIS_PROCESS);
    double start_time = get_seconds();
    unsigned long long cyc_start = get_cycle_count();

    /* shared array for every band’s power */
    double band_power[num_bands];

    /* spawn threads */
    pthread_t tids[num_threads];
    task_t     tasks[num_threads];

    int bands_per  = num_bands / num_threads;
    int leftover   = num_bands % num_threads;
    int next_band  = 0;

    for (int t = 0; t < num_threads; t++) {

        int take = bands_per + (t < leftover);

        tasks[t].sig          = sig;
        tasks[t].filter_order = filter_order;
        tasks[t].bandwidth    = bandwidth;
        tasks[t].band_lo      = next_band;
        tasks[t].band_hi      = next_band + take - 1;
        tasks[t].band_power   = band_power;
        tasks[t].tid          = t;
        tasks[t].num_procs    = num_procs;

        pthread_create(&tids[t], NULL, band_worker, &tasks[t]);

        next_band += take;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(tids[t], NULL);
    }

    unsigned long long cyc_end = get_cycle_count();
    double end_time = get_seconds();

    resources rend;
    get_resources(&rend, THIS_PROCESS);

    resources rdiff;
    get_resources_diff(&rstart, &rend, &rdiff);

    double max_band_power = max_of(band_power, num_bands);
    double avg_band_power = avg_of(band_power, num_bands);
    int    wow            = 0;
    *lb = *ub = -1.0;

    for (int band = 0; band < num_bands; band++) {

        double band_low  = band * bandwidth + 0.0001;
        double band_high = (band + 1) * bandwidth - 0.0001;

        printf("%5d %20lf to %20lf Hz: %20lf ",
               band, band_low, band_high, band_power[band]);

        int stars = (int)(MAXWIDTH * (band_power[band] / max_band_power));
        for (int i = 0; i < stars; i++) printf("*");

        if ((band_low  >= ALIENS_LOW && band_low  <= ALIENS_HIGH) ||
            (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

            if (band_power[band] > THRESHOLD * avg_band_power) {
                printf("(WOW)");
                wow = 1;
                if (*lb < 0) *lb = band_low;
                *ub = band_high;
            } else {
                printf("(meh)");
            }
        } else {
            printf("(meh)");
        }
        printf("\n");
    }

    printf("Resource usages:\n\
User time        %lf seconds\n\
System time      %lf seconds\n\
Page faults      %ld\n\
Page swaps       %ld\n\
Blocks of I/O    %ld\n\
Signals caught   %ld\n\
Context switches %ld\n",
           rdiff.usertime,
           rdiff.systime,
           rdiff.pagefaults,
           rdiff.pageswaps,
           rdiff.ioblocks,
           rdiff.sigs,
           rdiff.contextswitches);

    printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
           "Note that cycle count only makes sense if the thread stayed on one core\n",
           cyc_end - cyc_start,
           cycles_to_seconds(cyc_end - cyc_start),
           timing_overhead());

    printf("Analysis took %lf seconds by basic timing\n", end_time - start_time);

    return wow;
}

/* usage helper */
static void usage(void)
{
    printf("usage: p_band_scan text|bin|mmap signal_file Fs "
           "filter_order num_bands num_threads num_processors\n");
}

/* main */
int main(int argc, char *argv[])
{
    if (argc != 8) {
        usage();
        return -1;
    }

    char   sig_type     = toupper(argv[1][0]);
    char  *sig_file     = argv[2];
    double Fs           = atof(argv[3]);
    int    filter_order = atoi(argv[4]);
    int    num_bands    = atoi(argv[5]);
    int    num_threads  = atoi(argv[6]);
    int    num_procs    = atoi(argv[7]);

    assert(Fs > 0.0);
    assert(filter_order > 0 && !(filter_order & 1));
    assert(num_bands   > 0);
    assert(num_threads > 0);
    assert(num_procs   > 0);

    printf("type:     %s\n"
           "file:     %s\n"
           "Fs:       %lf Hz\n"
           "order:    %d\n"
           "bands:    %d\n"
           "threads:  %d\n"
           "procs:    %d\n",
           sig_type == 'T' ? "Text" :
           (sig_type == 'B' ? "Binary" :
           (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
           sig_file, Fs, filter_order, num_bands, num_threads, num_procs);

    printf("Load or map file\n");

    signal *sig = NULL;

    if      (sig_type == 'T') sig = load_text_format_signal(sig_file);
    else if (sig_type == 'B') sig = load_binary_format_signal(sig_file);
    else if (sig_type == 'M') sig = map_binary_format_signal(sig_file);
    else {
        printf("Unknown signal type\n");
        return -1;
    }

    if (!sig) {
        printf("Unable to load or map file\n");
        return -1;
    }

    sig->Fs = Fs;

    double lb = 0.0, ub = 0.0;

    int aliens =
        analyze_signal_parallel(sig,
                                filter_order,
                                num_bands,
                                num_threads,
                                num_procs,
                                &lb,
                                &ub);

    if (aliens) {
        printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n",
               lb, ub, (lb + ub) / 2.0);
    } else {
        printf("no aliens\n");
    }

    free_signal(sig);
    return 0;
}

