/* stream_load.c — controlled CPU memory load (multiple access patterns) for Exp2 / P2.
 *
 * Generates a sustained, intensity-controlled CPU->DRAM memory load so we can measure how
 * it contends with GPU decode on Apple Silicon's shared unified-memory bus.
 *
 *   usage: ./stream_load <seconds> <n_threads> <array_mb> <duty_pct> [mode]
 *   prints: "GBPS <achieved-average>" plus BYTES/ELAPSED/SINK.
 *
 *   mode (P2 — realistic agent-side CPU patterns, not just synthetic STREAM):
 *     0 = stream  : triad a=0.5b+0.5c  (2 read + 1 write, 24 B/elem)   [default; Exp2]
 *     1 = memcpy  : a[i]=b[i]          (1 read + 1 write, 16 B/elem)   copy-heavy work
 *     2 = scan    : sum += b[i]        (1 read sequential, 8 B/elem)   file/mmap-scan analogue
 *     3 = random  : sum += b[idx[i]]   (1 read random,    8 B/elem*)   vector-search / embedding lookup
 *   *random's logical 8 B/elem UNDERSTATES DRAM traffic (each gather pulls a full cache line),
 *    so equal logical GB/s for random means MORE bus pressure than sequential — noted in analysis.
 *
 * Why duty-cycle intensity (not thread count): on the M4 a SINGLE P-core already saturates
 * ~92 GB/s of the ~120 GB/s bus, so more threads don't raise aggregate bandwidth. One P-core
 * thread runs the kernel then sleeps a calibrated fraction; duty_pct in (0,100] sweeps the
 * achieved AVERAGE bandwidth from ~0 to ~92 GB/s, and leaves the other P-cores free for the
 * GPU-dispatch thread (isolating bandwidth contention from P-core scheduling pressure).
 *
 * Controls (RESEARCH.md §6): worker raises QoS to USER_INTERACTIVE (P-core bias). arrays >> LLC
 * (default 96 MB vs 16 MB L2) so traffic hits DRAM. Anti-hoist: stream/memcpy rotate pointers so
 * pass N+1 depends on pass N; read-only scan/random write one element per pass (negligible) so the
 * compiler cannot hoist the loop-invariant sum out of the time loop. GBPS is bytes/WALL-time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <signal.h>
#include <time.h>

static volatile sig_atomic_t stop_requested = 0;
static void request_stop(int sig) { (void)sig; stop_requested = 1; }

typedef struct {
    long n;            /* elements per array */
    double seconds;    /* run duration (wall) */
    double duty;       /* active duty cycle, percent (0,100] */
    int mode;          /* 0 stream, 1 memcpy, 2 scan, 3 random */
    int bpe;           /* bytes counted per element (per mode) */
    double *a, *b, *c;
    long *idx;         /* random-gather index permutation (mode 3) */
    double bytes;      /* out: bytes moved (active passes only) */
    double sink;       /* out: anti dead-code-elimination */
} targ;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_s(double s) {
    if (s <= 0) return;
    struct timespec rq;
    rq.tv_sec = (time_t)s;
    rq.tv_nsec = (long)((s - (double)rq.tv_sec) * 1e9);
    nanosleep(&rq, NULL);
}

static void *worker(void *p) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  /* bias to P-cores */
    targ *t = (targ *)p;
    long n = t->n;
    int mode = t->mode;
    double *a = t->a, *b = t->b, *c = t->c, *tmp;
    long *idx = t->idx;
    double acc = 0.0;
    double t0 = now_s();
    long passes = 0;
    while (!stop_requested && now_s() - t0 < t->seconds) {
        double p0 = now_s();
        if (mode == 0) {                                   /* stream triad: 2R + 1W */
            for (long i = 0; i < n; i++) a[i] = 0.5 * b[i] + 0.5 * c[i];
            tmp = c; c = b; b = a; a = tmp;                /* rotate: pass N+1 depends on N */
        } else if (mode == 1) {                            /* memcpy: 1R + 1W */
            for (long i = 0; i < n; i++) a[i] = b[i];
            tmp = b; b = a; a = tmp;
        } else if (mode == 2) {                            /* scan: 1R sequential (bandwidth-bound) */
            double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; long i = 0;
            for (; i + 8 <= n; i += 8) {                    /* 8 accumulators break the FP-add */
                s0+=b[i];   s1+=b[i+1]; s2+=b[i+2]; s3+=b[i+3];   /* dependency -> memory is */
                s4+=b[i+4]; s5+=b[i+5]; s6+=b[i+6]; s7+=b[i+7];   /* the limit, like a file scan */
            }
            for (; i < n; i++) s0 += b[i];
            double s = s0+s1+s2+s3+s4+s5+s6+s7;
            acc += s; b[passes % n] = s * 1e-18;            /* 1 write/pass defeats hoist */
        } else {                                           /* random gather: 1R random (MLP) */
            double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; long i = 0;
            for (; i + 8 <= n; i += 8) {                    /* 8 outstanding misses (memory-level */
                s0+=b[idx[i]];   s1+=b[idx[i+1]]; s2+=b[idx[i+2]]; s3+=b[idx[i+3]];  /* parallelism, */
                s4+=b[idx[i+4]]; s5+=b[idx[i+5]]; s6+=b[idx[i+6]]; s7+=b[idx[i+7]];  /* like vec search) */
            }
            for (; i < n; i++) s0 += b[idx[i]];
            double s = s0+s1+s2+s3+s4+s5+s6+s7;
            acc += s; b[idx[passes % n]] = s * 1e-18;
        }
        passes++;
        if (t->duty < 100.0) {                             /* throttle to target average BW */
            double pass_t = now_s() - p0;
            sleep_s(pass_t * (100.0 - t->duty) / t->duty);
        }
    }
    t->bytes = (double)passes * (double)n * (double)t->bpe;
    t->sink = (mode >= 2) ? acc + b[0] : b[0] + b[n - 1];
    return NULL;
}

int main(int argc, char **argv) {
    signal(SIGTERM, request_stop);
    signal(SIGINT, request_stop);
    double seconds = argc > 1 ? atof(argv[1]) : 3.0;
    int nthreads   = argc > 2 ? atoi(argv[2]) : 1;
    long mb        = argc > 3 ? atol(argv[3]) : 96;
    double duty    = argc > 4 ? atof(argv[4]) : 100.0;
    int mode       = argc > 5 ? atoi(argv[5]) : 0;
    if (nthreads <= 0 || duty <= 0.0) { printf("GBPS 0.000\nBYTES 0\nELAPSED 0\nSINK 0\n"); return 0; }
    if (nthreads > 64) nthreads = 64;
    if (duty > 100.0) duty = 100.0;
    if (mode < 0 || mode > 3) mode = 0;
    int bpe = (mode == 0) ? 24 : (mode == 1) ? 16 : 8;

    long n = (mb * 1024L * 1024L) / (long)sizeof(double);
    pthread_t th[64];
    targ ta[64];
    for (int k = 0; k < nthreads; k++) {
        ta[k].n = n; ta[k].seconds = seconds; ta[k].duty = duty; ta[k].mode = mode; ta[k].bpe = bpe;
        ta[k].a = malloc(n * sizeof(double));
        ta[k].b = malloc(n * sizeof(double));
        ta[k].c = malloc(n * sizeof(double));
        ta[k].idx = NULL;
        if (!ta[k].a || !ta[k].b || !ta[k].c) { fprintf(stderr, "alloc fail\n"); return 1; }
        for (long i = 0; i < n; i++) { ta[k].b[i] = 1.0; ta[k].c[i] = 2.0; ta[k].a[i] = 0.0; }
        if (mode == 3) {                                   /* random permutation index (fixed seed) */
            ta[k].idx = malloc(n * sizeof(long));
            if (!ta[k].idx) { fprintf(stderr, "alloc fail\n"); return 1; }
            for (long i = 0; i < n; i++) ta[k].idx[i] = i;
            srandom((unsigned)(1234 + k));
            for (long i = n - 1; i > 0; i--) {             /* Fisher-Yates shuffle */
                long j = random() % (i + 1);
                long tmp = ta[k].idx[i]; ta[k].idx[i] = ta[k].idx[j]; ta[k].idx[j] = tmp;
            }
        }
        ta[k].bytes = 0; ta[k].sink = 0;
    }
    double t0 = now_s();
    for (int k = 0; k < nthreads; k++) pthread_create(&th[k], NULL, worker, &ta[k]);
    for (int k = 0; k < nthreads; k++) pthread_join(th[k], NULL);
    double elapsed = now_s() - t0;

    double total = 0, sink = 0;
    for (int k = 0; k < nthreads; k++) { total += ta[k].bytes; sink += ta[k].sink; }
    printf("GBPS %.3f\nBYTES %.0f\nELAPSED %.3f\nSINK %.3e\n",
           total / 1e9 / elapsed, total, elapsed, sink);
    return 0;
}
