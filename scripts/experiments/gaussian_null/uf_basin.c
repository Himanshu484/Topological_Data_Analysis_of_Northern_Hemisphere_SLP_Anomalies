/*
 * Superlevel-set union-find on a padded 2-D image (8-connectivity, elder rule),
 * a line-for-line C port of scripts/prototype/pers_features.py::features, plus
 * the merge-tree basin of every feature.
 *
 *   uf_features   one sweep: every H0 pair (peak, saddle, parent) and, at the
 *                 moment of death, the weighted area of the dying component,
 *                 i.e. the 8-connected component of {g > saddle} containing the
 *                 peak (the merge-tree basin / outermost closed contour region),
 *                 plus whether that basin contains a given pixel (the pole).
 *   uf_level_area second sweep answering queries (peak, level): weighted area of
 *                 the 8-connected component of {g > level} containing the peak.
 *
 * The pixel order is passed in (numpy stable argsort of -g over non-padding
 * pixels), so ties are broken exactly as in pers_features.py.
 *
 *   cc -O3 -shared -fPIC -o libufbasin.so uf_basin.c
 */
#include <stdlib.h>

static const int NBR[8] = {-1, 0, 1, 0, 0, -1, 0, 1};   /* (dr, dc) pairs, first 4 */
static const int NBD[8] = {-1, -1, -1, 1, 1, -1, 1, 1};  /* diagonal pairs, last 4 */

static int find(int *par, int a) {
    int r = a;
    while (par[r] != r) r = par[r];
    while (par[a] != r) { int n = par[a]; par[a] = r; a = n; }
    return r;
}

static int elder(const double *val, int SEA, int pa, int pb) {
    if (pa == SEA) return 1;
    if (pb == SEA) return 0;
    if (val[pa] != val[pb]) return val[pa] > val[pb];
    return pa < pb;                      /* (val, -idx) lexicographic */
}

typedef struct {
    int *par, *peak, *pole;
    double *area;
    unsigned char *added;
} UF;

static void uf_init(UF *u, int N, const unsigned char *sea, int SEA) {
    u->par = malloc((N + 1) * sizeof(int));
    u->peak = malloc((N + 1) * sizeof(int));
    u->pole = calloc(N + 1, sizeof(int));
    u->area = calloc(N + 1, sizeof(double));
    u->added = malloc(N + 1);
    for (int i = 0; i <= N; i++) { u->par[i] = i; u->peak[i] = -1; }
    u->peak[SEA] = SEA;
    for (int i = 0; i < N; i++) { u->added[i] = sea[i]; if (sea[i]) u->par[i] = SEA; }
}

static void uf_free(UF *u) { free(u->par); free(u->peak); free(u->pole); free(u->area); free(u->added); }

/* Add pixel p. If emit != NULL, record deaths into the output arrays. */
static int uf_add(UF *u, const double *val, const double *parea, int H, int W, int SEA, int p, int polepix,
                  int *o_peak, int *o_sad, int *o_par, double *o_area, int *o_pole, int n, int maxout) {
    int r = p / W, c = p % W;
    int roots[8], nr = 0;
    for (int k = 0; k < 8; k++) {
        int dr = k < 4 ? NBR[2 * k] : NBD[2 * (k - 4)];
        int dc = k < 4 ? NBR[2 * k + 1] : NBD[2 * (k - 4) + 1];
        int rr = r + dr, cc = c + dc, x;
        if (rr < 0 || rr >= H || cc < 0 || cc >= W) x = SEA;
        else {
            int q = rr * W + cc;
            if (!u->added[q]) continue;
            x = find(u->par, q);
        }
        int dup = 0;
        for (int j = 0; j < nr; j++) if (roots[j] == x) { dup = 1; break; }
        if (!dup) roots[nr++] = x;
    }
    u->added[p] = 1;
    if (nr == 0) {
        u->peak[p] = p; u->area[p] = parea[p]; u->pole[p] = (p == polepix);
        return n;
    }
    int el = roots[0];
    for (int j = 1; j < nr; j++) if (elder(val, SEA, u->peak[roots[j]], u->peak[el])) el = roots[j];
    for (int j = 0; j < nr; j++) {
        int x = roots[j];
        if (x == el) continue;
        int pp = u->peak[el];
        if (o_peak && val[u->peak[x]] != val[p] && n < maxout) {
            o_peak[n] = u->peak[x]; o_sad[n] = p; o_par[n] = (pp == SEA) ? -1 : pp;
            o_area[n] = u->area[x]; o_pole[n] = u->pole[x];
            n++;
        }
        u->par[x] = el;
        u->area[el] += u->area[x];
        u->pole[el] |= u->pole[x];
    }
    u->par[p] = el;
    u->area[el] += parea[p];
    u->pole[el] |= (p == polepix);
    return n;
}

int uf_features(const double *val, const unsigned char *sea, const double *parea, const int *order, int norder,
                int H, int W, int polepix,
                int *o_peak, int *o_sad, int *o_par, double *o_area, int *o_pole, int maxout) {
    int N = H * W, SEA = N, n = 0;
    UF u; uf_init(&u, N, sea, SEA);
    for (int i = 0; i < norder; i++)
        n = uf_add(&u, val, parea, H, W, SEA, order[i], polepix, o_peak, o_sad, o_par, o_area, o_pole, n, maxout);
    uf_free(&u);
    return n;
}

/* queries must be sorted by level, descending (qorder gives that order). */
void uf_level_area(const double *val, const unsigned char *sea, const double *parea, const int *order, int norder,
                   int H, int W, const int *qpeak, const double *qlevel, const int *qorder, int nq,
                   double *o_area) {
    int N = H * W, SEA = N, k = 0;
    UF u; uf_init(&u, N, sea, SEA);
    for (int i = 0; i <= norder && k < nq; i++) {
        double v = (i < norder) ? val[order[i]] : -1e300;
        while (k < nq && v <= qlevel[qorder[k]]) {       /* region is {g > level} */
            int q = qorder[k];
            int root = find(u.par, qpeak[q]);
            o_area[q] = (root == SEA || !u.added[qpeak[q]]) ? -1.0 : u.area[root];
            k++;
        }
        if (i < norder) uf_add(&u, val, parea, H, W, SEA, order[i], -1, NULL, NULL, NULL, NULL, NULL, 0, 0);
    }
    uf_free(&u);
}
