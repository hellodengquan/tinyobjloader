/*
 * tobj_tess.c -- robust polygon tessellation (see tobj_tess.h).
 *
 * Strategy: project the 3D ring to 2D (Newell normal, bbox fallback), then
 * triangulate the 2D polygon with a convex fan fast-path or robust ear
 * clipping. All 2D work is done in double precision regardless of the input
 * real type. The ear-clip loop is monotone in the number of remaining
 * vertices (an unconditional force-clip backstop), so it always terminates
 * and always emits exactly n-2 triangles for n >= 3.
 *
 * SPDX-License-Identifier: MIT
 */
#include "tobj_tess.h"

/* Relative epsilon for the 2D "is zero area" test (scaled by polygon size). */
#define TOBJ_TESS_REL 1e-9
/* Newell-normal degeneracy threshold (relative to coord magnitude^4). */
#define TOBJ_TESS_REL_N 1e-18
/* Largest n handled with on-stack scratch when no buffer/allocator is given. */
#define TOBJ_TESS_STACK_N 64u
#define TOBJ_TESS_STACK_BYTES 2048u

/* ----------------------------------------------------------------------- */
/* scratch layout                                                          */
/* ----------------------------------------------------------------------- */

typedef struct {
  double *p2d;     /* 2*n */
  uint32_t *nxt;   /* n */
  uint32_t *prv;   /* n */
  uint8_t *reflex; /* (n+7)/8 */
} tobj_tess_work;

static size_t tobj_tess_align(size_t x, size_t a) {
  return (x + (a - 1)) & ~(a - 1);
}

size_t tobj_tess_scratch_size(uint32_t n) {
  size_t sn = (size_t)n;
  size_t a = tobj_tess_align(sn * 2u * sizeof(double), 16);
  size_t b = tobj_tess_align(sn * sizeof(uint32_t), 16);
  size_t c = tobj_tess_align(sn * sizeof(uint32_t), 16);
  size_t d = tobj_tess_align((sn + 7u) / 8u, 16);
  return a + b + c + d;
}

static int tobj_tess_work_bind(tobj_tess_work *w, void *scratch, size_t size,
                               uint32_t n) {
  if (size < tobj_tess_scratch_size(n)) return 0;
  unsigned char *base = (unsigned char *)scratch;
  size_t sn = (size_t)n;
  size_t off = 0;
  w->p2d = (double *)(base + off);
  off += tobj_tess_align(sn * 2u * sizeof(double), 16);
  w->nxt = (uint32_t *)(base + off);
  off += tobj_tess_align(sn * sizeof(uint32_t), 16);
  w->prv = (uint32_t *)(base + off);
  off += tobj_tess_align(sn * sizeof(uint32_t), 16);
  w->reflex = (uint8_t *)(base + off);
  return 1;
}

static void tobj_tess_set_reflex(tobj_tess_work *w, uint32_t i, int v) {
  if (v)
    w->reflex[i >> 3] |= (uint8_t)(1u << (i & 7u));
  else
    w->reflex[i >> 3] &= (uint8_t)~(1u << (i & 7u));
}
static int tobj_tess_is_reflex(tobj_tess_work *w, uint32_t i) {
  return (w->reflex[i >> 3] >> (i & 7u)) & 1u;
}

/* ----------------------------------------------------------------------- */
/* 2D predicates                                                           */
/* ----------------------------------------------------------------------- */

static double tobj_orient2d(const double *a, const double *b, const double *c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

static double tobj_dabs(double x) { return x < 0 ? -x : x; }
static int tobj_is_finite(double x) { return x == x && tobj_dabs(x) < 1e308; }

/* Strictly-inside test for triangle (a,b,c) wound so that `wind` * area > 0. */
static int tobj_point_in_tri(const double *p, const double *a, const double *b,
                             const double *c, double wind, double eps) {
  double d1 = wind * tobj_orient2d(a, b, p);
  double d2 = wind * tobj_orient2d(b, c, p);
  double d3 = wind * tobj_orient2d(c, a, p);
  return d1 > eps && d2 > eps && d3 > eps;
}

/* Non-negative (inside-or-on-boundary) variant. A vertex on an ear's
 * diagonal is still a reason to prefer a different ear. */
static int tobj_point_in_or_on_tri(const double *p, const double *a,
                                   const double *b, const double *c,
                                   double wind, double eps) {
  double d1 = wind * tobj_orient2d(a, b, p);
  double d2 = wind * tobj_orient2d(b, c, p);
  double d3 = wind * tobj_orient2d(c, a, p);
  return d1 >= -eps && d2 >= -eps && d3 >= -eps;
}

/* ----------------------------------------------------------------------- */
/* 2D triangulation core (operates entirely on w->p2d)                     */
/* ----------------------------------------------------------------------- */

static void tobj_tess_emit(uint32_t *out, uint32_t *k, uint32_t a, uint32_t b,
                           uint32_t c) {
  out[*k * 3u + 0u] = a;
  out[*k * 3u + 1u] = b;
  out[*k * 3u + 2u] = c;
  (*k)++;
}

/* Fan from corner 0; always valid count, used for convex/forced cases. */
static void tobj_tess_fan(uint32_t n, uint32_t *out, uint32_t *kp) {
  for (uint32_t k = 1; k + 1 < n; k++) tobj_tess_emit(out, kp, 0, k, k + 1);
}

/* Local feature scale: largest projected edge length. Edge based, so it is
 * independent of where the ring sits in the plane. Returns the squared
 * length to avoid pulling in sqrt() in the no-libc build. */
static double tobj_tess_edge_scale2_sq(const double *p2d, uint32_t n) {
  double m = 0.0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t j = (i + 1u) % n;
    double ex = p2d[2 * j] - p2d[2 * i];
    double ey = p2d[2 * j + 1] - p2d[2 * i + 1];
    double d = ex * ex + ey * ey;
    if (d > m) m = d;
  }
  return m;
}

/* Largest edge length (sqrt of the squared scale above). */
static double tobj_tess_scale2d(const double *p2d, uint32_t n) {
  double m2 = tobj_tess_edge_scale2_sq(p2d, n);
  /* Newton sqrt, good enough for an epsilon scale; no libc needed. */
  if (m2 <= 0.0) return 0.0;
  double x = m2;
  for (int it = 0; it < 24; it++) x = 0.5 * (x + m2 / x);
  return x;
}

static double tobj_tess_signed_area(const double *p2d, uint32_t n) {
  double a = 0.0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t j = (i + 1u) % n;
    a += p2d[2 * i] * p2d[2 * j + 1] - p2d[2 * j] * p2d[2 * i + 1];
  }
  return 0.5 * a;
}

static const double *tobj_tess_pt(tobj_tess_work *w, uint32_t i) {
  return &w->p2d[2 * i];
}

/* Recompute and store the reflex flag for active vertex `cur`. */
static void tobj_tess_update_reflex(tobj_tess_work *w, uint32_t cur,
                                    double wind, double eps_a) {
  uint32_t p = w->prv[cur], n = w->nxt[cur];
  double o = wind * tobj_orient2d(tobj_tess_pt(w, p), tobj_tess_pt(w, cur),
                                  tobj_tess_pt(w, n));
  tobj_tess_set_reflex(w, cur, o < -eps_a ? 1 : 0);
}

/* Triangulate the ring (n>=5) already projected into w->p2d.
 * Returns OK or DEGENERATE_BESTEFFORT. Always writes n-2 triangles. */
static tobj_tess_status tobj_tess_run2d(tobj_tess_work *w, uint32_t n,
                                        uint32_t *out, uint32_t flags) {
  uint32_t k = 0;

  if (flags & (TOBJ_TESS_FLAG_FORCE_FAN | TOBJ_TESS_FLAG_ASSUME_CONVEX)) {
    tobj_tess_fan(n, out, &k);
    return (flags & TOBJ_TESS_FLAG_FORCE_FAN) ? TOBJ_TESS_DEGENERATE_BESTEFFORT
                                              : TOBJ_TESS_OK;
  }

  double scale = tobj_tess_scale2d(w->p2d, n);
  double eps_a = scale * scale * TOBJ_TESS_REL;
  double area = tobj_tess_signed_area(w->p2d, n);
  double wind = area < 0 ? -1.0 : 1.0;

  /* Build ring + reflex flags; count reflex vertices. */
  uint32_t reflex_count = 0;
  for (uint32_t i = 0; i < n; i++) {
    w->nxt[i] = (i + 1u) % n;
    w->prv[i] = (i + n - 1u) % n;
  }
  for (uint32_t i = 0; i < n; i++) {
    double o = wind * tobj_orient2d(tobj_tess_pt(w, w->prv[i]),
                                    tobj_tess_pt(w, i),
                                    tobj_tess_pt(w, w->nxt[i]));
    int rfx = o < -eps_a ? 1 : 0;
    tobj_tess_set_reflex(w, i, rfx);
    reflex_count += (uint32_t)rfx;
  }

  /* Convex (and simple) -> fan fast path. */
  if (reflex_count == 0) {
    tobj_tess_fan(n, out, &k);
    /* near-zero overall area means a degenerate (collinear) polygon */
    return tobj_dabs(area) <= eps_a ? TOBJ_TESS_DEGENERATE_BESTEFFORT
                                    : TOBJ_TESS_OK;
  }

  tobj_tess_status status = TOBJ_TESS_OK;
  uint32_t remaining = n;

  /* Each iteration scans the whole active ring and clips the best available
   * ear: a strictly-clean convex ear first, then a boundary-touching one,
   * then a non-reflex collinear ear, with a winding-corrected force-clip
   * backstop. Scanning (rather than taking the first plausible corner) keeps
   * every emitted triangle inside the polygon even when a diagonal passes
   * through another vertex. */
  while (remaining > 3) {
    int relax = 0; /* 0,1,2 epsilon escalation level */
    int best = -1;
    int best_kind = -1; /* 0=clean, 1=boundary-touching, 2=collinear */

    uint32_t scan = w->nxt[0];
    for (uint32_t probed = 0; probed < remaining; probed++) {
      uint32_t pv = w->prv[scan];
      uint32_t nv = w->nxt[scan];
      const double *pp = tobj_tess_pt(w, pv);
      const double *pc = tobj_tess_pt(w, scan);
      const double *pn = tobj_tess_pt(w, nv);

      double raw = tobj_orient2d(pp, pc, pn);
      double o = wind * raw;
      double eps_scaled =
          eps_a * (relax == 0 ? 1.0 : (relax == 1 ? 1e3 : 1e12));

      int kind = -1;
      if (tobj_dabs(raw) <= eps_scaled) {
        /* collinear ear, but never a reflex corner (that would invert) */
        if (!tobj_tess_is_reflex(w, scan)) kind = 2;
      } else if (o > eps_scaled) {
        int strictly_inside = 0;
        int on_boundary = 0;
        uint32_t r = w->nxt[nv];
        while (r != pv) {
          const double *pr = tobj_tess_pt(w, r);
          if (tobj_point_in_tri(pr, pp, pc, pn, wind, eps_scaled)) {
            strictly_inside = 1;
            break;
          }
          if (tobj_point_in_or_on_tri(pr, pp, pc, pn, wind, eps_scaled)) {
            on_boundary = 1;
          }
          r = w->nxt[r];
        }
        if (!strictly_inside) kind = on_boundary ? 1 : 0;
      }

      if (kind >= 0 && (best < 0 || kind < best_kind)) {
        best = (int)scan;
        best_kind = kind;
        if (kind == 0) break; /* optimal ear found */
      }
      scan = nv;
    }

    if (best < 0) {
      /* No usable ear with the current epsilon: escalate once, then force a
       * winding-corrected clip so progress is guaranteed. */
      if (relax < 2) {
        relax++;
        continue;
      }
      best = (int)w->nxt[0];
      uint32_t probe = w->nxt[0];
      for (uint32_t probed = 0; probed < remaining; probed++) {
        double raw =
            tobj_orient2d(tobj_tess_pt(w, w->prv[probe]),
                          tobj_tess_pt(w, probe),
                          tobj_tess_pt(w, w->nxt[probe]));
        if (wind * raw >= 0.0) {
          best = (int)probe;
          break;
        }
        probe = w->nxt[probe];
      }
      status = TOBJ_TESS_DEGENERATE_BESTEFFORT;
    } else if (best_kind >= 1) {
      status = TOBJ_TESS_DEGENERATE_BESTEFFORT;
    }

    uint32_t cur = (uint32_t)best;
    uint32_t prev = w->prv[cur];
    uint32_t next = w->nxt[cur];
    double raw = tobj_orient2d(tobj_tess_pt(w, prev), tobj_tess_pt(w, cur),
                               tobj_tess_pt(w, next));
    /* never emit an inverted triangle */
    if (wind * raw >= 0.0) {
      tobj_tess_emit(out, &k, prev, cur, next);
    } else {
      tobj_tess_emit(out, &k, prev, next, cur);
    }
    w->nxt[prev] = next;
    w->prv[next] = prev;
    remaining--;
    tobj_tess_update_reflex(w, prev, wind, eps_a);
    tobj_tess_update_reflex(w, next, wind, eps_a);
  }

  /* final triangle from the three survivors, reordered to match the ring
   * winding so a reflex middle vertex cannot produce an inverted triangle */
  uint32_t a = w->nxt[0];
  uint32_t b = w->nxt[a];
  uint32_t c = w->nxt[b];
  double final_raw =
      tobj_orient2d(tobj_tess_pt(w, a), tobj_tess_pt(w, b),
                    tobj_tess_pt(w, c));
  if (wind * final_raw >= 0.0) {
    tobj_tess_emit(out, &k, a, b, c);
  } else {
    tobj_tess_emit(out, &k, a, c, b);
    status = TOBJ_TESS_DEGENERATE_BESTEFFORT;
  }
  return status;
}

/* ----------------------------------------------------------------------- */
/* per-precision wrappers (projection + dispatch)                          */
/* ----------------------------------------------------------------------- */

#define TOBJ_TESS_REAL float
#define TOBJ_TESS_SUF _f
#include "tobj_tess_impl.inc"
#undef TOBJ_TESS_REAL
#undef TOBJ_TESS_SUF

#define TOBJ_TESS_REAL double
#define TOBJ_TESS_SUF _d
#include "tobj_tess_impl.inc"
#undef TOBJ_TESS_REAL
#undef TOBJ_TESS_SUF
