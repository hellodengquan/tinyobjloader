//
// Robust polygon triangulation shared by every C++ loading path (legacy
// LoadObj, optimized LoadObjOpt incl. its multithreaded mode, and the
// experimental stream loader).
//
// The previous implementation picked two global coordinate axes and ran a
// heuristic ear clip whose orientation test mixed a difference-based cross
// product with an absolute-coordinate triangle area. That made the result
// depend on the polygon's position in space: translating a face could change
// the chosen ears, emit triangles outside the polygon (overlapping / flipped
// winding) or collapse everything to zero-area triangles for large offsets.
//
// This module instead:
//   1. computes a Newell normal (translation invariant by construction) and
//      projects the 3D ring onto an oriented in-plane basis, recentered at the
//      first corner so large world offsets cannot cancel in float;
//   2. performs all 2D predicates in double precision with a relative,
//      scale-aware epsilon;
//   3. takes a convex fan fast path (no quality regression for convex faces)
//      and otherwise runs ear clipping with a whole-ring ear scan: clean
//      convex ears are preferred over ears whose diagonal touches another
//      vertex, with a collinear-ear rule and an unconditional force-clip
//      backstop so it always terminates and always emits n-2 triangles;
//   4. returns a status so callers can warn on degenerate / non-finite input
//      instead of silently dropping coverage.
//
// Output winding matches the input ring winding, and the emitted indices are
// ring-local positions (0..n-1); callers map them back to their own
// per-corner attribute records, which keeps vertex / normal / texcoord
// indices, materials and smoothing groups attached to the same corners.
//
// SPDX-License-Identifier: MIT
//
#ifndef TINYOBJ_POLYGON_TRIANGULATION_H_
#define TINYOBJ_POLYGON_TRIANGULATION_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyobj {
namespace polygon_triangulation {

// Result of triangulating one ring.
enum TriangulationStatus {
  kTriangulationOk = 0,               // clean triangulation
  kTriangulationDegenerateBestEffort  // n-2 tris emitted, but input was bad
};

namespace detail {

// Relative epsilon for 2D "is zero area" tests (scaled by polygon size^2).
static const double kRelEps = 1e-9;
// Newell-normal degeneracy threshold (relative to coord magnitude^4).
static const double kRelNormalEps = 1e-18;

inline double AbsD(double x) { return x < 0.0 ? -x : x; }
inline bool IsFiniteD(double x) {
  return x == x && AbsD(x) < 1e308;
}

// (b-a) x (c-a) in 2D. Positive means a->b->c is counter-clockwise.
inline double Orient2d(double ax, double ay, double bx, double by,
                       double cx, double cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Strictly-inside test for triangle (a,b,c) wound so that `wind` * area > 0.
inline bool PointInTriangle(double px, double py, double ax, double ay,
                            double bx, double by, double cx, double cy,
                            double wind, double eps) {
  const double d1 = wind * Orient2d(ax, ay, bx, by, px, py);
  const double d2 = wind * Orient2d(bx, by, cx, cy, px, py);
  const double d3 = wind * Orient2d(cx, cy, ax, ay, px, py);
  return d1 > eps && d2 > eps && d3 > eps;
}

// Non-negative (inside-or-on-boundary) variant. A vertex sitting exactly on
// an ear's diagonal is still a reason to prefer a different ear, since
// clipping through it would overlap adjacent triangles.
inline bool PointInOrOnTriangle(double px, double py, double ax, double ay,
                                double bx, double by, double cx, double cy,
                                double wind, double eps) {
  const double d1 = wind * Orient2d(ax, ay, bx, by, px, py);
  const double d2 = wind * Orient2d(bx, by, cx, cy, px, py);
  const double d3 = wind * Orient2d(cx, cy, ax, ay, px, py);
  return d1 >= -eps && d2 >= -eps && d3 >= -eps;
}

inline double SignedArea2d(const std::vector<double> &p2d, uint32_t n) {
  double a = 0.0;
  for (uint32_t i = 0; i < n; i++) {
    const uint32_t j = (i + 1u) % n;
    a += p2d[2 * i] * p2d[2 * j + 1] - p2d[2 * j] * p2d[2 * i + 1];
  }
  return 0.5 * a;
}

// Local feature scale: largest edge length in the projected ring. Using
// edge lengths rather than absolute coordinates keeps the collinearity /
// reflex tolerances independent of where the ring sits in the plane.
inline double Scale2d(const std::vector<double> &p2d, uint32_t n) {
  double m = 0.0;
  for (uint32_t i = 0; i < n; i++) {
    const uint32_t j = (i + 1u) % n;
    const double ex = p2d[2 * j] - p2d[2 * i];
    const double ey = p2d[2 * j + 1] - p2d[2 * i + 1];
    m = (std::max)(m, std::sqrt(ex * ex + ey * ey));
  }
  return m;
}

// Project the 3D ring into p2d (doubles, recentered at corner 0). Returns
// false when the Newell normal is degenerate and a bounding-box fallback
// projection had to be used. `points` is laid out as n triples with stride 3.
template <typename Real>
bool ProjectRing(const Real *points, uint32_t n,
                 std::vector<double> *p2d_out) {
  std::vector<double> &p2d = *p2d_out;
  p2d.resize(static_cast<size_t>(2) * n);

  const double cx = static_cast<double>(points[0]);
  const double cy = static_cast<double>(points[1]);
  const double cz = static_cast<double>(points[2]);

  // Edge-length scale. Built from coordinate differences, so it is
  // unaffected by any rigid translation of the ring (an absolute-coordinate
  // scale would make the Newell-normal and collinearity tolerances loosen
  // with the model's distance from the origin).
  double edge_scale2 = 0.0;

  // Newell's method. Differences only, so the normal is unaffected by any
  // rigid translation of the ring.
  double nx = 0.0, ny = 0.0, nz = 0.0;
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t j = (k + 1u) % n;
    const double ax = static_cast<double>(points[3 * k]);
    const double ay = static_cast<double>(points[3 * k + 1]);
    const double az = static_cast<double>(points[3 * k + 2]);
    const double bx = static_cast<double>(points[3 * j]);
    const double by = static_cast<double>(points[3 * j + 1]);
    const double bz = static_cast<double>(points[3 * j + 2]);
    nx += (ay - by) * (az + bz);
    ny += (az - bz) * (ax + bx);
    nz += (ax - bx) * (ay + by);
    const double ex = ax - bx, ey = ay - by, ez = az - bz;
    edge_scale2 = (std::max)(edge_scale2, ex * ex + ey * ey + ez * ez);
  }

  const double len2 = nx * nx + ny * ny + nz * nz;
  // |Newell normal| scales like edge^2; compare against edge^4 * eps.
  const double thresh =
      edge_scale2 * edge_scale2 * kRelNormalEps;

  if (IsFiniteD(len2) && len2 > thresh && len2 > 0.0) {
    // In-plane basis from the Newell normal. Unnormalized is fine: only the
    // sign of 2D orientations matters, and the orientation convention keeps
    // the projected ring winding identical to the 3D ring winding.
    const double wx = nx, wy = ny, wz = nz;
    const double axd = AbsD(wx), ayd = AbsD(wy), azd = AbsD(wz);
    double sx = 0.0, sy = 0.0, sz = 0.0;
    if (axd <= ayd && axd <= azd) {
      sx = 1.0;
    } else if (ayd <= azd) {
      sy = 1.0;
    } else {
      sz = 1.0;
    }
    // u = w x s, v = w x u.
    const double ux = wy * sz - wz * sy;
    const double uy = wz * sx - wx * sz;
    const double uz = wx * sy - wy * sx;
    const double vx = wy * uz - wz * uy;
    const double vy = wz * ux - wx * uz;
    const double vz = wx * uy - wy * ux;
    for (uint32_t k = 0; k < n; k++) {
      const double dx = static_cast<double>(points[3 * k]) - cx;
      const double dy = static_cast<double>(points[3 * k + 1]) - cy;
      const double dz = static_cast<double>(points[3 * k + 2]) - cz;
      p2d[2 * k] = dx * ux + dy * uy + dz * uz;
      p2d[2 * k + 1] = dx * vx + dy * vy + dz * vz;
    }
    return true;
  }

  // Fallback: drop the smallest-extent world axis and use the other two.
  double mn[3], mx[3];
  mn[0] = mx[0] = static_cast<double>(points[0]);
  mn[1] = mx[1] = static_cast<double>(points[1]);
  mn[2] = mx[2] = static_cast<double>(points[2]);
  for (uint32_t k = 0; k < n; k++) {
    for (int c = 0; c < 3; c++) {
      const double val = static_cast<double>(points[3 * k + c]);
      if (val < mn[c]) mn[c] = val;
      if (val > mx[c]) mx[c] = val;
    }
  }
  const double span[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
  int drop = 0;
  if (span[1] < span[0]) drop = 1;
  if (span[2] < span[drop]) drop = 2;
  const int a0 = (drop == 0) ? 1 : 0;
  const int a1 = (drop == 2) ? 1 : 2;
  for (uint32_t k = 0; k < n; k++) {
    p2d[2 * k] =
        static_cast<double>(points[3 * k + a0]) - (drop == 0 ? cy : cx);
    p2d[2 * k + 1] =
        static_cast<double>(points[3 * k + a1]) - (drop == 2 ? cy : cz);
  }
  return false;
}

inline void EmitTriangle(std::vector<uint32_t> *out, uint32_t a, uint32_t b,
                         uint32_t c) {
  out->push_back(a);
  out->push_back(b);
  out->push_back(c);
}

inline void FanTriangles(uint32_t n, std::vector<uint32_t> *out) {
  for (uint32_t k = 1; k + 1 < n; k++) {
    EmitTriangle(out, 0, k, k + 1);
  }
}

// Ear clipping for n >= 5 on an already-projected ring. Always emits n-2
// triangles. Mirrors the proven C11 tobj_tess_run2d strategy.
inline TriangulationStatus Run2d(std::vector<double> &p2d, uint32_t n,
                                 std::vector<uint32_t> *out) {
  out->clear();
  out->reserve(static_cast<size_t>(3) * (n - 2u));

  const double scale = Scale2d(p2d, n);
  const double eps_a = scale * scale * kRelEps;
  const double area = SignedArea2d(p2d, n);
  const double wind = area < 0.0 ? -1.0 : 1.0;

  std::vector<uint32_t> nxt(n), prv(n);
  std::vector<unsigned char> reflex(n, 0);
  uint32_t reflex_count = 0;
  for (uint32_t i = 0; i < n; i++) {
    nxt[i] = (i + 1u) % n;
    prv[i] = (i + n - 1u) % n;
  }
  for (uint32_t i = 0; i < n; i++) {
    const double o =
        wind * Orient2d(p2d[2 * prv[i]], p2d[2 * prv[i] + 1], p2d[2 * i],
                        p2d[2 * i + 1], p2d[2 * nxt[i]],
                        p2d[2 * nxt[i] + 1]);
    const unsigned char rfx = o < -eps_a ? 1u : 0u;
    reflex[i] = rfx;
    reflex_count += rfx;
  }

  // Convex (and simple) -> fan fast path.
  if (reflex_count == 0) {
    FanTriangles(n, out);
    return AbsD(area) <= eps_a ? kTriangulationDegenerateBestEffort
                               : kTriangulationOk;
  }

  TriangulationStatus status = kTriangulationOk;
  uint32_t remaining = n;

  // Each iteration scans the whole active ring and clips the best available
  // ear. Preference order: a strictly-clean convex ear (no other vertex even
  // on its boundary), then a convex ear whose boundary touches a vertex, then
  // a non-reflex collinear ear, and finally a force-clip backstop. Scanning
  // rather than clipping the first plausible corner keeps the triangles
  // inside the polygon even when diagonals pass through other vertices.
  while (remaining > 3) {
    int relax = 0;  // 0,1,2 epsilon escalation level
    int best = -1;   // ring vertex chosen as the ear tip
    int best_kind = -1;  // 0=clean, 1=boundary-touching, 2=collinear

    uint32_t scan = nxt[0];
    for (uint32_t probed = 0; probed < remaining; probed++) {
      const uint32_t prev = prv[scan];
      const uint32_t next = nxt[scan];
      const double ppx = p2d[2 * prev], ppy = p2d[2 * prev + 1];
      const double pcx = p2d[2 * scan], pcy = p2d[2 * scan + 1];
      const double pnx = p2d[2 * next], pny = p2d[2 * next + 1];
      const double raw = Orient2d(ppx, ppy, pcx, pcy, pnx, pny);
      const double o = wind * raw;
      const double eps_scaled =
          eps_a * (relax == 0 ? 1.0 : (relax == 1 ? 1e3 : 1e12));

      int kind = -1;
      if (AbsD(raw) <= eps_scaled) {
        // Collinear ear, but never a reflex corner (that would invert).
        if (!reflex[scan]) kind = 2;
      } else if (o > eps_scaled) {
        bool strictly_inside = false;
        bool on_boundary = false;
        uint32_t r = nxt[next];
        while (r != prev) {
          if (PointInTriangle(p2d[2 * r], p2d[2 * r + 1], ppx, ppy, pcx, pcy,
                              pnx, pny, wind, eps_scaled)) {
            strictly_inside = true;
            break;
          }
          if (PointInOrOnTriangle(p2d[2 * r], p2d[2 * r + 1], ppx, ppy, pcx,
                                  pcy, pnx, pny, wind, eps_scaled)) {
            on_boundary = true;
          }
          r = nxt[r];
        }
        if (!strictly_inside) kind = on_boundary ? 1 : 0;
      }

      if (kind >= 0 && (best < 0 || kind < best_kind)) {
        best = static_cast<int>(scan);
        best_kind = kind;
        if (kind == 0) break;  // optimal ear found
      }
      scan = next;
    }

    if (best < 0) {
      // No usable ear even with exact predicates: escalate the epsilon once,
      // then force-clip with correct winding so progress is guaranteed.
      if (relax < 2) {
        relax++;
        continue;
      }
      // Backstop: pick any convex (or collinear) corner if one exists,
      // otherwise clip vertex 0 with corrected winding.
      best = static_cast<int>(nxt[0]);
      uint32_t probe = nxt[0];
      for (uint32_t probed = 0; probed < remaining; probed++) {
        const double raw =
            Orient2d(p2d[2 * prv[probe]], p2d[2 * prv[probe] + 1],
                     p2d[2 * probe], p2d[2 * probe + 1],
                     p2d[2 * nxt[probe]], p2d[2 * nxt[probe] + 1]);
        if (wind * raw >= 0.0) {
          best = static_cast<int>(probe);
          break;
        }
        probe = nxt[probe];
      }
      status = kTriangulationDegenerateBestEffort;
    } else if (best_kind >= 1) {
      status = kTriangulationDegenerateBestEffort;
    }

    const uint32_t cur = static_cast<uint32_t>(best);
    const uint32_t prev = prv[cur];
    const uint32_t next = nxt[cur];
    const double raw = Orient2d(p2d[2 * prev], p2d[2 * prev + 1],
                                p2d[2 * cur], p2d[2 * cur + 1],
                                p2d[2 * next], p2d[2 * next + 1]);
    // Never emit an inverted triangle.
    if (wind * raw >= 0.0) {
      EmitTriangle(out, prev, cur, next);
    } else {
      EmitTriangle(out, prev, next, cur);
    }
    nxt[prev] = next;
    prv[next] = prev;
    remaining--;

    // Refresh reflex flags of the two neighbours whose angle changed.
    for (int u = 0; u < 2; u++) {
      const uint32_t q = u == 0 ? prev : next;
      const double qo =
          wind * Orient2d(p2d[2 * prv[q]], p2d[2 * prv[q] + 1], p2d[2 * q],
                          p2d[2 * q + 1], p2d[2 * nxt[q]],
                          p2d[2 * nxt[q] + 1]);
      reflex[q] = qo < -eps_a ? 1u : 0u;
    }
  }

  // Final triangle from the three survivors. Reorder it to match the ring
  // winding: the ear-clip loop can leave a reflex vertex in the middle, and
  // emitting (a, b, c) blindly would then produce an inverted triangle whose
  // "coverage" lies outside the original polygon.
  const uint32_t a = nxt[0];
  const uint32_t b = nxt[a];
  const uint32_t c = nxt[b];
  const double final_raw =
      Orient2d(p2d[2 * a], p2d[2 * a + 1], p2d[2 * b], p2d[2 * b + 1],
               p2d[2 * c], p2d[2 * c + 1]);
  if (wind * final_raw >= 0.0) {
    EmitTriangle(out, a, b, c);
  } else {
    EmitTriangle(out, a, c, b);
    status = kTriangulationDegenerateBestEffort;
  }
  return status;
}

// Quad path: pick an interior diagonal. A diagonal is valid when both
// resulting triangles have the ring's winding and non-tiny area; otherwise
// fall back to the shorter 3D diagonal (legacy convex behaviour).
template <typename Real>
TriangulationStatus TriangulateQuad(const Real *points,
                                    std::vector<double> &p2d,
                                    std::vector<uint32_t> *out) {
  const double scale = Scale2d(p2d, 4);
  const double eps = scale * scale * kRelEps;
  const double *p0 = &p2d[0], *p1 = &p2d[2], *p2 = &p2d[4], *p3 = &p2d[6];
  const double o012 = Orient2d(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]);
  const double o023 = Orient2d(p0[0], p0[1], p2[0], p2[1], p3[0], p3[1]);
  const double o123 = Orient2d(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]);
  const double o130 = Orient2d(p1[0], p1[1], p3[0], p3[1], p0[0], p0[1]);

  const bool d02 = AbsD(o012) > eps && AbsD(o023) > eps &&
                   ((o012 > 0.0) == (o023 > 0.0));
  const bool d13 = AbsD(o123) > eps && AbsD(o130) > eps &&
                   ((o123 > 0.0) == (o130 > 0.0));

  TriangulationStatus status = kTriangulationOk;
  bool split02;
  if (d02 && !d13) {
    split02 = true;
  } else if (d13 && !d02) {
    split02 = false;
  } else {
    // Convex (both valid) or degenerate (neither): shorter 3D diagonal.
    const double e02x = static_cast<double>(points[6]) -
                        static_cast<double>(points[0]);
    const double e02y = static_cast<double>(points[7]) -
                        static_cast<double>(points[1]);
    const double e02z = static_cast<double>(points[8]) -
                        static_cast<double>(points[2]);
    const double e13x = static_cast<double>(points[9]) -
                        static_cast<double>(points[3]);
    const double e13y = static_cast<double>(points[10]) -
                        static_cast<double>(points[4]);
    const double e13z = static_cast<double>(points[11]) -
                        static_cast<double>(points[5]);
    const double s02 = e02x * e02x + e02y * e02y + e02z * e02z;
    const double s13 = e13x * e13x + e13y * e13y + e13z * e13z;
    split02 = s02 <= s13;
    if (!d02 && !d13) status = kTriangulationDegenerateBestEffort;
  }

  if (split02) {
    EmitTriangle(out, 0, 1, 2);
    EmitTriangle(out, 0, 2, 3);
  } else {
    EmitTriangle(out, 0, 1, 3);
    EmitTriangle(out, 1, 2, 3);
  }
  return status;
}

}  // namespace detail

// Double-precision entry point. All geometric predicates run on these
// coordinates directly.
inline TriangulationStatus TriangulateRingDouble(
    const double *points, uint32_t n,
    std::vector<uint32_t> *out_indices) {
  if (n < 3 || points == NULL || out_indices == NULL) {
    return kTriangulationDegenerateBestEffort;
  }
  for (uint32_t i = 0; i < 3u * n; i++) {
    if (!detail::IsFiniteD(points[i])) {
      return kTriangulationDegenerateBestEffort;
    }
  }

  std::vector<double> p2d;
  const bool proj_ok = detail::ProjectRing<double>(points, n, &p2d);

  TriangulationStatus status;
  if (n == 3) {
    detail::EmitTriangle(out_indices, 0, 1, 2);
    const double ar = detail::AbsD(
        detail::Orient2d(p2d[0], p2d[1], p2d[2], p2d[3], p2d[4], p2d[5]));
    const double scale = detail::Scale2d(p2d, n);
    status = ar <= scale * scale * detail::kRelEps
                 ? kTriangulationDegenerateBestEffort
                 : kTriangulationOk;
  } else if (n == 4) {
    status = detail::TriangulateQuad<double>(points, p2d, out_indices);
  } else {
    status = detail::Run2d(p2d, n, out_indices);
  }

  if (!proj_ok && status == kTriangulationOk) {
    status = kTriangulationDegenerateBestEffort;
  }
  return status;
}

// Triangulate one 3D polygon ring.
//
// `points` points to n*3 coordinates of type Real with stride 3. Coordinates
// are promoted to double before any geometric decision is made, which keeps
// the result identical for any rigid translation of the ring (float inputs
// would otherwise lose small features at large world offsets). On success
// exactly n-2 triangles (3*(n-2) ring-local indices) are appended to
// `out_indices` and kTriangulationOk or kTriangulationDegenerateBestEffort is
// returned. Non-finite coordinates are rejected: nothing is appended and
// callers are expected to warn instead of silently losing the face.
template <typename Real>
TriangulationStatus TriangulateRing(const Real *points, uint32_t n,
                                    std::vector<uint32_t> *out_indices) {
  if (n < 3 || points == NULL || out_indices == NULL) {
    return kTriangulationDegenerateBestEffort;
  }
  // Promote first so the finite check and every predicate see the same
  // values the geometry actually runs on.
  std::vector<double> promoted(static_cast<size_t>(3) * n);
  for (uint32_t i = 0; i < 3u * n; i++) {
    promoted[i] = static_cast<double>(points[i]);
  }
  return TriangulateRingDouble(promoted.data(), n, out_indices);
}

// Float/double overload: no copy needed when the input is already double.
template <>
inline TriangulationStatus TriangulateRing<double>(
    const double *points, uint32_t n,
    std::vector<uint32_t> *out_indices) {
  return TriangulateRingDouble(points, n, out_indices);
}

}  // namespace polygon_triangulation
}  // namespace tinyobj

#endif  // TINYOBJ_POLYGON_TRIANGULATION_H_
