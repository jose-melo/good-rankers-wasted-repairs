// Toy system of Figure 1, boxes 1 and 2: five features on a chain x1 -> x2 -> x3 -> x4 -> x5 (unit variance,
// coefficient A), linear predictors, and the three tests of the figure (KS, conditional, leave-one-out) on one window.
// Box 3 of Figure 1 is not computed here: it reads results/graphs.csv.
var TOY = (function () {
  var D = 5, A = 0.8, NFIT = 500, NCAL = 500;
  function rng(seed) { var a = seed >>> 0; return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function row(r, shift, from) {           // shift added to the mechanism of node `from` (propagating change); -1 = none
    var x = new Array(D), s = Math.sqrt(1 - A * A);
    x[0] = gauss(r) + (from === 0 ? shift : 0);
    for (var i = 1; i < D; i++) x[i] = A * x[i - 1] + s * gauss(r) + (from === i ? shift : 0);
    return x;
  }
  function rows(r, n, shift, from) { var X = []; for (var i = 0; i < n; i++) X.push(row(r, shift || 0, from === undefined ? -1 : from)); return X; }
  function solve(M, b) {                    // Gaussian elimination, small system
    var n = b.length, i, j, k;
    for (i = 0; i < n; i++) { M[i] = M[i].slice(); M[i].push(b[i]); }
    for (i = 0; i < n; i++) {
      var p = i; for (k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
      var t = M[i]; M[i] = M[p]; M[p] = t;
      for (k = i + 1; k < n; k++) { var f = M[k][i] / M[i][i]; for (j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) { var s = M[i][n]; for (j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }
  function fit(X, j, cols) {                // least squares of column j on columns `cols`, with intercept
    var p = cols.length + 1, M = [], b = [], i, a, c;
    for (a = 0; a < p; a++) { M.push(new Array(p).fill(0)); b.push(0); }
    for (i = 0; i < X.length; i++) {
      var z = [1]; for (a = 0; a < cols.length; a++) z.push(X[i][cols[a]]);
      for (a = 0; a < p; a++) { b[a] += z[a] * X[i][j]; for (c = 0; c < p; c++) M[a][c] += z[a] * z[c]; }
    }
    for (a = 0; a < p; a++) M[a][a] += 1e-8;
    return { j: j, cols: cols, beta: solve(M, b) };
  }
  function resid(m, x) { var s = m.beta[0]; for (var a = 0; a < m.cols.length; a++) s += m.beta[a + 1] * x[m.cols[a]]; return x[m.j] - s; }
  function Phi(z) { var t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989423 * Math.exp(-z * z / 2);
    var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; }
  function countBelow(sorted, v) { var lo = 0, hi = sorted.length; while (lo < hi) { var m = (lo + hi) >> 1; if (sorted[m] < v) lo = m + 1; else hi = m; } return lo; }
  // One-sided Mann-Whitney: are the window's absolute residuals larger than the calibration ones?
  function mwP(calSorted, win) {
    var n = calSorted.length, m = win.length, U = 0;
    for (var i = 0; i < m; i++) U += countBelow(calSorted, win[i]);
    if (m === 1) { var p1 = (1 + (n - U)) / (n + 1); return { p: p1, z: -p1 }; }   // exact conformal p-value for one row
    var mu = n * m / 2, sd = Math.sqrt(n * m * (n + m + 1) / 12), z = (U - mu - 0.5) / sd;
    return { p: 1 - Phi(z), z: z };                         // z kept for ranking: p-values underflow to 0 and would tie
  }
  function ksP(refSorted, win) {            // two-sample Kolmogorov-Smirnov, asymptotic p-value
    var w = win.slice().sort(function (a, b) { return a - b; }), n = refSorted.length, m = w.length, i = 0, j = 0, d = 0;
    while (i < n && j < m) { var v = Math.min(refSorted[i], w[j]); while (i < n && refSorted[i] <= v) i++; while (j < m && w[j] <= v) j++; d = Math.max(d, Math.abs(i / n - j / m)); }
    var en = Math.sqrt(n * m / (n + m)), lam = (en + 0.12 + 0.11 / en) * d, s = 0;
    for (var k = 1; k < 101; k++) s += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
    return { p: Math.max(0, Math.min(1, s)), z: lam };
  }
  function holm(p, alpha) {                 // returns which hypotheses are rejected
    var idx = p.map(function (_, i) { return i; }).sort(function (a, b) { return p[a] - p[b]; }), out = p.map(function () { return false; });
    for (var r = 0; r < idx.length; r++) { if (p[idx[r]] <= alpha / (idx.length - r)) out[idx[r]] = true; else break; }
    return out;
  }
  function absSorted(m, X) { return X.map(function (x) { return Math.abs(resid(m, x)); }).sort(function (a, b) { return a - b; }); }
  function others(j, k) { var c = []; for (var a = 0; a < D; a++) if (a !== j && a !== k) c.push(a); return c; }
  function makeUnit(seed) {                 // a unit = its own reference: fitting rows + calibration rows + fitted predictors
    var r = rng(seed), Xfit = rows(r, NFIT), Xcal = rows(r, NCAL), u = { seed: seed, fit: Xfit, cal: Xcal, full: [], loo: [], marg: [] };
    for (var j = 0; j < D; j++) {
      var mf = fit(Xfit, j, others(j, -1)); mf.cal = absSorted(mf, Xcal); u.full.push(mf);
      var L = []; for (var k = 0; k < D; k++) { if (k === j) { L.push(null); continue; } var ml = fit(Xfit, j, others(j, k)); ml.cal = absSorted(ml, Xcal); L.push(ml); } u.loo.push(L);
      u.marg.push(Xcal.map(function (x) { return x[j]; }).sort(function (a, b) { return a - b; }));
    }
    return u;
  }
  // Change types. `acted` is the answer key: the sensors something was done to.
  var TYPES = {
    none: { acted: [] }, offset: { acted: [2] }, stuck: { acted: [2] }, half: { acted: [2] }, broken: { acted: [2] },
    two: { acted: [1, 3] }, propagating: { acted: [2] }, select: { acted: [] }
  };
  function makeWindow(r, type, mag, w) {
    var X, i;
    if (type === 'propagating') X = rows(r, w, mag, 2);
    else if (type === 'select') { var pool = rows(r, 12 * w), wt = pool.map(function (x) { return Math.exp(mag * x[2]); }), tot = wt.reduce(function (a, b) { return a + b; }, 0); X = [];
      for (i = 0; i < w; i++) { var t = r() * tot, k = 0; while (t > wt[k] && k < pool.length - 1) { t -= wt[k]; k++; } X.push(pool[k].slice()); } }
    else X = rows(r, w);
    if (type === 'offset') for (i = 0; i < w; i++) X[i][2] += mag;
    if (type === 'half') for (i = Math.floor(w / 2); i < w; i++) X[i][2] += mag;
    if (type === 'stuck') { var c = gauss(r); for (i = 0; i < w; i++) X[i][2] = c; }
    if (type === 'broken') for (i = 0; i < w; i++) X[i][2] = gauss(r);      // same distribution, no longer follows its neighbours
    if (type === 'two') for (i = 0; i < w; i++) { X[i][1] += mag; X[i][3] += mag; }
    return { X: X, acted: TYPES[type].acted, type: type };
  }
  function test(u, win) {
    var ks = [], full = [], loo = [], detail = [], rk = { ks: [], full: [], loo: [] }, j, k;
    for (j = 0; j < D; j++) {
      var a = ksP(u.marg[j], win.X.map(function (x) { return x[j]; })); ks.push(a.p); rk.ks.push(a.z);
      var b = mwP(u.full[j].cal, win.X.map(function (x) { return Math.abs(resid(u.full[j], x)); })); full.push(b.p); rk.full.push(b.z);
      var ps = [], best = 0, zmin = Infinity;
      for (k = 0; k < D; k++) { if (k === j) { ps.push(null); continue; } var c = mwP(u.loo[j][k].cal, win.X.map(function (x) { return Math.abs(resid(u.loo[j][k], x)); })); ps.push(c.p); if (c.p > best) best = c.p; if (c.z < zmin) zmin = c.z; }
      detail.push(ps); loo.push(best); rk.loo.push(zmin);
    }
    return { ks: ks, full: full, loo: loo, detail: detail, rank: rk };   // rank: larger = more suspicious
  }
  function score(named, acted) {            // what the operator lives through
    var wasted = 0, repaired = 0, i;
    for (i = 0; i < D; i++) if (named[i]) { if (acted.indexOf(i) >= 0) repaired++; else wasted++; }
    return { wasted: wasted, repaired: repaired, left: acted.length - repaired };
  }
  function many(u, seed, type, mag, w, nwin, alpha, raw) {   // raw = no correction: each p-value against alpha
    var r = rng(seed), keys = ['ks', 'full', 'loo'], out = {}, a;
    keys.forEach(function (k) { out[k] = { waste: 0, power: 0, hit1: 0 }; });
    for (var n = 0; n < nwin; n++) {
      var win = makeWindow(r, type, mag, w), t = test(u, win);
      keys.forEach(function (k) {
        var named = raw ? t[k].map(function (p) { return p <= alpha; }) : holm(t[k], alpha), s = score(named, win.acted);
        if (s.wasted > 0) out[k].waste++;
        if (win.acted.length) { out[k].power += s.repaired / win.acted.length;
          var mx = Math.max.apply(null, t.rank[k]), top = []; for (a = 0; a < D; a++) if (t.rank[k][a] === mx) top.push(a);
          out[k].hit1 += top.filter(function (i) { return win.acted.indexOf(i) >= 0; }).length / top.length; }
      });
    }
    keys.forEach(function (k) { out[k].waste /= nwin; out[k].power = win.acted.length ? out[k].power / nwin : NaN; out[k].hit1 = win.acted.length ? out[k].hit1 / nwin : NaN; });
    return out;
  }
  return { D: D, A: A, rng: rng, gauss: gauss, makeUnit: makeUnit, makeWindow: makeWindow, test: test, holm: holm, score: score, many: many, TYPES: TYPES, resid: resid, Phi: Phi };
})();
if (typeof module !== 'undefined') module.exports = TOY;
