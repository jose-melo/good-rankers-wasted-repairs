// Writes results/toy.json, the toy window of Figure 1: unit seed 3, window seed 11, broken dependence of x3, w = 250,
// Holm at 0.10; plus rates over units 100..259 (window seeds 900..1059, 20 windows each). Run: node paper/toy/toy.js
var TOY = require('./sim.js');
var U = 3, WS = 11, W = 250, ALPHA = 0.10, NU = 160, NW = 20;
var u = TOY.makeUnit(U), win = TOY.makeWindow(TOY.rng(WS), 'broken', 0, W), t = TOY.test(u, win);
var keys = ['ks', 'full', 'loo'], one = {};
keys.forEach(function (k) {
  var named = TOY.holm(t[k], ALPHA), s = TOY.score(named, win.acted);
  one[k] = { p: t[k], named: named, wasted: s.wasted, repaired: s.repaired, V: s.wasted > 0 ? 1 : 0, power: s.repaired / win.acted.length };
});
var rates = {};
keys.forEach(function (k) { rates[k] = { waste: [], power: [] }; });
for (var i = 0; i < NU; i++) {
  var uu = TOY.makeUnit(100 + i), m = TOY.many(uu, 900 + i, 'broken', 0, W, NW, ALPHA, false);
  keys.forEach(function (k) { rates[k].waste.push(m[k].waste); rates[k].power.push(m[k].power); });
}
var out = { meta: { unit_seed: U, window_seed: WS, w: W, alpha: ALPHA, n_units: NU, n_windows: NW, type: 'broken', acted: win.acted,
  rate_units: '100..' + (100 + NU - 1), rate_window_seeds: '900..' + (900 + NU - 1) },
  ref_x2: u.cal.map(function (x) { return x[1]; }), ref_x3: u.cal.map(function (x) { return x[2]; }),
  win_x2: win.X.map(function (x) { return x[1]; }), win_x3: win.X.map(function (x) { return x[2]; }),
  one_window: one, loo_detail: t.detail, rates: rates };
require('fs').writeFileSync(require('path').join(__dirname, '..', '..', 'results', 'toy.json'), JSON.stringify(out));
