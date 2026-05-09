const params = { a: 0, alpha: 1.0, p: 2.0, k: 5.0, beta: 0.5, gamma: 0, xmax: 5 };
const defaults = { ...params };

const presets = {
  sharp: { a: 0,    alpha: 0.5,  p: 3,   k: 12, beta: 0.4,  gamma: 0,   xmax: 5  },
  soft:  { a: -0.5, alpha: 2.0,  p: 1.5, k: 2,  beta: 0.2,  gamma: 0,   xmax: 6  },
  bowl:  { a: 0,    alpha: 0.05, p: 2,   k: 8,  beta: 0.05, gamma: 1,   xmax: 8  },
  cfd:   { a: 0,    alpha: 1.5,  p: 2.5, k: 10, beta: 0.3,  gamma: 0.5, xmax: 10 },
};

function deltaOf(P) {
  const d = 1 - P.a;
  return -P.alpha * Math.exp(-P.k * d) * (P.k + P.p / d) / Math.pow(d, P.p);
}

function fEval(x, P) {
  const dx = x - P.a;
  if (dx <= 0) return NaN;
  return P.alpha * Math.exp(-P.k * dx) / Math.pow(dx, P.p)
       + P.beta * (x - 1) ** 2
       - P.deltaVal * (x - 1)
       + P.gamma;
}

function fpEval(x, P) {
  const dx = x - P.a;
  if (dx <= 0) return NaN;
  return -P.alpha * Math.exp(-P.k * dx) / Math.pow(dx, P.p) * (P.k + P.p / dx)
       + 2 * P.beta * (x - 1)
       - P.deltaVal;
}

function fppEval(x, P) {
  const h = 1e-4;
  return (fpEval(x + h, P) - fpEval(x - h, P)) / (2 * h);
}

// Equidistribution from the Euler–Lagrange first integral √f(x)·x'(ξ) = C.
// Cumulative-trapezoidal of √f on a fine mesh, then invert at uniform ξ_i.
function computeEqGrid(P, N, xmax) {
  const span = xmax - P.a;
  const xL = P.a + Math.max(1e-6, 1e-3 * span);
  const M = 4001;
  const dx = (xmax - xL) / (M - 1);
  let prevSqrt = 0;
  {
    const v = fEval(xL, P);
    prevSqrt = (Number.isFinite(v) && v > 0) ? Math.sqrt(v) : 0;
  }
  const xs = new Float64Array(M);
  const s  = new Float64Array(M);
  xs[0] = xL;
  s[0] = 0;
  for (let i = 1; i < M; i++) {
    const x = xL + i * dx;
    const v = fEval(x, P);
    const sq = (Number.isFinite(v) && v > 0) ? Math.sqrt(v) : 0;
    xs[i] = x;
    s[i] = s[i - 1] + 0.5 * (prevSqrt + sq) * dx;
    prevSqrt = sq;
  }
  const C = s[M - 1];
  const grid = new Array(N);
  if (!(C > 0) || !Number.isFinite(C)) {
    for (let i = 0; i < N; i++) grid[i] = xL + (i / (N - 1)) * (xmax - xL);
    return { grid, C: NaN, xL };
  }
  let j = 0;
  for (let i = 0; i < N; i++) {
    const target = (i / (N - 1)) * C;
    while (j < M - 2 && s[j + 1] < target) j++;
    const ds = s[j + 1] - s[j];
    grid[i] = ds > 0
      ? xs[j] + (target - s[j]) / ds * (xs[j + 1] - xs[j])
      : xs[j];
  }
  grid[0] = xL;
  grid[N - 1] = xmax;
  return { grid, C, xL };
}

let userView = null;
let programmatic = false;
let plotReady = false;
let relayoutTimer = null;

function buildPlot() {
  const P = { ...params, deltaVal: deltaOf(params) };
  const xmax = userView && userView.xRange ? userView.xRange[1] : params.xmax;

  const fmin = fEval(1, P);
  const fpp1 = fppEval(1, P);
  const xnear = P.a + 0.01 * (1 - P.a);
  const fnear = fEval(xnear, P);
  const fmax  = fEval(xmax, P);

  const showGrid = document.getElementById('show-grid').checked;
  const gridN = Math.max(3, Math.min(200, parseInt(document.getElementById('grid-n').value, 10) || 25));
  const gridInfo = computeEqGrid(P, gridN, xmax);

  document.getElementById('info-delta').textContent = P.deltaVal.toExponential(3);
  document.getElementById('info-fmin').textContent  = fmin.toFixed(4);
  const fppEl = document.getElementById('info-fpp');
  fppEl.textContent = fpp1.toFixed(3);
  fppEl.classList.toggle('warn', fpp1 <= 0);
  document.getElementById('info-fnear').textContent = fnear.toExponential(2);
  document.getElementById('info-tail').textContent  = (fmax / (xmax * xmax)).toFixed(4);
  document.getElementById('info-elc').textContent   = Number.isFinite(gridInfo.C) ? gridInfo.C.toFixed(4) : '∞';
  document.getElementById('info-eldx').textContent  = Number.isFinite(gridInfo.C) ? (gridInfo.C / (gridN - 1)).toExponential(2) : '—';

  const N = 1400;
  const xs = [], ys = [], yps = [];
  const x0 = P.a + Math.max(1e-5, (xmax - P.a) * 1e-5);
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const x = P.a + (x0 - P.a) * Math.pow((xmax - P.a) / (x0 - P.a), t);
    xs.push(x);
    ys.push(fEval(x, P));
    yps.push(fpEval(x, P));
  }

  const showF    = document.getElementById('show-f').checked;
  const showFp   = document.getElementById('show-fp').checked;
  const showMin  = document.getElementById('show-min').checked;
  const showAsym = document.getElementById('show-asym').checked;
  const logY     = document.getElementById('log-y').checked;

  const traces = [];
  if (showF) traces.push({
    x: xs, y: ys, mode: 'lines', name: 'f(x)',
    line: { color: '#B8442B', width: 2.5, shape: 'spline' },
    yaxis: 'y',
    hovertemplate: 'x = %{x:.3f}<br>f = %{y:.4g}<extra></extra>',
  });
  if (showFp) traces.push({
    x: xs, y: yps, mode: 'lines', name: "f′(x)",
    line: { color: '#1E4D6B', width: 1.8, dash: 'dot' },
    yaxis: 'y2',
    hovertemplate: 'x = %{x:.3f}<br>f′ = %{y:.4g}<extra></extra>',
  });
  if (showMin) traces.push({
    x: [1], y: [fmin], mode: 'markers', name: 'min',
    marker: { color: '#1A1A1A', size: 9, symbol: 'diamond', line: { color: '#FDFCF8', width: 1.5 } },
    yaxis: 'y',
    hovertemplate: 'minimum<br>x = 1<br>f = %{y:.4g}<extra></extra>',
  });

  const xRangeAuto = [P.a - 0.06 * (xmax - P.a), xmax];
  let yRangeAuto = null;
  if (!logY && showF) {
    const finite = ys.filter(Number.isFinite);
    if (finite.length) {
      const yLow = Math.min(...finite);
      const bulkMax = fmin + Math.max(2, P.beta * Math.pow(xmax - 1, 2)) + 2;
      const yHigh = Math.min(Math.max(...finite), bulkMax * 1.4);
      yRangeAuto = [yLow - 0.05 * (yHigh - yLow), yHigh];
    }
  }

  const xRange = userView && userView.xRange ? userView.xRange : xRangeAuto;
  const yRange = userView && userView.yRange ? userView.yRange : yRangeAuto;

  const shapes = [];
  const annotations = [];
  if (showAsym) {
    shapes.push({
      type: 'line', x0: P.a, x1: P.a, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#9A8E73', width: 1, dash: 'dot' },
    });
    annotations.push({
      x: P.a, y: 1, yref: 'paper', xanchor: 'left', yanchor: 'top',
      text: '  x = a', showarrow: false,
      font: { family: 'Fraunces, serif', size: 13, color: '#9A8E73' },
    });
  }
  shapes.push({
    type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
    line: { color: '#1A1A1A', width: 0.6, dash: 'dot' },
  });
  annotations.push({
    x: 1, y: 1, yref: 'paper', xanchor: 'left', yanchor: 'top',
    text: '  x = 1', showarrow: false,
    font: { family: 'Fraunces, serif', size: 13, color: '#1A1A1A' },
  });

  if (showGrid) {
    for (const xg of gridInfo.grid) {
      shapes.push({
        type: 'line', x0: xg, x1: xg, y0: 0, y1: 0.045, yref: 'paper',
        line: { color: '#1E4D6B', width: 1 },
      });
    }
    annotations.push({
      x: gridInfo.grid[gridInfo.grid.length - 1], y: 0.05, yref: 'paper',
      xanchor: 'right', yanchor: 'bottom',
      text: `<i>x</i>(ξ<sub>i</sub>), N=${gridN}`, showarrow: false,
      font: { family: 'Fraunces, serif', size: 12, color: '#1E4D6B' },
    });
  }

  const axis = {
    gridcolor: '#E5DEC9',
    zerolinecolor: '#3A3A3A', zerolinewidth: 1.2,
    linecolor: '#3A3A3A', linewidth: 1.6,
    tickcolor: '#3A3A3A', tickwidth: 1.2, ticklen: 6,
    showline: true, mirror: true,
  };

  const layout = {
    margin: { t: 24, r: 56, b: 48, l: 56 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#FBF8F1',
    font: { family: 'IBM Plex Sans, sans-serif', size: 12, color: '#1A1A1A' },
    xaxis: {
      ...axis,
      title: { text: '<i>x</i>', font: { family: 'Fraunces, serif', size: 16, color: '#1A1A1A' } },
      range: xRange,
      autorange: false,
    },
    yaxis: {
      ...axis,
      title: { text: '<i>f</i>(<i>x</i>)', font: { family: 'Fraunces, serif', size: 16, color: '#B8442B' } },
      type: logY ? 'log' : 'linear',
      range: !logY && yRange ? yRange : undefined,
      autorange: !(yRange && !logY),
      tickfont: { color: '#B8442B', size: 12 },
      linecolor: '#B8442B',
    },
    yaxis2: {
      ...axis,
      title: { text: "<i>f</i>′(<i>x</i>)", font: { family: 'Fraunces, serif', size: 16, color: '#1E4D6B' } },
      overlaying: 'y', side: 'right',
      gridcolor: 'transparent',
      type: logY ? 'log' : 'linear',
      autorange: true,
      tickfont: { color: '#1E4D6B', size: 12 },
      linecolor: '#1E4D6B',
      mirror: false,
    },
    legend: {
      x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top',
      bgcolor: 'rgba(253,252,248,0.92)', bordercolor: '#E0DAC9', borderwidth: 1,
      font: { size: 12, family: 'Fraunces, serif', color: '#1A1A1A' },
    },
    shapes,
    annotations,
    hoverlabel: {
      bgcolor: '#1A1A1A',
      bordercolor: '#FDFCF8',
      font: { family: 'JetBrains Mono, monospace', size: 12, color: '#FDFCF8' },
    },
  };

  programmatic = true;
  Plotly.react('plot', traces, layout, {
    displaylogo: false, responsive: true, scrollZoom: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  });
  setTimeout(() => { programmatic = false; }, 60);

  if (!plotReady) {
    plotReady = true;
    document.getElementById('plot').on('plotly_relayout', onRelayout);
  }
}

function onRelayout(e) {
  if (programmatic) return;
  if (e['xaxis.autorange'] || e['yaxis.autorange']) {
    userView = null;
    buildPlot();
    return;
  }
  let xChanged = false;
  if (e['xaxis.range[0]'] !== undefined || e['yaxis.range[0]'] !== undefined) {
    if (!userView) userView = {};
    if (e['xaxis.range[0]'] !== undefined) {
      userView.xRange = [e['xaxis.range[0]'], e['xaxis.range[1]']];
      xChanged = true;
    }
    if (e['yaxis.range[0]'] !== undefined) {
      userView.yRange = [e['yaxis.range[0]'], e['yaxis.range[1]']];
    }
  }
  if (xChanged) {
    clearTimeout(relayoutTimer);
    relayoutTimer = setTimeout(buildPlot, 80);
  }
}

function bindRange(id, key, valId, fmt) {
  const slider = document.getElementById(id);
  const input = document.getElementById(valId);
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);

  slider.addEventListener('input', () => {
    params[key] = parseFloat(slider.value);
    input.value = fmt(params[key]);
    buildPlot();
  });

  input.addEventListener('change', () => {
    let v = parseFloat(input.value);
    if (!Number.isFinite(v)) {
      input.value = fmt(params[key]);
      return;
    }
    v = Math.min(max, Math.max(min, v));
    params[key] = v;
    slider.value = v;
    input.value = fmt(v);
    buildPlot();
  });

  input.value = fmt(params[key]);
}

bindRange('param-a',     'a',     'val-a',     v => v.toFixed(3));
bindRange('param-alpha', 'alpha', 'val-alpha', v => v.toFixed(3));
bindRange('param-p',     'p',     'val-p',     v => v.toFixed(2));
bindRange('param-k',     'k',     'val-k',     v => v.toFixed(2));
bindRange('param-beta',  'beta',  'val-beta',  v => v.toFixed(2));
bindRange('param-gamma', 'gamma', 'val-gamma', v => v.toFixed(2));

['show-f', 'show-fp', 'show-min', 'show-asym', 'log-y', 'show-grid'].forEach(id => {
  document.getElementById(id).addEventListener('change', buildPlot);
});

document.getElementById('grid-n').addEventListener('input', buildPlot);
document.getElementById('grid-n').addEventListener('change', buildPlot);

function applyParams(P) {
  Object.assign(params, P);
  document.getElementById('param-a').value     = params.a;
  document.getElementById('param-alpha').value = params.alpha;
  document.getElementById('param-p').value     = params.p;
  document.getElementById('param-k').value     = params.k;
  document.getElementById('param-beta').value  = params.beta;
  document.getElementById('param-gamma').value = params.gamma;
  document.getElementById('val-a').value     = params.a.toFixed(3);
  document.getElementById('val-alpha').value = params.alpha.toFixed(3);
  document.getElementById('val-p').value     = params.p.toFixed(2);
  document.getElementById('val-k').value     = params.k.toFixed(2);
  document.getElementById('val-beta').value  = params.beta.toFixed(2);
  document.getElementById('val-gamma').value = params.gamma.toFixed(2);
  buildPlot();
}

document.getElementById('btn-reset').addEventListener('click', () => {
  userView = null;
  applyParams({ ...defaults });
});

document.querySelectorAll('.preset').forEach(btn => {
  btn.addEventListener('click', () => {
    const name = btn.dataset.preset;
    if (presets[name]) {
      userView = null;
      applyParams({ ...presets[name] });
    }
  });
});

document.getElementById('btn-fit').addEventListener('click', () => {
  userView = null;
  buildPlot();
});

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1600);
}

function pythonCode(P) {
  return `import numpy as np

# Grid spacing control function
# - C^1 smooth on (a, infty)
# - vertical asymptote at x = a
# - quadratic far-field growth ~ beta * x^2
# - local minimum at x = 1 (enforced by delta)

a, alpha, p, k = ${P.a}, ${P.alpha}, ${P.p}, ${P.k}
beta, gamma   = ${P.beta}, ${P.gamma}

# delta is chosen so that f'(1) = 0
d = 1.0 - a
delta = -alpha * np.exp(-k * d) * (k + p / d) / d**p

def f(x):
    dx = x - a
    return (alpha * np.exp(-k * dx) / dx**p
            + beta * (x - 1)**2
            - delta * (x - 1)
            + gamma)

def f_prime(x):
    dx = x - a
    E = np.exp(-k * dx)
    return (-alpha * E / dx**p * (k + p / dx)
            + 2 * beta * (x - 1)
            - delta)
`;
}

function matlabCode(P) {
  return `% Grid spacing control function
%   C^1 smooth on (a, inf)
%   vertical asymptote at x = a
%   quadratic far-field growth ~ beta * x^2
%   local minimum at x = 1 (enforced by delta)

a = ${P.a};  alpha = ${P.alpha};  p = ${P.p};  k = ${P.k};
beta = ${P.beta};  gamma = ${P.gamma};

% delta is chosen so that f'(1) = 0
d = 1 - a;
delta = -alpha * exp(-k * d) * (k + p / d) / d^p;

f       = @(x)  alpha .* exp(-k .* (x - a)) ./ (x - a).^p ...
              + beta  .* (x - 1).^2 ...
              - delta .* (x - 1) ...
              + gamma;

f_prime = @(x) -alpha .* exp(-k .* (x - a)) ./ (x - a).^p .* (k + p ./ (x - a)) ...
              + 2 .* beta .* (x - 1) ...
              - delta;
`;
}

document.getElementById('btn-export-py').addEventListener('click', () => {
  navigator.clipboard.writeText(pythonCode(params))
    .then(() => showToast('python copied to clipboard'));
});

document.getElementById('btn-export-m').addEventListener('click', () => {
  navigator.clipboard.writeText(matlabCode(params))
    .then(() => showToast('matlab copied to clipboard'));
});

buildPlot();
