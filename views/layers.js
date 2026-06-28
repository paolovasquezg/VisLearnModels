(function () {

  /* ── Layout ───────────────────────────────────────────────────────────── */
  const PM = { top: 36, right: 16, bottom: 14, left: 16 };
  const N_LAYERS = 4;

  const t2el = document.getElementById('t2-chart');
  const AVAIL = (t2el.clientWidth || 1100) - PM.left - PM.right;
  // plots take ~62%, gaps take ~38% of available width
  const PW = Math.floor(AVAIL * 0.62 / N_LAYERS);
  const GAP = Math.floor(AVAIL * 0.38 / (N_LAYERS - 1));
  const PH = 300;

  const TOTAL_W = N_LAYERS * PW + (N_LAYERS - 1) * GAP;
  const TOTAL_H = PH + PM.top + PM.bottom;

  const COLOR = d3.scaleOrdinal(d3.schemeTableau10).domain(d3.range(10));

  const DEFAULT_TRAIL_OPACITY = 0.045;
  const HIGHLIGHT_OPACITY = 0.55;
  const DIM_OPACITY = 0.008;

  /* ── State ────────────────────────────────────────────────────────────── */
  let data, points, layerNames;
  let activeClass = null;
  let hoveredPoint = null;
  let xScale, yScale;

  /* ── SVG ──────────────────────────────────────────────────────────────── */
  const svg = d3.select('#t2-chart')
    .append('svg')
    .attr('width', t2el.clientWidth || TOTAL_W + PM.left + PM.right)
    .attr('height', TOTAL_H);

  const root = svg.append('g')
    .attr('transform', `translate(${PM.left},${PM.top})`);

  /* ── Scales (shared across all layer plots) ───────────────────────────── */
  function buildScales(pts) {
    const allX = pts.flatMap(p => p.positions.map(pos => pos.x));
    const allY = pts.flatMap(p => p.positions.map(pos => pos.y));
    xScale = d3.scaleLinear().domain(d3.extent(allX)).range([0, PW]).nice();
    yScale = d3.scaleLinear().domain(d3.extent(allY)).range([PH, 0]).nice();
  }

  /* ── Helpers: SVG coords for a point in a given layer plot ───────────── */
  function plotX(layerIdx) { return layerIdx * (PW + GAP); }

  function svgX(layerIdx, dataX) { return plotX(layerIdx) + xScale(dataX); }
  function svgY(dataY) { return yScale(dataY); }

  /* ── Compute class centroids per layer ────────────────────────────────── */
  function computeCentroids(pts) {
    // centroids[layerIdx][classLabel] = {x, y}
    const sums = Array.from({ length: N_LAYERS }, () =>
      Object.fromEntries(d3.range(10).map(c => [c, { sx: 0, sy: 0, n: 0 }]))
    );
    pts.forEach(p =>
      p.positions.forEach(pos => {
        const s = sums[pos.layer][p.label];
        s.sx += pos.x; s.sy += pos.y; s.n++;
      })
    );
    return sums.map(layer =>
      Object.fromEntries(
        d3.range(10).map(c => [c, {
          x: sums[0][c].n ? layer[c].sx / layer[c].n : 0,
          y: sums[0][c].n ? layer[c].sy / layer[c].n : 0
        }])
      )
    );
  }

  /* ── Build trail path string for one point between two adjacent layers ── */
  function trailPath(pt, li, centroids, bundleAlpha) {
    const p0 = pt.positions[li];
    const p1 = pt.positions[li + 1];

    // Source and dest in SVG space
    const x0 = svgX(li, p0.x), y0 = svgY(p0.y);
    const x1 = svgX(li + 1, p1.x), y1 = svgY(p1.y);

    // Centroid positions in SVG space (for bundling attraction)
    const cx0 = svgX(li, centroids[li][pt.label].x);
    const cy0 = svgY(centroids[li][pt.label].y);
    const cx1 = svgX(li + 1, centroids[li + 1][pt.label].x);
    const cy1 = svgY(centroids[li + 1][pt.label].y);

    // Control points: pulled toward class centroid at each side of the gap
    const cpx0 = x0 + (cx0 - x0) * bundleAlpha + (x1 - x0) * 0.33;
    const cpy0 = y0 + (cy0 - y0) * bundleAlpha;
    const cpx1 = x1 + (cx1 - x1) * bundleAlpha - (x1 - x0) * 0.33;
    const cpy1 = y1 + (cy1 - y1) * bundleAlpha;

    return `M${x0},${y0} C${cpx0},${cpy0} ${cpx1},${cpy1} ${x1},${y1}`;
  }

  /* ── Draw trails (all layer-to-layer connections) ─────────────────────── */
  let trailGroups = [];  // one <g> per gap (3 total)

  function drawTrails(pts, centroids) {
    trailGroups = [];
    for (let li = 0; li < N_LAYERS - 1; li++) {
      const g = root.append('g').attr('class', `trails trails-gap-${li}`);
      trailGroups.push(g);

      g.selectAll('path')
        .data(pts)
        .join('path')
        .attr('class', d => `trail trail-c${d.label} trail-id${d.id}`)
        .attr('d', d => trailPath(d, li, centroids, 0.45))
        .attr('fill', 'none')
        .attr('stroke', d => COLOR(d.label))
        .attr('stroke-width', 1)
        .attr('opacity', DEFAULT_TRAIL_OPACITY);
    }
  }

  /* ── Draw mini scatterplots ───────────────────────────────────────────── */
  let dotGroups = [];   // one <g> per layer

  function drawPlots(pts) {
    layerNames.forEach((name, li) => {
      const ox = plotX(li);

      // Plot background
      root.append('rect')
        .attr('x', ox).attr('y', 0)
        .attr('width', PW).attr('height', PH)
        .attr('fill', '#ffffff')
        .attr('rx', 6);

      // Layer label
      root.append('text')
        .attr('class', 'layer-label')
        .attr('x', ox + PW / 2).attr('y', -10)
        .attr('text-anchor', 'middle')
        .text(name);

      // Dots
      const g = root.append('g').attr('class', `dots layer-${li}`);
      dotGroups.push(g);

      g.selectAll('circle')
        .data(pts)
        .join('circle')
        .attr('class', d => `dot dot-c${d.label} dot-id${d.id}`)
        .attr('cx', d => ox + xScale(d.positions[li].x))
        .attr('cy', d => yScale(d.positions[li].y))
        .attr('r', 2.8)
        .attr('fill', d => COLOR(d.label))
        .attr('opacity', 0.8)
        .attr('stroke', '#00000012')
        .attr('stroke-width', 0.4);
    });
  }

  /* ── Hover interactions ───────────────────────────────────────────────── */
  const tooltip = d3.select('#tooltip');

  function highlightPoint(pt) {
    hoveredPoint = pt.id;

    // Raise & enlarge dots for this point across all layers
    dotGroups.forEach((g, li) => {
      g.selectAll(`.dot-id${pt.id}`)
        .raise()
        .transition().duration(100)
        .attr('r', 5.5)
        .attr('opacity', 1)
        .attr('stroke', '#1a1a2e')
        .attr('stroke-width', 1.5);
    });

    // Highlight this point's trails
    trailGroups.forEach(g => {
      g.selectAll(`.trail-id${pt.id}`)
        .raise()
        .transition().duration(100)
        .attr('stroke-width', 2.5)
        .attr('opacity', 0.9);
    });
  }

  function unhighlightPoint(pt) {
    if (hoveredPoint !== pt.id) return;
    hoveredPoint = null;

    dotGroups.forEach(g => {
      g.selectAll(`.dot-id${pt.id}`)
        .transition().duration(100)
        .attr('r', 2.8)
        .attr('opacity', activeClass === null ? 0.8 : (pt.label === activeClass ? 0.9 : 0.08))
        .attr('stroke', '#00000012')
        .attr('stroke-width', 0.4);
    });

    trailGroups.forEach(g => {
      g.selectAll(`.trail-id${pt.id}`)
        .transition().duration(100)
        .attr('stroke-width', 1)
        .attr('opacity', activeClass === null ? DEFAULT_TRAIL_OPACITY
          : (pt.label === activeClass ? HIGHLIGHT_OPACITY : DIM_OPACITY));
    });
  }

  function wirePointInteractions(pts) {
    dotGroups.forEach((g, li) => {
      g.selectAll('circle')
        .on('mouseenter', function (event, d) {
          highlightPoint(d);
          tooltip
            .style('display', 'block')
            .style('left', (event.clientX + 14) + 'px')
            .style('top', (event.clientY - 28) + 'px')
            .html(`<strong>${d.label}</strong><br>`);
        })
        .on('mousemove', function (event) {
          tooltip
            .style('left', (event.clientX + 14) + 'px')
            .style('top', (event.clientY - 28) + 'px');
        })
        .on('mouseleave', function (_, d) {
          unhighlightPoint(d);
          tooltip.style('display', 'none');
        })
        .on('click', function (_, d) {
          setClassFilter(d.label);
        });
    });
  }

  /* ── Class filter ─────────────────────────────────────────────────────── */
  function setClassFilter(cls) {
    activeClass = (activeClass === cls) ? null : cls;

    // Dots
    dotGroups.forEach(g => {
      g.selectAll('circle')
        .attr('opacity', d =>
          activeClass === null ? 0.8
            : d.label === activeClass ? 0.9 : 0.08
        )
        .attr('r', d =>
          activeClass === null ? 2.8
            : d.label === activeClass ? 3.5 : 2
        );
    });

    // Trails
    trailGroups.forEach(g => {
      g.selectAll('path')
        .attr('opacity', d =>
          activeClass === null ? DEFAULT_TRAIL_OPACITY
            : d.label === activeClass ? HIGHLIGHT_OPACITY : DIM_OPACITY
        )
        .attr('stroke-width', d =>
          activeClass !== null && d.label === activeClass ? 1.4 : 1
        );
    });

    // Legend
    d3.selectAll('#t2-legend .legend-item')
      .classed('dimmed', (_, i) => activeClass !== null && i !== activeClass);

    // Class select sync
    d3.select('#t2-class-select').property('value', activeClass === null ? -1 : activeClass);
  }

  /* ── Trail opacity control ────────────────────────────────────────────── */
  function setTrailOpacity(op) {
    trailGroups.forEach(g => {
      g.selectAll('path')
        .attr('opacity', d =>
          op === 0 ? 0
            : activeClass === null ? op
              : d.label === activeClass ? HIGHLIGHT_OPACITY : DIM_OPACITY
        );
    });
  }

  /* ── Animate trails (reveal left → right) ─────────────────────────────── */
  function animateTrails() {
    // Reset trails to invisible
    trailGroups.forEach(g =>
      g.selectAll('path')
        .attr('stroke-dasharray', function () { return this.getTotalLength() + ' ' + this.getTotalLength(); })
        .attr('stroke-dashoffset', function () { return this.getTotalLength(); })
        .attr('opacity', DEFAULT_TRAIL_OPACITY)
    );

    const delay = 600;
    trailGroups.forEach((g, li) => {
      g.selectAll('path')
        .transition()
        .delay(li * delay)
        .duration(900)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0)
        .on('end', function () {
          d3.select(this).attr('stroke-dasharray', null).attr('stroke-dashoffset', null);
        });
    });
  }

  /* ── Legend ───────────────────────────────────────────────────────────── */
  function buildLegend(pts) {
    const legend = d3.select('#t2-legend');
    d3.range(10).forEach(cls => {
      const item = legend.append('div').attr('class', 'legend-item');
      item.append('span').attr('class', 'legend-dot').style('background', COLOR(cls));
      item.append('span').text(`${cls}`);
      item.on('click', () => setClassFilter(cls));
    });

    // Populate class select
    const sel = d3.select('#t2-class-select');
    d3.range(10).forEach(cls => {
      sel.append('option').attr('value', cls).text(`${cls}`);
    });
    sel.on('change', function () {
      const v = +this.value;
      if (v === -1) { activeClass = null; setClassFilter(null); }
      else setClassFilter(v);
    });
  }

  /* ── Wire controls ────────────────────────────────────────────────────── */
  function wireControls() {
    d3.select('#t2-animate').on('click', animateTrails);

    d3.select('#t2-reset').on('click', () => {
      activeClass = null;
      setClassFilter(null);
      setTrailOpacity(DEFAULT_TRAIL_OPACITY);
    });

    d3.selectAll('.trail-btn').on('click', function () {
      d3.selectAll('.trail-btn').classed('active', false);
      d3.select(this).classed('active', true);
      setTrailOpacity(+this.dataset.opacity);
    });
  }

  /* ── Initialize ───────────────────────────────────────────────────────── */
  function init(d) {
    data = d;
    points = d.points;
    layerNames = d.layers;

    buildScales(points);
    const centroids = computeCentroids(points);

    drawTrails(points, centroids);   // trails behind dots
    drawPlots(points);               // dots in front
    wirePointInteractions(points);
    buildLegend(points);
    wireControls();
  }

  /* ── Fetch data ───────────────────────────────────────────────────────── */
  d3.json('/layers').then(init).catch(err => {
    console.error('T2 data load error:', err);
    d3.select('#t2-chart').append('p')
      .style('color', '#e15759')
      .style('padding', '20px')
      .text('⚠ Could not load layer data. Run data/data.py first.');
  });

})();
