/* ============================================================
   T1 – Inter-Epoch Evolution
   Animated t-SNE scatterplot of Layer-4 activations across
   training epochs. Implements the Visualization Mantra:
     Overview First → all 1 000 points visible on load
     Zoom / Filter  → d3.zoom + legend class filter
     Details on Demand → hover tooltip per point
   ============================================================ */
(function () {

  /* ── Layout constants ─────────────────────────────────────────────────── */
  const M = { top: 20, right: 20, bottom: 30, left: 30 };
  const container = document.getElementById('t1-chart');
  const TW = container.clientWidth || 800;
  const TH = Math.round(TW * 0.44);
  const W = TW - M.left - M.right;
  const H = TH - M.top - M.bottom;
  const R = 3.5;
  const DUR = 900;

  const COLOR = d3.scaleOrdinal(d3.schemeTableau10).domain(d3.range(10));

  /* ── State ────────────────────────────────────────────────────────────── */
  let epochIdx = 0;
  let playing = false;
  let playTimer = null;
  let activeClass = null;
  let data, epochs, points, xScale, yScale, circles, zoom;

  /* ── SVG setup ────────────────────────────────────────────────────────── */
  const svg = d3.select('#t1-chart')
    .append('svg')
    .attr('width', TW)
    .attr('height', TH);

  // Clip path so points don't spill outside plot area on zoom
  svg.append('defs').append('clipPath').attr('id', 't1-clip')
    .append('rect').attr('width', W).attr('height', H);

  const root = svg.append('g')
    .attr('transform', `translate(${M.left},${M.top})`);

  // Background rect for zoom capture
  root.append('rect')
    .attr('width', W).attr('height', H)
    .attr('fill', '#ffffff');

  const plotArea = root.append('g').attr('clip-path', 'url(#t1-clip)');

  // Epoch watermark label
  const epochLabel = root.append('text')
    .attr('class', 'epoch-label')
    .attr('x', W - 10).attr('y', H - 10)
    .attr('text-anchor', 'end');

  /* ── Tooltip helpers ──────────────────────────────────────────────────── */
  const tooltip = d3.select('#tooltip');

  function showTip(event, d, epoch) {
    tooltip
      .style('display', 'block')
      .style('left', (event.clientX + 14) + 'px')
      .style('top', (event.clientY - 28) + 'px')
      .html(`<strong>Digit ${d.label}</strong><br>
             Point&nbsp;ID:&nbsp;${d.id}<br>
             Epoch:&nbsp;${epoch}`);
  }
  function hideTip() { tooltip.style('display', 'none'); }

  /* ── Scales ───────────────────────────────────────────────────────────── */
  function buildScales(pts) {
    const allX = pts.flatMap(p => p.positions.map(pos => pos.x));
    const allY = pts.flatMap(p => p.positions.map(pos => pos.y));
    xScale = d3.scaleLinear().domain(d3.extent(allX)).range([0, W]).nice();
    yScale = d3.scaleLinear().domain(d3.extent(allY)).range([H, 0]).nice();
  }

  /* ── Update to epoch index ────────────────────────────────────────────── */
  function updateEpoch(idx, animate) {
    epochIdx = idx;
    const ep = epochs[idx];

    // Move circles
    (animate ? circles.transition().duration(DUR).ease(d3.easeCubicInOut) : circles)
      .attr('cx', d => xScale(d.positions[idx].x))
      .attr('cy', d => yScale(d.positions[idx].y));

    // Watermark
    epochLabel.text(`Epoch ${ep}`);

    // Highlight active epoch button
    d3.selectAll('#t1-epoch-btns button')
      .classed('active', (_, i) => i === idx);

    // Stats badge
    const s = data.stats[ep];
    if (s) {
      const accTxt = `Acc: ${s.acc}%`;
      const lossTxt = s.loss != null ? `  Loss: ${s.loss}` : '';
      d3.select('#t1-stat').text(accTxt + lossTxt);
    } else {
      d3.select('#t1-stat').text('');
    }
  }

  /* ── Play / pause ─────────────────────────────────────────────────────── */
  function startPlay() {
    playing = true;
    d3.select('#t1-play').text('⏸ Pause');
    playTimer = setInterval(() => {
      const next = (epochIdx + 1) % epochs.length;
      updateEpoch(next, true);
      if (next === epochs.length - 1) stopPlay();
    }, DUR + 400);
  }
  function stopPlay() {
    playing = false;
    clearInterval(playTimer);
    d3.select('#t1-play').text('▶ Play');
  }

  /* ── Class filter ─────────────────────────────────────────────────────── */
  function setClassFilter(cls) {
    activeClass = (activeClass === cls) ? null : cls;

    circles.attr('opacity', d =>
      activeClass === null ? 0.75
        : d.label === activeClass ? 0.95 : 0.07
    ).attr('r', d =>
      activeClass === null ? R
        : d.label === activeClass ? R + 1.5 : R - 1
    );

    d3.selectAll('#t1-legend .legend-item')
      .classed('dimmed', (_, i) =>
        activeClass !== null && i !== activeClass
      );
  }

  /* ── Zoom ─────────────────────────────────────────────────────────────── */
  zoom = d3.zoom()
    .scaleExtent([0.5, 20])
    .on('zoom', ({ transform }) => {
      plotArea.attr('transform', transform);
    });
  svg.call(zoom);

  /* ── Legend ───────────────────────────────────────────────────────────── */
  function buildLegend() {
    const legend = d3.select('#t1-legend');
    d3.range(10).forEach(cls => {
      const item = legend.append('div').attr('class', 'legend-item');
      item.append('span').attr('class', 'legend-dot')
        .style('background', COLOR(cls));
      item.append('span').text(`Digit ${cls}`);
      item.on('click', () => setClassFilter(cls));
    });
  }

  /* ── Epoch buttons ────────────────────────────────────────────────────── */
  function buildEpochButtons() {
    const container = d3.select('#t1-epoch-btns');
    epochs.forEach((ep, i) => {
      container.append('button')
        .text(ep)
        .classed('active', i === 0)
        .on('click', () => { stopPlay(); updateEpoch(i, true); });
    });
  }

  /* ── Wire global controls ─────────────────────────────────────────────── */
  function wireControls() {
    d3.select('#t1-play').on('click', () =>
      playing ? stopPlay() : startPlay()
    );
    d3.select('#t1-prev').on('click', () => {
      stopPlay();
      updateEpoch((epochIdx - 1 + epochs.length) % epochs.length, true);
    });
    d3.select('#t1-next').on('click', () => {
      stopPlay();
      updateEpoch((epochIdx + 1) % epochs.length, true);
    });
  }

  /* ── Initialize ───────────────────────────────────────────────────────── */
  function init(d) {
    data = d;
    epochs = d.epochs;
    points = d.points;

    buildScales(points);

    // Draw circles
    circles = plotArea.selectAll('circle')
      .data(points)
      .join('circle')
      .attr('cx', d => xScale(d.positions[0].x))
      .attr('cy', d => yScale(d.positions[0].y))
      .attr('r', R)
      .attr('fill', d => COLOR(d.label))
      .attr('opacity', 0.75)
      .attr('stroke', '#00000015')
      .attr('stroke-width', 0.5);

    // Hover interactions
    circles
      .on('mouseenter', function (event, d) {
        if (activeClass !== null && d.label !== activeClass) return;
        d3.select(this).raise()
          .transition().duration(120)
          .attr('r', R + 3.5)
          .attr('opacity', 1)
          .attr('stroke', '#1a1a2e')
          .attr('stroke-width', 1.5);
        showTip(event, d, epochs[epochIdx]);
      })
      .on('mousemove', function (event, d) {
        showTip(event, d, epochs[epochIdx]);
      })
      .on('mouseleave', function (_, d) {
        d3.select(this)
          .transition().duration(120)
          .attr('r', activeClass === null ? R : (d.label === activeClass ? R + 1.5 : R - 1))
          .attr('opacity', activeClass === null ? 0.75 : (d.label === activeClass ? 0.95 : 0.07))
          .attr('stroke', '#00000015')
          .attr('stroke-width', 0.5);
        hideTip();
      })
      .on('click', function (_, d) {
        setClassFilter(d.label);
      });

    // Initial epoch label
    epochLabel.text('Epoch 0');
    d3.select('#t1-stat').text(`Acc: ${data.stats['0'].acc}%`);

    buildLegend();
    buildEpochButtons();
    wireControls();
  }

  /* ── Fetch data ───────────────────────────────────────────────────────── */
  d3.json('/epochs').then(init).catch(err => {
    console.error('T1 data load error:', err);
    d3.select('#t1-chart').append('p')
      .style('color', '#e15759')
      .style('padding', '20px')
      .text('⚠ Could not load epoch data. Run data/data.py first.');
  });

})();
