(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var accent3 = style.getPropertyValue('--accent3').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var success = style.getPropertyValue('--success').trim();
  var danger = style.getPropertyValue('--danger').trim();

  // Initialize Mermaid
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'neutral',
      securityLevel: 'loose',
      flowchart: {
        curve: 'basis',
        useMaxWidth: true,
        htmlLabels: true,
      },
    });
  }

  // ========== Chart 1: P0/P1/P2 Completion ==========
  var chart1 = echarts.init(
    document.getElementById('chart-completion'),
    null,
    { renderer: 'svg' }
  );
  chart1.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      appendToBody: true,
    },
    legend: {
      data: ['已达成', '未达成'],
      top: 5,
      textStyle: { color: ink, fontSize: 12 },
    },
    grid: { left: 60, right: 30, top: 50, bottom: 40 },
    xAxis: {
      type: 'category',
      data: ['P0 保底', 'P1 预期', 'P2 理想'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: ink, fontSize: 13, fontWeight: 600 },
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLine: { show: false },
      axisLabel: { color: muted, fontSize: 11, formatter: '{value}%' },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
    },
    series: [
      {
        name: '已达成',
        type: 'bar',
        stack: 'total',
        data: [
          { value: 100, itemStyle: { color: success } },
          { value: 100, itemStyle: { color: accent } },
          { value: 0, itemStyle: { color: accent3 } },
        ],
        label: {
          show: true,
          position: 'inside',
          formatter: function (p) {
            return p.value > 0 ? p.value + '%' : '';
          },
          color: '#fff',
          fontSize: 12,
          fontWeight: 700,
        },
        barWidth: '45%',
      },
      {
        name: '未达成',
        type: 'bar',
        stack: 'total',
        data: [
          { value: 0, itemStyle: { color: bg2 } },
          { value: 0, itemStyle: { color: '#fecaca' } },
          { value: 100, itemStyle: { color: '#fde68a' } },
        ],
        label: {
          show: true,
          position: 'inside',
          formatter: function (p) {
            return p.value > 0 ? p.value + '%' : '';
          },
          color: muted,
          fontSize: 12,
          fontWeight: 600,
        },
      },
    ],
  });
  window.addEventListener('resize', function () { chart1.resize(); });

  // ========== Chart 2: Code Distribution ==========
  var chart2 = echarts.init(
    document.getElementById('chart-code-distribution'),
    null,
    { renderer: 'svg' }
  );
  chart2.setOption({
    animation: false,
    tooltip: {
      trigger: 'item',
      appendToBody: true,
      formatter: '{b}: {c} 行 ({d}%)',
    },
    legend: {
      orient: 'vertical',
      right: 10,
      top: 'center',
      textStyle: { color: ink, fontSize: 12 },
      itemWidth: 12,
      itemHeight: 12,
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['40%', '50%'],
        avoidLabelOverlap: true,
        label: {
          show: true,
          formatter: '{b}\n{c} 行',
          color: ink,
          fontSize: 11,
        },
        labelLine: { show: true, length: 8, length2: 8 },
        data: [
          { value: 384, name: 'Dispatcher', itemStyle: { color: accent } },
          { value: 1708, name: 'Handlers (8 文件)', itemStyle: { color: accent2 } },
          { value: 441, name: 'Skills', itemStyle: { color: accent3 } },
          { value: 1388, name: 'Jobs Tab UI', itemStyle: { color: '#8b5cf6' } },
          { value: 280, name: 'i18n', itemStyle: { color: '#06b6d4' } },
          { value: 559, name: 'app.py', itemStyle: { color: '#f59e0b' } },
          { value: 556, name: 'core/ (schema+security)', itemStyle: { color: '#ef4444' } },
          { value: 5410, name: '测试 (11 文件)', itemStyle: { color: '#10b981' } },
        ],
      },
    ],
  });
  window.addEventListener('resize', function () { chart2.resize(); });

  // ========== Chart 3: Test Distribution ==========
  var chart3 = echarts.init(
    document.getElementById('chart-test-distribution'),
    null,
    { renderer: 'svg' }
  );
  chart3.setOption({
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      appendToBody: true,
      formatter: function (params) {
        var p = params[0];
        return p.name + '<br/>用例数: ' + p.value + '<br/>代码行: ' + p.data.lines + ' 行';
      },
    },
    grid: { left: 120, right: 40, top: 30, bottom: 40 },
    xAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      data: [
        'test_security_sanitizer',
        'test_engines_artifact',
        'test_handlers_framework',
        'test_dispatcher',
        'test_skills',
        'test_predict_diagnose',
        'test_jobs_tab',
        'test_app_integration',
        'test_phase1_handlers',
        'test_val_batch_runtime',
      ],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: ink, fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        data: [
          { value: 6, lines: 200, itemStyle: { color: '#ef4444' } },
          { value: 5, lines: 180, itemStyle: { color: '#ef4444' } },
          { value: 17, lines: 234, itemStyle: { color: '#8b5cf6' } },
          { value: 15, lines: 338, itemStyle: { color: '#8b5cf6' } },
          { value: 15, lines: 446, itemStyle: { color: accent3 } },
          { value: 26, lines: 365, itemStyle: { color: accent2 } },
          { value: 32, lines: 500, itemStyle: { color: accent3 } },
          { value: 39, lines: 700, itemStyle: { color: accent } },
          { value: 41, lines: 567, itemStyle: { color: accent } },
          { value: 44, lines: 881, itemStyle: { color: accent2 } },
        ],
        label: {
          show: true,
          position: 'right',
          color: ink,
          fontSize: 12,
          fontWeight: 600,
          formatter: '{c}',
        },
        barWidth: '55%',
      },
    ],
  });
  window.addEventListener('resize', function () { chart3.resize(); });
})();
