import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import streamlit.components.v1 as components


# ─── REALISTIC 2D CANVAS FLIGHT FEED ─────────────────────────────────────────

def render_flight_feed(anomaly_type: str, n_steps: int):
    """
    Renders a realistic Canvas 2D flight simulation.
    Pure HTML5 Canvas — no WebGL, no external libraries.
    Parallax sky, terrain, clouds, detailed airplane drawing,
    and anomaly-specific particle effects all in 2D perspective.
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:100%; height:430px; background:#0a0f1e; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:430px; }}
  #c {{ display:block; width:100%; height:380px; }}
  #hud-top {{
    position:absolute; top:8px; left:8px;
    background:rgba(0,8,20,0.82);
    border:1px solid rgba(0,220,120,0.35);
    border-radius:6px; padding:8px 12px;
    color:#00ee88; font-family:'Courier New',monospace;
    font-size:11px; line-height:1.85; min-width:175px;
    pointer-events:none;
  }}
  #hud-top .v {{ color:#ffffff; font-weight:bold; float:right; margin-left:20px; }}
  #hud-top .warn {{ color:#ff4444; font-weight:bold; }}
  #hud-right {{
    position:absolute; top:8px; right:8px;
    background:rgba(0,8,20,0.82);
    border:1px solid rgba(0,220,120,0.35);
    border-radius:6px; padding:8px 12px;
    color:#00ee88; font-family:'Courier New',monospace;
    font-size:11px; text-align:right;
    pointer-events:none;
  }}
  #anom-warn {{
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    background:rgba(180,0,0,0.88);
    border:2px solid #ff4444;
    border-radius:8px; padding:7px 22px;
    color:#fff; font-family:'Courier New',monospace;
    font-size:14px; letter-spacing:2px;
    display:none; pointer-events:none;
  }}
  #pbar-wrap {{
    position:absolute; bottom:0; left:0; right:0;
    height:6px; background:rgba(255,255,255,0.08);
  }}
  #pbar-fill {{
    height:100%; width:0%;
    background:linear-gradient(90deg,#00ee88,#00aaff);
    transition:width 0.15s linear;
  }}
  @keyframes blink {{ 50%{{opacity:0;}} }}
  .blink {{ animation:blink 0.5s step-end infinite; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>

  <div id="hud-top">
    <div>ALT &nbsp;<span class="v" id="h-alt">30,000 ft</span></div>
    <div>SPD &nbsp;<span class="v" id="h-spd">250 kts</span></div>
    <div>RPM &nbsp;<span class="v" id="h-rpm">2500</span></div>
    <div>HYD &nbsp;<span class="v" id="h-hyd">3000 psi</span></div>
    <div>FUEL <span class="v" id="h-fuel">800 pph</span></div>
    <div>VIB &nbsp;<span class="v" id="h-vib">0.50 mm/s</span></div>
  </div>

  <div id="hud-right">
    <div style="font-size:9px;color:#555;margin-bottom:2px;">FLIGHT STATUS</div>
    <div id="st-txt" style="font-weight:bold;font-size:13px;">NORMAL</div>
    <div id="st-step" style="font-size:9px;color:#444;margin-top:3px;">Step 0 / {n_steps}</div>
  </div>

  <div id="anom-warn" class="blink">&#9888;&nbsp; {anomaly_type.upper()} DETECTED &nbsp;&#9888;</div>

  <div id="pbar-wrap"><div id="pbar-fill"></div></div>
</div>

<script>
(function(){{
  // ── Config ────────────────────────────────────────────────────────────────
  const ANOMALY    = "{anomaly_type}";
  const TOTAL      = {n_steps};
  const ANOM_START = TOTAL - 4;

  const CFG = {{
    SensorBias:        {{ shake:0,   roll:0,    pitch:0,    fire:false, smoke:false, sparks:false, debris:false, pressWave:false, fogAlpha:0,   skyH:210 }},
    HydraulicLeak:     {{ shake:2,   roll:0.04, pitch:0.01, fire:false, smoke:true,  sparks:false, debris:false, pressWave:false, fogAlpha:0.18,skyH:200 }},
    EngineFailure:     {{ shake:6,   roll:0.07, pitch:0.08, fire:true,  smoke:true,  sparks:true,  debris:false, pressWave:false, fogAlpha:0.30,skyH:190 }},
    CabinPressureLoss: {{ shake:3,   roll:0.02, pitch:0.03, fire:false, smoke:false, sparks:false, debris:false, pressWave:true,  fogAlpha:0.22,skyH:215 }},
    BirdStrike:        {{ shake:12,  roll:0.12, pitch:0.14, fire:true,  smoke:true,  sparks:true,  debris:true,  pressWave:false, fogAlpha:0.35,skyH:185 }},
    FuelLeak:          {{ shake:2,   roll:0.02, pitch:0.01, fire:false, smoke:true,  sparks:false, debris:false, pressWave:false, fogAlpha:0.20,skyH:200 }},
    ElectricalFault:   {{ shake:4,   roll:0.03, pitch:0,    fire:false, smoke:false, sparks:true,  debris:false, pressWave:false, fogAlpha:0.25,skyH:195 }},
  }};
  const C = CFG[ANOMALY] || CFG.SensorBias;

  // ── Canvas setup ──────────────────────────────────────────────────────────
  const canvas = document.getElementById('c');
  const ctx    = canvas.getContext('2d');
  let W, H;

  function resize() {{
    W = canvas.offsetWidth;
    H = canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;
  }}
  resize();
  window.addEventListener('resize', resize);

  // ── Utility ───────────────────────────────────────────────────────────────
  const rand  = (lo,hi) => lo + Math.random()*(hi-lo);
  const lerp  = (a,b,t) => a+(b-a)*t;
  const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));

  // ── Stars ─────────────────────────────────────────────────────────────────
  const STARS = Array.from({{length:120}}, ()=>([rand(0,1),rand(0,0.45),rand(0.3,1)]));

  // ── Terrain layers (parallax) ─────────────────────────────────────────────
  // Each layer: {{ speed, color, yBase, amplitude, phase }}
  const TERRAIN_LAYERS = [
    {{ speed:0.12, color:'#1a3a50', yBase:0.72, amp:0.06 }},
    {{ speed:0.22, color:'#1e4a28', yBase:0.76, amp:0.05 }},
    {{ speed:0.40, color:'#254d2a', yBase:0.80, amp:0.045 }},
    {{ speed:0.70, color:'#2d5c30', yBase:0.84, amp:0.040 }},
    {{ speed:1.20, color:'#346635', yBase:0.88, amp:0.030 }},
  ];
  const terrainOffsets = TERRAIN_LAYERS.map(()=>0);

  function terrainY(layer, xFrac) {{
    const L = TERRAIN_LAYERS[layer];
    const x = xFrac*8 + terrainOffsets[layer]*0.003;
    return (L.yBase + Math.sin(x*1.1)*L.amp + Math.sin(x*2.3+1)*L.amp*0.5
                    + Math.sin(x*4.7+2)*L.amp*0.25) * H;
  }}

  function drawTerrain() {{
    for (let li=0; li<TERRAIN_LAYERS.length; li++) {{
      const L = TERRAIN_LAYERS[li];
      ctx.beginPath();
      ctx.moveTo(0, H);
      const steps = 60;
      for (let s=0; s<=steps; s++) {{
        ctx.lineTo(s/steps*W, terrainY(li, s/steps));
      }}
      ctx.lineTo(W, H);
      ctx.closePath();
      ctx.fillStyle = L.color;
      ctx.fill();
    }}
  }}

  // ── Clouds ────────────────────────────────────────────────────────────────
  function makeCloud() {{
    return {{
      x: rand(0,1), y: rand(0.08,0.42),
      scale: rand(0.6,1.5),
      speed: rand(0.00008,0.00025),
      alpha: rand(0.55,0.90),
    }};
  }}
  const CLOUDS = Array.from({{length:22}}, makeCloud);

  function drawCloud(cl) {{
    const cx = cl.x * W, cy = cl.y * H, s = cl.scale;
    ctx.save();
    ctx.globalAlpha = cl.alpha;
    const blobs = [
      [0,0,55*s], [50*s,-14*s,42*s], [-50*s,-10*s,38*s],
      [90*s,4*s,32*s],  [-88*s,4*s,30*s], [25*s,-30*s,30*s],
    ];
    blobs.forEach(([bx,by,br]) => {{
      const grd = ctx.createRadialGradient(cx+bx,cy+by,0,cx+bx,cy+by,br);
      grd.addColorStop(0,'rgba(255,255,255,0.95)');
      grd.addColorStop(0.5,'rgba(220,235,255,0.75)');
      grd.addColorStop(1,'rgba(180,200,240,0)');
      ctx.beginPath();
      ctx.arc(cx+bx, cy+by, br, 0, Math.PI*2);
      ctx.fillStyle = grd;
      ctx.fill();
    }});
    ctx.restore();
  }}

  // ── Sky gradient ──────────────────────────────────────────────────────────
  function drawSky(anomActive, t) {{
    const horizonY = H * 0.65;
    let topColor, midColor, horizColor;
    if (anomActive) {{
      const p = clamp((t-ANOM_START)/4, 0, 1);
      if (ANOMALY==='EngineFailure'||ANOMALY==='BirdStrike') {{
        topColor  = `rgba(${{Math.round(lerp(10,40,p))}},10,20,1)`;
        midColor  = `rgba(${{Math.round(lerp(20,80,p))}},30,50,1)`;
        horizColor= `rgba(${{Math.round(lerp(80,160,p))}},90,80,1)`;
      }} else if (ANOMALY==='ElectricalFault') {{
        topColor  = 'rgba(5,5,25,1)';
        midColor  = 'rgba(20,10,50,1)';
        horizColor= 'rgba(60,30,100,1)';
      }} else if (ANOMALY==='CabinPressureLoss') {{
        topColor  = 'rgba(0,5,30,1)';
        midColor  = 'rgba(5,20,60,1)';
        horizColor= 'rgba(40,100,180,1)';
      }} else {{
        topColor  = 'rgba(8,18,40,1)';
        midColor  = 'rgba(12,30,60,1)';
        horizColor= 'rgba(50,100,140,1)';
      }}
    }} else {{
      topColor  = 'rgba(8,20,65,1)';
      midColor  = 'rgba(20,60,130,1)';
      horizColor= 'rgba(80,150,210,1)';
    }}
    const grd = ctx.createLinearGradient(0,0,0,horizonY);
    grd.addColorStop(0, topColor);
    grd.addColorStop(0.55, midColor);
    grd.addColorStop(1, horizColor);
    ctx.fillStyle = grd;
    ctx.fillRect(0,0,W,H);

    // Subtle sun / glow near horizon
    if (!anomActive) {{
      const sunX = W*0.72, sunY = H*0.18;
      const sg = ctx.createRadialGradient(sunX,sunY,0,sunX,sunY,W*0.28);
      sg.addColorStop(0,'rgba(255,240,180,0.22)');
      sg.addColorStop(1,'rgba(255,220,100,0)');
      ctx.fillStyle = sg;
      ctx.fillRect(0,0,W,H);
    }}
  }}

  // ── Stars ─────────────────────────────────────────────────────────────────
  function drawStars(anomActive) {{
    STARS.forEach(([sx,sy,br]) => {{
      const alpha = anomActive ? br*0.8 : br*0.35;
      ctx.beginPath();
      ctx.arc(sx*W, sy*H*0.65, 0.9, 0, Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${{alpha.toFixed(2)}})`;
      ctx.fill();
    }});
  }}

  // ── Horizon haze ─────────────────────────────────────────────────────────
  function drawHaze(anomActive) {{
    const hY = H*0.65;
    const hg = ctx.createLinearGradient(0,hY-30,0,hY+60);
    if (anomActive && (ANOMALY==='EngineFailure'||ANOMALY==='BirdStrike')) {{
      hg.addColorStop(0,'rgba(180,80,30,0)');
      hg.addColorStop(0.5,'rgba(140,50,20,0.35)');
      hg.addColorStop(1,'rgba(80,20,10,0.5)');
    }} else {{
      hg.addColorStop(0,'rgba(80,140,210,0)');
      hg.addColorStop(0.5,'rgba(100,170,220,0.3)');
      hg.addColorStop(1,'rgba(120,190,230,0.5)');
    }}
    ctx.fillStyle = hg;
    ctx.fillRect(0, hY-30, W, 90);
  }}

  // ── Airplane drawing ──────────────────────────────────────────────────────
  // Perspective 3/4 view — drawn with Canvas 2D paths
  function drawAirplane(cx, cy, roll, pitch, anomActive, t) {{
    ctx.save();
    ctx.translate(cx, cy);

    // Atmospheric perspective scale (simulate altitude)
    const sc = 1.0 + Math.sin(t*0.3)*0.015;
    ctx.scale(sc, sc);
    ctx.rotate(roll);

    // Shadow on ground (subtle)
    ctx.save();
    ctx.translate(20, 180);
    ctx.scale(1, 0.18);
    ctx.beginPath();
    ctx.ellipse(0, 0, 120, 30, 0, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(0,0,0,0.18)';
    ctx.fill();
    ctx.restore();

    // ── Fuselage ─────────────────────────────────────────────────────────
    // Main body — tapered ellipse shape
    const fuseGrad = ctx.createLinearGradient(-120,0,120,0);
    fuseGrad.addColorStop(0,   '#c8d8ee');
    fuseGrad.addColorStop(0.35,'#e8f0fa');
    fuseGrad.addColorStop(0.65,'#d8e8f8');
    fuseGrad.addColorStop(1,   '#a0b8d0');

    ctx.beginPath();
    ctx.moveTo(130, 0);
    // top profile
    ctx.bezierCurveTo(120,-8, 60,-14, 0,-14);
    ctx.bezierCurveTo(-60,-14,-110,-12,-130,-8);
    // tail
    ctx.lineTo(-130,0);
    ctx.lineTo(-130,8);
    // bottom profile
    ctx.bezierCurveTo(-110,12,-60,14,0,14);
    ctx.bezierCurveTo(60,14,120,8,130,0);
    ctx.closePath();
    ctx.fillStyle = fuseGrad;
    ctx.fill();
    ctx.strokeStyle = '#8aa8c8';
    ctx.lineWidth = 0.8;
    ctx.stroke();

    // Nose cone
    ctx.beginPath();
    ctx.moveTo(130, 0);
    ctx.bezierCurveTo(145,-3,152,0,145,4);
    ctx.bezierCurveTo(140,6,130,8,130,0);
    ctx.fillStyle = '#d0e0f0';
    ctx.fill();

    // Airline livery stripe
    ctx.beginPath();
    ctx.moveTo(80,-13); ctx.lineTo(-125,-7);
    ctx.lineTo(-125,-1); ctx.lineTo(80,-5);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,60,180,0.55)';
    ctx.fill();

    // Cockpit windows
    const cockGrad = ctx.createLinearGradient(90,-12,110,-4);
    cockGrad.addColorStop(0,'rgba(100,200,255,0.85)');
    cockGrad.addColorStop(1,'rgba(40,100,200,0.65)');
    ctx.beginPath();
    ctx.moveTo(100,-10); ctx.lineTo(125,-3); ctx.lineTo(122,3); ctx.lineTo(98,5);
    ctx.closePath();
    ctx.fillStyle = cockGrad;
    ctx.fill();
    ctx.strokeStyle = 'rgba(180,220,255,0.6)';
    ctx.lineWidth = 0.6;
    ctx.stroke();

    // Window row
    for (let i=0; i<8; i++) {{
      const wx = 70 - i*18;
      if (wx < -100) break;
      ctx.beginPath();
      ctx.ellipse(wx, -10, 4.5, 5, 0, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(140,210,255,0.7)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(200,230,255,0.5)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }}

    // ── Wings (perspective foreshortened) ─────────────────────────────────
    // Right wing (near — fuller)
    const wGrad = ctx.createLinearGradient(0,0,0,90);
    wGrad.addColorStop(0,'#c0d4ea');
    wGrad.addColorStop(0.6,'#a8c0e0');
    wGrad.addColorStop(1,'#8aaccc');

    ctx.beginPath();
    ctx.moveTo(20,10);
    ctx.bezierCurveTo(0,12,-30,18,-50,22);
    ctx.bezierCurveTo(-70,28,-90,40,-100,85);
    ctx.lineTo(-85,88);
    ctx.bezierCurveTo(-75,50,-55,34,-35,28);
    ctx.bezierCurveTo(-15,22,10,16,25,14);
    ctx.closePath();
    ctx.fillStyle = wGrad;
    ctx.fill();
    ctx.strokeStyle = '#7090b8';
    ctx.lineWidth = 0.8;
    ctx.stroke();

    // Winglet right
    ctx.beginPath();
    ctx.moveTo(-100,85); ctx.lineTo(-108,78); ctx.lineTo(-104,90); ctx.lineTo(-96,90);
    ctx.closePath();
    ctx.fillStyle = '#a0b8d0';
    ctx.fill();

    // Left wing (far — foreshortened/darker)
    const wGrad2 = ctx.createLinearGradient(0,0,0,-55);
    wGrad2.addColorStop(0,'#a8bcdc');
    wGrad2.addColorStop(1,'#7898bc');

    ctx.beginPath();
    ctx.moveTo(20,-10);
    ctx.bezierCurveTo(0,-12,-30,-16,-50,-20);
    ctx.bezierCurveTo(-70,-24,-88,-32,-96,-55);
    ctx.lineTo(-84,-56);
    ctx.bezierCurveTo(-76,-36,-58,-28,-38,-24);
    ctx.bezierCurveTo(-18,-20,8,-14,24,-12);
    ctx.closePath();
    ctx.fillStyle = wGrad2;
    ctx.fill();
    ctx.strokeStyle = '#607890';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    // Winglet left (far)
    ctx.beginPath();
    ctx.moveTo(-96,-55); ctx.lineTo(-104,-50); ctx.lineTo(-100,-58); ctx.lineTo(-92,-58);
    ctx.closePath();
    ctx.fillStyle = '#8098b0';
    ctx.fill();

    // ── Engines (4) ───────────────────────────────────────────────────────
    // Right wing engines
    const engDefs = [
      [[-38,38],false],  // right inboard
      [[-72,62],false],  // right outboard
      [[-40,-28],true],  // left inboard
      [[-70,-46],true],  // left outboard
    ];
    engDefs.forEach(([[ex,ey],isFar],ei) => {{
      const sc2 = isFar ? 0.72 : 1.0;
      const ec  = isFar ? '#556677' : '#667788';
      const ec2 = isFar ? '#445566' : '#556677';

      ctx.save();
      ctx.translate(ex, ey);
      ctx.scale(sc2, sc2);

      // Nacelle body
      const ng = ctx.createLinearGradient(-22,0,22,0);
      ng.addColorStop(0, ec2);
      ng.addColorStop(0.4,'#8899aa');
      ng.addColorStop(1, ec);
      ctx.beginPath();
      ctx.ellipse(0,0,22,9,0,0,Math.PI*2);
      ctx.fillStyle = ng;
      ctx.fill();
      ctx.strokeStyle = '#445566';
      ctx.lineWidth = 0.7;
      ctx.stroke();

      // Intake ring
      ctx.beginPath();
      ctx.ellipse(20,0,10,9,0,0,Math.PI*2);
      ctx.fillStyle = '#1a2a3a';
      ctx.fill();
      ctx.strokeStyle = '#556677';
      ctx.lineWidth = 0.6;
      ctx.stroke();

      // Fan blades (simplified)
      ctx.beginPath();
      ctx.ellipse(20,0,6,6,0,0,Math.PI*2);
      const fg = ctx.createRadialGradient(20,0,0,20,0,6);
      fg.addColorStop(0,'#667788');
      fg.addColorStop(1,'#1a2a3a');
      ctx.fillStyle = fg;
      ctx.fill();

      // Exhaust glow (anomaly)
      if (anomActive && (C.fire||(ei<2)) && ANOMALY==='EngineFailure') {{
        const eg = ctx.createRadialGradient(-22,0,0,-22,0,16);
        eg.addColorStop(0,'rgba(255,120,0,0.9)');
        eg.addColorStop(0.5,'rgba(255,60,0,0.5)');
        eg.addColorStop(1,'rgba(200,0,0,0)');
        ctx.beginPath();
        ctx.ellipse(-22,0,16,8,0,0,Math.PI*2);
        ctx.fillStyle = eg;
        ctx.fill();
      }}

      ctx.restore();
    }});

    // ── Tail section ──────────────────────────────────────────────────────
    // Horizontal stabiliser right
    ctx.beginPath();
    ctx.moveTo(-115,6);
    ctx.bezierCurveTo(-118,8,-128,18,-132,35);
    ctx.lineTo(-124,36);
    ctx.bezierCurveTo(-120,22,-112,12,-112,8);
    ctx.closePath();
    ctx.fillStyle = '#b0c4dc';
    ctx.fill();
    ctx.strokeStyle = '#7090b0';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    // Horizontal stabiliser left
    ctx.beginPath();
    ctx.moveTo(-115,-6);
    ctx.bezierCurveTo(-118,-8,-128,-14,-130,-22);
    ctx.lineTo(-123,-22);
    ctx.bezierCurveTo(-120,-16,-112,-10,-112,-8);
    ctx.closePath();
    ctx.fillStyle = '#9ab0c8';
    ctx.fill();
    ctx.strokeStyle = '#607090';
    ctx.lineWidth = 0.6;
    ctx.stroke();

    // Vertical fin
    ctx.beginPath();
    ctx.moveTo(-112,0);
    ctx.bezierCurveTo(-115,-5,-122,-25,-118,-42);
    ctx.bezierCurveTo(-116,-50,-112,-48,-110,-38);
    ctx.bezierCurveTo(-108,-25,-110,-10,-110,0);
    ctx.closePath();
    ctx.fillStyle = '#b0c4dc';
    ctx.fill();
    ctx.strokeStyle = '#7090b0';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    // Tail livery
    ctx.beginPath();
    ctx.moveTo(-112,0);
    ctx.bezierCurveTo(-114,-4,-120,-20,-116,-36);
    ctx.lineTo(-114,-35);
    ctx.bezierCurveTo(-112,-22,-110,-8,-110,0);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,60,180,0.5)';
    ctx.fill();

    // Condensation trails (contrails)
    if (!anomActive) {{
      for (let ci=0; ci<2; ci++) {{
        const trailX = -130, trailY = ci===0 ? 10 : -8;
        const trailGrad = ctx.createLinearGradient(trailX,trailY,trailX-200,trailY);
        trailGrad.addColorStop(0,'rgba(255,255,255,0.5)');
        trailGrad.addColorStop(0.3,'rgba(255,255,255,0.25)');
        trailGrad.addColorStop(1,'rgba(255,255,255,0)');
        ctx.beginPath();
        ctx.moveTo(trailX, trailY-3);
        ctx.lineTo(trailX-200, trailY-8+(ci*4));
        ctx.lineTo(trailX-200, trailY+8+(ci*4));
        ctx.lineTo(trailX, trailY+3);
        ctx.closePath();
        ctx.fillStyle = trailGrad;
        ctx.fill();
      }}
    }}

    ctx.restore();
  }}

  // ── Particles ─────────────────────────────────────────────────────────────
  const particles = [];

  function spawnSmoke(ox,oy,color) {{
    for (let i=0;i<3;i++) {{
      particles.push({{
        x:ox+rand(-8,8), y:oy+rand(-4,4),
        vx:rand(-2.5,-0.4), vy:rand(-0.8,0.8),
        life:1, maxLife:1, r:rand(6,18),
        color:color||'rgba(80,80,80,', type:'smoke'
      }});
    }}
  }}

  function spawnFire(ox,oy) {{
    for (let i=0;i<4;i++) {{
      particles.push({{
        x:ox+rand(-6,6), y:oy+rand(-4,4),
        vx:rand(-3,-0.5), vy:rand(-1.5,1.5),
        life:1, maxLife:1, r:rand(4,12),
        color:'rgba(255,100,0,', type:'fire'
      }});
    }}
  }}

  function spawnSparks(ox,oy) {{
    for (let i=0;i<5;i++) {{
      particles.push({{
        x:ox+rand(-15,15), y:oy+rand(-10,10),
        vx:rand(-4,4), vy:rand(-4,1),
        life:1, maxLife:1, r:rand(1.5,4),
        color:'rgba(255,220,50,', type:'spark'
      }});
    }}
  }}

  function spawnDebris(ox,oy) {{
    for (let i=0;i<6;i++) {{
      particles.push({{
        x:ox+rand(-20,20), y:oy+rand(-15,15),
        vx:rand(-5,5), vy:rand(-4,2),
        life:1, maxLife:1, r:rand(2,6),
        color:'rgba(180,150,100,', type:'debris'
      }});
    }}
  }}

  function updateDrawParticles() {{
    for (let i=particles.length-1; i>=0; i--) {{
      const p = particles[i];
      p.x  += p.vx; p.y += p.vy;
      p.vy += 0.05;  // gravity for debris
      if (p.type==='smoke') {{ p.r*=1.04; p.vx*=0.97; }}
      const decay = p.type==='smoke'?0.018 : p.type==='fire'?0.03 : 0.055;
      p.life -= decay;
      if (p.life<=0) {{ particles.splice(i,1); continue; }}
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = p.color + p.life.toFixed(2) + ')';
      ctx.fill();
    }}
  }}

  // ── Pressure wave ─────────────────────────────────────────────────────────
  const waves = [];
  function spawnWave(ox,oy) {{
    waves.push({{ x:ox, y:oy, r:5, life:1 }});
  }}

  function updateDrawWaves() {{
    for (let i=waves.length-1; i>=0; i--) {{
      const w=waves[i];
      w.r   += 4;
      w.life -= 0.022;
      if (w.life<=0) {{ waves.splice(i,1); continue; }}
      ctx.beginPath();
      ctx.arc(w.x, w.y, w.r, 0, Math.PI*2);
      ctx.strokeStyle = `rgba(100,200,255,${{(w.life*0.55).toFixed(2)}})`;
      ctx.lineWidth   = 2.5;
      ctx.stroke();
    }}
  }}

  // ── Fog overlay ───────────────────────────────────────────────────────────
  function drawFog(alpha) {{
    if (alpha<=0) return;
    const fg = ctx.createLinearGradient(0,0,0,H);
    fg.addColorStop(0, `rgba(30,30,50,${{(alpha*0.4).toFixed(2)}}`+')');
    fg.addColorStop(0.5,`rgba(20,20,40,${{(alpha*0.25).toFixed(2)}}`+')');
    fg.addColorStop(1,  `rgba(10,10,20,${{(alpha*0.5).toFixed(2)}}`+')');
    ctx.fillStyle = fg;
    ctx.fillRect(0,0,W,H);
  }}

  // ── Screen flash ──────────────────────────────────────────────────────────
  let flashAlpha = 0;
  function triggerFlash(color) {{
    flashAlpha = 0.45;
    flashColor = color || 'rgba(255,100,0,';
  }}
  let flashColor = 'rgba(255,100,0,';

  // ── State ─────────────────────────────────────────────────────────────────
  let step       = 0;
  let planeX, planeY, planeRoll=0, planePitch=0;
  let tRoll=0, tPitch=0;
  let altOsc=0;
  let lastStep=0;
  const STEP_DT = 0.20;
  let elapsed=0, lastTime=null;
  let anomalyActive=false;
  let waveTimer=0;

  // HUD
  function nr(v,sp) {{ return (v+(Math.random()-0.5)*sp*2).toFixed(0); }}

  function updateHUD(anom, st) {{
    document.getElementById('h-alt').textContent =
      (anom&&ANOMALY==='CabinPressureLoss'?(30000+st*190):nr(30000,35))+' ft';
    document.getElementById('h-spd').textContent =
      (anom&&ANOMALY==='BirdStrike'?Math.max(185,250-st*3):nr(250,3))+' kts';
    document.getElementById('h-rpm').textContent =
      (anom&&ANOMALY==='EngineFailure'?Math.max(420,2500-st*55):nr(2500,25));
    document.getElementById('h-hyd').textContent =
      (anom&&ANOMALY==='HydraulicLeak'?Math.max(900,3000-st*26):nr(3000,18))+' psi';
    document.getElementById('h-fuel').textContent =
      (anom&&ANOMALY==='FuelLeak'?(800+st*22):nr(800,10))+' pph';
    document.getElementById('h-vib').textContent =
      (anom&&ANOMALY==='BirdStrike'?(0.5+st*0.1).toFixed(2):(0.5+(Math.random()-0.5)*0.06).toFixed(2))+' mm/s';

    document.getElementById('st-step').textContent = 'Step '+step+' / '+TOTAL;
    document.getElementById('pbar-fill').style.width = (step/TOTAL*100)+'%';

    const stEl   = document.getElementById('st-txt');
    const banner = document.getElementById('anom-warn');
    if (anom) {{
      stEl.textContent = '⚠ ANOMALY';
      stEl.style.color = '#ff4444';
      banner.style.display = 'block';
    }} else {{
      stEl.textContent = 'NORMAL';
      stEl.style.color = '#00ee88';
      banner.style.display = 'none';
    }}
  }}

  // ── Main loop ─────────────────────────────────────────────────────────────
  function frame(ts) {{
    requestAnimationFrame(frame);
    if (!lastTime) lastTime = ts;
    const dt = Math.min((ts-lastTime)/1000, 0.05);
    lastTime = ts;
    elapsed += dt;

    anomalyActive = (step >= ANOM_START);

    // Step counter
    lastStep += dt;
    if (lastStep >= STEP_DT && step < TOTAL) {{
      step++;
      lastStep = 0;
      updateHUD(anomalyActive, step);
      if (step >= TOTAL)
        document.getElementById('st-step').textContent = 'Simulation complete';
    }}

    // Scroll terrain parallax
    for (let li=0;li<TERRAIN_LAYERS.length;li++) {{
      terrainOffsets[li] += TERRAIN_LAYERS[li].speed * dt * 200;
    }}

    // Scroll clouds
    CLOUDS.forEach(cl => {{
      cl.x -= cl.speed * 60 * dt / W * W;
      cl.x += cl.speed * 60 * dt;
      if (cl.x * W > W + 200) cl.x = -200/W;
      if (cl.x * W < -200)    cl.x = (W+200)/W;
    }});

    // Plane position
    planeX = W * 0.52;
    const turbY = (Math.sin(elapsed*1.6)*0.55+Math.sin(elapsed*3.0)*0.25+Math.sin(elapsed*5.1)*0.1)*0.009;
    altOsc = Math.sin(elapsed*0.22)*12;

    // Normal flight attitude
    tRoll  = turbY*0.4 + Math.sin(elapsed*0.28)*0.03;
    tPitch = turbY*0.2 + Math.sin(elapsed*0.51)*0.015;

    if (anomalyActive) {{
      const sev = clamp((step-ANOM_START)/4, 0, 1);
      const shk = C.shake * sev;
      planeX += (Math.random()-0.5)*shk*1.5;
      altOsc += (Math.random()-0.5)*shk*1.5;
      tRoll  += C.roll  * sev * Math.sin(elapsed*5.0);
      tPitch += C.pitch * sev * Math.sin(elapsed*4.0);

      if (sev>0) {{
        // Spawn effects
        const px = planeX, py = H*0.38 + altOsc;
        if (C.fire   && Math.random()<0.9) spawnFire(px-90,  py+20);
        if (C.smoke  && Math.random()<0.9) spawnSmoke(px-100, py+15, ANOMALY==='FuelLeak'?'rgba(180,160,0,':'rgba(70,70,70,');
        if (C.sparks && Math.random()<0.4) spawnSparks(px+rand(-30,30), py+rand(-20,20));
        if (C.debris && Math.random()<0.3) spawnDebris(px+rand(-20,20), py+rand(-15,15));
        if (C.pressWave) {{
          waveTimer += dt;
          if (waveTimer>0.18) {{ spawnWave(px, py); waveTimer=0; }}
        }}
        if (C.fire && sev>0.5 && Math.random()<0.05) triggerFlash('rgba(255,80,0,');
        if (C.sparks && Math.random()<0.03) triggerFlash('rgba(180,100,255,');
      }}
    }}

    planeRoll  += (tRoll -planeRoll) *5*dt;
    planePitch += (tPitch-planePitch)*5*dt;
    planeY = H * 0.38 + altOsc;

    // ── Draw frame ───────────────────────────────────────────────────────
    ctx.clearRect(0,0,W,H);

    drawSky(anomalyActive, step);
    drawStars(anomalyActive);
    drawHaze(anomalyActive);

    // Clouds (behind plane)
    CLOUDS.forEach(cl => drawCloud(cl));

    drawTerrain();

    // Particles behind plane
    updateDrawParticles();
    updateDrawWaves();

    drawAirplane(planeX, planeY, planeRoll, planePitch, anomalyActive, elapsed);

    // Fog overlay
    if (anomalyActive) drawFog(C.fogAlpha * clamp((step-ANOM_START)/4,0,1));

    // Screen flash
    if (flashAlpha > 0) {{
      ctx.fillStyle = flashColor + flashAlpha.toFixed(2) + ')';
      ctx.fillRect(0,0,W,H);
      flashAlpha -= 0.03;
    }}

    // Vignette
    const vig = ctx.createRadialGradient(W/2,H/2,H*0.25,W/2,H/2,H*0.85);
    vig.addColorStop(0,'rgba(0,0,0,0)');
    vig.addColorStop(1,'rgba(0,0,0,0.45)');
    ctx.fillStyle = vig;
    ctx.fillRect(0,0,W,H);
  }}

  requestAnimationFrame(frame);
}})();
</script>
</body>
</html>"""

    components.html(html, height=430, scrolling=False)


# ─── MAIN APP ──────────────────────────────────────────────────────────────────

def run_anomaly_app():

    with open("models/anomaly_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    autoencoder = model_data["model"]
    threshold   = model_data["threshold"]

    st.title("Aircraft Anomaly Detection Simulator")

    anomaly_type = st.selectbox(
        "Select an anomaly to inject",
        ["SensorBias", "HydraulicLeak", "EngineFailure",
         "CabinPressureLoss", "BirdStrike", "FuelLeak", "ElectricalFault"]
    )

    n_steps = st.slider("Simulation Time Steps", 10, 50, 25)

    # ── Synthetic data generator ─────────────────────────────────────────────
    def generate_flight_data(n_steps, anomaly=None):
        np.random.seed(42)
        data = {
            "Altitude_ft":           np.random.normal(30000, 1000, n_steps),
            "Airspeed_knots":        np.random.normal(250, 10, n_steps),
            "EngineRPM":             np.random.normal(2500, 200, n_steps),
            "EngineOilPressure_psi": np.random.normal(50, 5, n_steps),
            "FuelFlow_pph":          np.random.normal(800, 50, n_steps),
            "Vibration_mm_s":        np.random.normal(0.5, 0.1, n_steps),
            "HydraulicPressure_psi": np.random.normal(3000, 100, n_steps),
        }
        X = np.column_stack(list(data.values()))

        if anomaly == "SensorBias":
            X[-5:, 0] += 3000
        elif anomaly == "HydraulicLeak":
            X[-5:, 6] -= 500
        elif anomaly == "EngineFailure":
            X[-5:, 2] -= 1000
        elif anomaly == "CabinPressureLoss":
            X[-5:, 0] += 5000
        elif anomaly == "BirdStrike":
            X[-5:, 5] += 2.0
            X[-5:, 1] -= 50
        elif anomaly == "FuelLeak":
            X[-5:, 4] += 400
        elif anomaly == "ElectricalFault":
            X[-5:, 3] -= 15
            X[-5:, 6] -= 800

        return X, list(data.keys())

    # ── Run button ───────────────────────────────────────────────────────────
    if st.button("Run Simulation"):

        # ── NEW: Realistic flight feed ──────────────────────────────────────
        st.markdown("### ✈️ Live Flight Simulation")
        render_flight_feed(anomaly_type, n_steps)

        # ── Existing simulation logic (completely unchanged) ────────────────
        X, feature_names = generate_flight_data(n_steps, anomaly_type)
        recon  = autoencoder.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1)

        explainer   = shap.Explainer(autoencoder, X)
        shap_values = explainer(X)

        # plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(errors, label="Reconstruction Error")
        ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax.set_title(f"Anomaly Detection Timeline (Injected: {anomaly_type})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Reconstruction Error")
        ax.legend()
        st.pyplot(fig)

        # timeline loop
        for t in range(n_steps):

            status = "NORMAL"
            if errors[t] > threshold:
                status = "🚨 ANOMALY DETECTED!"

            metrics = {f: X[t, i] for i, f in enumerate(feature_names)}

            st.markdown(
                f"**Time: {t} | Status: {status} | Rec. Error: {errors[t]:.5f} "
                f"(Threshold: {threshold:.5f})**"
            )
            st.write("   └─ Key Metrics:", metrics)

            if status != "NORMAL":

                shap_vals = shap_values[t].values.flatten()
                shap_vals = np.nan_to_num(shap_vals)

                idxs = np.argsort(np.abs(shap_vals))[-3:][::-1]

                top_factors = []
                for idx in idxs:
                    if idx < len(feature_names):
                        top_factors.append({
                            "feature": feature_names[idx],
                            "value":   float(X[t, idx]),
                            "shap":    float(shap_vals[idx])
                        })

                # Build metrics HTML
                metrics_html = ""
                for k, v in metrics.items():
                    metrics_html += "<p style='margin:2px 0;'><b>{}</b>: {:.3f}</p>".format(k, v)

                # Build factors HTML
                factors_html = ""
                for f in top_factors:
                    factors_html += (
                        "<p style='margin:2px 0;'><b>{}</b>: value={:.3f}, shap={:.3f}</p>"
                        .format(f['feature'], f['value'], f['shap'])
                    )

                # Summary box
                html = """
                <div style="border: 3px solid #ff4d4d; padding: 25px; border-radius: 12px;
                            background-color: #ffe6e6; margin-top: 20px;">
                    <h2 style="color:#cc0000; margin-top:0;">🚨 ANOMALY DETECTED</h2>
                    <p><b>Possible Failure:</b> {}</p>
                    <p><b>Time Index:</b> {}</p>
                    <p><b>Reconstruction Error:</b> {:.5f}</p>
                    <p><b>Threshold:</b> {:.5f}</p>
                    <h4 style="margin-top: 20px;"><b>Key Metrics:</b></h4>
                    {}
                    <h4 style="margin-top: 20px;">🔎 <b>Explainable AI (Top 3 Factors):</b></h4>
                    {}
                </div>
                """.format(anomaly_type, t, errors[t], threshold, metrics_html, factors_html)

                st.markdown(html, unsafe_allow_html=True)

                # store popup
                st.session_state["anomaly_popup"] = {
                    "time":             int(t),
                    "rec_error":        float(errors[t]),
                    "threshold":        float(threshold),
                    "status":           "ANOMALY DETECTED",
                    "possible_failure": anomaly_type,
                    "metrics":          {k: float(v) for k, v in metrics.items()},
                    "top_factors":      top_factors
                }

                break


if __name__ == "__main__":
    run_anomaly_app()