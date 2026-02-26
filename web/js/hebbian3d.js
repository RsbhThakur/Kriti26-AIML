// ──────────────────────────────────────────────────────────────────────────────
// hebbian3d.js — Three.js 3D Hebbian Learning Visualizer
// "Neurons that fire together, wire together" — animated demonstration
//
// Shows synapse reinforcement accumulating across 30 boards:
//   - Edges start cold/dark, glow hotter as Hebbian weight grows
//   - "Pulse" particles travel along recently-strengthened edges
//   - Nodes bloom in proportion to their cumulative participation
//   - Camera slowly orbits so the viewer can see structure form
// ──────────────────────────────────────────────────────────────────────────────

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════
let scene, camera, renderer, controls;
let nodeInstances, glowInstances, edgeLines, pulseParticles, ambientParticles;
let containerEl   = null;
let animFrameId   = null;
let isActive      = false;
let time          = 0;

// Data
let topo          = null;   // { nodes, edges }
let positions3D   = null;   // Float64Array [x0,y0,z0,...]
let nodeScales    = null;   // base radius per node
let edgeSrcArr    = null;   // Int32Array
let edgeTgtArr    = null;   // Int32Array
let N = 0, E = 0;

// Hebbian animation
let hebbWeights   = null;   // Float32Array[E]   current normalised weights [0..1]
let prevWeights   = null;   // Float32Array[E]   previous frame weights
let nodeStrength  = null;   // Float32Array[N]
let pulseAge      = null;   // Float32Array[E]   age of edge pulse (0=none, 1→0 decaying)

// Lights
let centerLight   = null;
let pulseLightA   = null;
let pulseLightB   = null;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════
const SPRING_LEN      = 6;
const PARTICLE_COUNT  = 1800;
const PULSE_PTS       = 3000;      // max pulse particles
const PULSE_SPEED     = 0.04;
const PULSE_LIFETIME  = 1.2;       // seconds (normalised)

// Color ramp: cold → warm → hot  (synapse strength)
const C_COLD   = new THREE.Color(0x18181b);   // zinc-900 — dormant
const C_COOL   = new THREE.Color(0x3730a3);   // indigo-800
const C_WARM   = new THREE.Color(0x0ea5e9);   // sky-500
const C_HOT    = new THREE.Color(0xfbbf24);   // amber-400
const C_WHITE  = new THREE.Color(0xfefce8);   // near-white — peak

// Node colors
const C_NODE_DIM   = new THREE.Color(0x27272a);
const C_NODE_LIT   = new THREE.Color(0xfbbf24);
const C_NODE_PEAK  = new THREE.Color(0xfef3c7);

// ═══════════════════════════════════════════════════════════════════════════════
// 3-D FORCE LAYOUT (same robust method as viz3d.js)
// ═══════════════════════════════════════════════════════════════════════════════
function compute3DLayout(nodes, edges) {
    const n = nodes.length;
    const pos = new Float64Array(n * 3);
    const vel = new Float64Array(n * 3);

    const EP = [];
    for (const e of edges) {
        EP.push(+e.source, +e.target);
    }
    const edgePairs = new Int32Array(EP);
    const eCount = edgePairs.length / 2;

    // Random initial on sphere shell
    for (let i = 0; i < n; i++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const r  = 14 + Math.random() * 22;
        pos[i * 3]     = r * Math.sin(ph) * Math.cos(th);
        pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
        pos[i * 3 + 2] = r * Math.cos(ph);
    }

    const ITER = n < 500 ? 140 : n < 1500 ? 90 : 55;
    const useSample = n > 1500;
    const SAMP = useSample ? n * 10 : 0;

    for (let it = 0; it < ITER; it++) {
        const cool = 1.0 - it / ITER;
        const repK = 45 * cool;
        const sprK = 0.005 * cool;

        // Repulsion
        if (useSample) {
            const sc = (n * (n - 1)) / (2 * SAMP);
            for (let s = 0; s < SAMP; s++) {
                const i = (Math.random() * n) | 0;
                let j = (Math.random() * (n - 1)) | 0;
                if (j >= i) j++;
                const ii = i * 3, jj = j * 3;
                const dx = pos[ii] - pos[jj], dy = pos[ii + 1] - pos[jj + 1], dz = pos[ii + 2] - pos[jj + 2];
                const d2 = dx * dx + dy * dy + dz * dz + 0.01;
                const d  = Math.sqrt(d2);
                const f  = repK * sc / d2;
                const fx = dx / d * f, fy = dy / d * f, fz = dz / d * f;
                vel[ii] += fx; vel[ii + 1] += fy; vel[ii + 2] += fz;
                vel[jj] -= fx; vel[jj + 1] -= fy; vel[jj + 2] -= fz;
            }
        } else {
            for (let i = 0; i < n; i++) {
                const ii = i * 3;
                for (let j = i + 1; j < n; j++) {
                    const jj = j * 3;
                    const dx = pos[ii] - pos[jj], dy = pos[ii + 1] - pos[jj + 1], dz = pos[ii + 2] - pos[jj + 2];
                    const d2 = dx * dx + dy * dy + dz * dz + 0.01;
                    const d  = Math.sqrt(d2);
                    const f  = repK / d2;
                    const fx = dx / d * f, fy = dy / d * f, fz = dz / d * f;
                    vel[ii] += fx; vel[ii + 1] += fy; vel[ii + 2] += fz;
                    vel[jj] -= fx; vel[jj + 1] -= fy; vel[jj + 2] -= fz;
                }
            }
        }

        // Springs
        for (let e = 0; e < eCount; e++) {
            const si = edgePairs[e * 2] * 3, ti = edgePairs[e * 2 + 1] * 3;
            const dx = pos[ti] - pos[si], dy = pos[ti + 1] - pos[si + 1], dz = pos[ti + 2] - pos[si + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
            const f = (d - SPRING_LEN) * sprK;
            const fx = dx / d * f, fy = dy / d * f, fz = dz / d * f;
            vel[si] += fx; vel[si + 1] += fy; vel[si + 2] += fz;
            vel[ti] -= fx; vel[ti + 1] -= fy; vel[ti + 2] -= fz;
        }

        // Damping + centering + integrate
        const damp = 0.88, cen = 0.001;
        for (let i = 0; i < n; i++) {
            const ii = i * 3;
            vel[ii]     = (vel[ii]     - pos[ii]     * cen) * damp;
            vel[ii + 1] = (vel[ii + 1] - pos[ii + 1] * cen) * damp;
            vel[ii + 2] = (vel[ii + 2] - pos[ii + 2] * cen) * damp;
            pos[ii]     += vel[ii];
            pos[ii + 1] += vel[ii + 1];
            pos[ii + 2] += vel[ii + 2];
        }
    }
    return pos;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
const _c = new THREE.Color();

function hebbEdgeColor(h, out) {
    // h ∈ [0,1] — normalised Hebbian weight
    if (h < 0.05)       out.copy(C_COLD);
    else if (h < 0.25)  out.copy(C_COLD).lerp(C_COOL, (h - 0.05) / 0.20);
    else if (h < 0.50)  out.copy(C_COOL).lerp(C_WARM, (h - 0.25) / 0.25);
    else if (h < 0.80)  out.copy(C_WARM).lerp(C_HOT,  (h - 0.50) / 0.30);
    else                 out.copy(C_HOT).lerp(C_WHITE, (h - 0.80) / 0.20);
    return out;
}

function nodeColor(s, out) {
    const t = Math.min(s, 1);
    if (t < 0.15)      out.copy(C_NODE_DIM);
    else if (t < 0.55) out.copy(C_NODE_DIM).lerp(C_NODE_LIT, (t - 0.15) / 0.40);
    else               out.copy(C_NODE_LIT).lerp(C_NODE_PEAK, (t - 0.55) / 0.45);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: initHebbian3D
// ═══════════════════════════════════════════════════════════════════════════════
export function initHebbian3D(container, topology) {
    containerEl = container;
    topo = topology;
    N = topology.nodes.length;
    E = topology.edges.length;

    const W = container.clientWidth  || 900;
    const H = container.clientHeight || 560;

    // ── Scene ──
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050508);
    scene.fog = new THREE.FogExp2(0x050508, 0.002);

    // ── Camera ──
    camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 600);
    camera.position.set(0, 28, 65);

    // ── Renderer ──
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // ── Controls ──
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping   = true;
    controls.dampingFactor   = 0.06;
    controls.autoRotate      = true;
    controls.autoRotateSpeed = 0.35;
    controls.minDistance      = 8;
    controls.maxDistance      = 200;

    // ── Lights ──
    scene.add(new THREE.AmbientLight(0x303050, 0.35));

    centerLight = new THREE.PointLight(0x6366f1, 1.0, 140);
    centerLight.position.set(0, 0, 0);
    scene.add(centerLight);

    pulseLightA = new THREE.PointLight(0x0ea5e9, 0, 80);
    pulseLightA.position.set(15, 10, 10);
    scene.add(pulseLightA);

    pulseLightB = new THREE.PointLight(0xfbbf24, 0, 80);
    pulseLightB.position.set(-15, -8, -12);
    scene.add(pulseLightB);

    // ── Layout ──
    console.time('[Hebb3D] layout');
    positions3D = compute3DLayout(topology.nodes, topology.edges);
    console.timeEnd('[Hebb3D] layout');

    // ── Degree → scale ──
    const deg = new Uint16Array(N);
    edgeSrcArr = new Int32Array(E);
    edgeTgtArr = new Int32Array(E);
    for (let i = 0; i < E; i++) {
        const s = +topology.edges[i].source;
        const t = +topology.edges[i].target;
        edgeSrcArr[i] = s;
        edgeTgtArr[i] = t;
        if (s < N) deg[s]++;
        if (t < N) deg[t]++;
    }
    const maxDeg = Math.max(...deg, 1);
    nodeScales = Float32Array.from(deg, d => 0.3 + (d / maxDeg) * 0.7);

    // Init Hebbian state arrays
    hebbWeights  = new Float32Array(E);
    prevWeights  = new Float32Array(E);
    nodeStrength = new Float32Array(N);
    pulseAge     = new Float32Array(E);   // all zero initially

    // ── Nodes (InstancedMesh) ──
    const sphGeo = new THREE.SphereGeometry(1, 10, 10);
    const sphMat = new THREE.MeshPhongMaterial({
        color: 0xffffff, emissive: 0x000000, shininess: 80,
    });
    nodeInstances = new THREE.InstancedMesh(sphGeo, sphMat, N);
    nodeInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    const dummy = new THREE.Object3D();
    for (let i = 0; i < N; i++) {
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(nodeScales[i] * 0.5);
        dummy.updateMatrix();
        nodeInstances.setMatrixAt(i, dummy.matrix);
        nodeInstances.setColorAt(i, C_NODE_DIM);
    }
    nodeInstances.instanceMatrix.needsUpdate = true;
    nodeInstances.instanceColor.needsUpdate  = true;
    scene.add(nodeInstances);

    // ── Glow shells (additive) ──
    const glowGeo = new THREE.SphereGeometry(1, 6, 6);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0xfbbf24, transparent: true, opacity: 0.08,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });
    glowInstances = new THREE.InstancedMesh(glowGeo, glowMat, N);
    glowInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < N; i++) {
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(nodeScales[i] * 1.8);
        dummy.updateMatrix();
        glowInstances.setMatrixAt(i, dummy.matrix);
        glowInstances.setColorAt(i, C_NODE_DIM);
    }
    glowInstances.instanceMatrix.needsUpdate = true;
    glowInstances.instanceColor.needsUpdate  = true;
    scene.add(glowInstances);

    // ── Edges (LineSegments) ──
    const ePos = new Float32Array(E * 6);
    const eCol = new Float32Array(E * 6);
    for (let i = 0; i < E; i++) {
        const s = edgeSrcArr[i], t = edgeTgtArr[i];
        const o = i * 6;
        ePos[o]     = positions3D[s * 3];     ePos[o + 1] = positions3D[s * 3 + 1]; ePos[o + 2] = positions3D[s * 3 + 2];
        ePos[o + 3] = positions3D[t * 3]; ePos[o + 4] = positions3D[t * 3 + 1]; ePos[o + 5] = positions3D[t * 3 + 2];
        // Start cold
        eCol[o] = C_COLD.r; eCol[o + 1] = C_COLD.g; eCol[o + 2] = C_COLD.b;
        eCol[o + 3] = C_COLD.r; eCol[o + 4] = C_COLD.g; eCol[o + 5] = C_COLD.b;
    }
    const eGeo = new THREE.BufferGeometry();
    eGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
    eGeo.setAttribute('color',    new THREE.BufferAttribute(eCol, 3));
    edgeLines = new THREE.LineSegments(eGeo, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false,
        linewidth: 1,
    }));
    scene.add(edgeLines);

    // ── Pulse particles (Points) ──
    const ppPos  = new Float32Array(PULSE_PTS * 3);
    const ppCol  = new Float32Array(PULSE_PTS * 3);
    const ppSize = new Float32Array(PULSE_PTS);
    const ppGeo  = new THREE.BufferGeometry();
    ppGeo.setAttribute('position', new THREE.BufferAttribute(ppPos, 3));
    ppGeo.setAttribute('color',    new THREE.BufferAttribute(ppCol, 3));
    ppGeo.setAttribute('size',     new THREE.BufferAttribute(ppSize, 1));

    // Hide all initially
    for (let i = 0; i < PULSE_PTS * 3; i++) ppPos[i] = 1e5;

    pulseParticles = new THREE.Points(ppGeo, new THREE.PointsMaterial({
        size: 0.5, vertexColors: true, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    }));
    scene.add(pulseParticles);

    // ── Ambient dust ──
    const aPos = new Float32Array(PARTICLE_COUNT * 3);
    const aCol = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const r  = 55 + Math.random() * 150;
        aPos[i * 3]     = r * Math.sin(ph) * Math.cos(th);
        aPos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
        aPos[i * 3 + 2] = r * Math.cos(ph);
        const b = 0.08 + Math.random() * 0.12;
        aCol[i * 3] = b * 0.7; aCol[i * 3 + 1] = b * 0.75; aCol[i * 3 + 2] = b;
    }
    const aGeo = new THREE.BufferGeometry();
    aGeo.setAttribute('position', new THREE.BufferAttribute(aPos, 3));
    aGeo.setAttribute('color',    new THREE.BufferAttribute(aCol, 3));
    ambientParticles = new THREE.Points(aGeo, new THREE.PointsMaterial({
        size: 0.25, vertexColors: true, transparent: true, opacity: 0.35,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    }));
    scene.add(ambientParticles);

    // ── Resize ──
    window.addEventListener('resize', () => {
        const w = containerEl.clientWidth, h = containerEl.clientHeight;
        if (w < 10 || h < 10) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    isActive = true;
    animate();
    console.log(`[Hebb3D] ready — ${N} nodes, ${E} edges`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION
// ═══════════════════════════════════════════════════════════════════════════════
function animate() {
    if (!isActive) return;
    animFrameId = requestAnimationFrame(animate);
    time += 0.016;
    controls.update();

    // Breathing center light
    if (centerLight) {
        const maxH = hebbWeights ? Math.max(0.1, ...hebbWeights) : 0.1;
        centerLight.intensity = 0.6 + maxH * 2.0 + Math.sin(time * 0.6) * 0.2;
    }

    // Fade pulse particles
    updatePulseParticles();

    // Ambient drift
    if (ambientParticles) ambientParticles.rotation.y += 0.00003;

    renderer.render(scene, camera);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PULSE PARTICLES – travel along strengthened edges
// ═══════════════════════════════════════════════════════════════════════════════
let pulseCursor = 0;   // round-robin index into PULSE_PTS

function spawnPulsesForFrame() {
    // For every edge whose Hebbian weight INCREASED this frame, spawn particles
    if (!hebbWeights || !prevWeights) return;

    const posArr = pulseParticles.geometry.attributes.position.array;
    const colArr = pulseParticles.geometry.attributes.color.array;

    let spawned = 0;
    const MAX_SPAWN = Math.min(400, PULSE_PTS / 3);

    for (let i = 0; i < E && spawned < MAX_SPAWN; i++) {
        const delta = hebbWeights[i] - prevWeights[i];
        if (delta < 0.002) continue;   // only visually-significant increases

        const count = Math.min(Math.ceil(delta * 15), 5);   // 1–5 particles per edge
        const s = edgeSrcArr[i], t = edgeTgtArr[i];
        const sx = positions3D[s * 3], sy = positions3D[s * 3 + 1], sz = positions3D[s * 3 + 2];
        const tx = positions3D[t * 3], ty = positions3D[t * 3 + 1], tz = positions3D[t * 3 + 2];

        hebbEdgeColor(hebbWeights[i], _c);

        for (let p = 0; p < count && spawned < MAX_SPAWN; p++) {
            const idx = pulseCursor % PULSE_PTS;
            pulseCursor++;
            spawned++;

            const frac = Math.random();
            const o3 = idx * 3;
            posArr[o3]     = sx + (tx - sx) * frac;
            posArr[o3 + 1] = sy + (ty - sy) * frac + (Math.random() - 0.5) * 0.5;
            posArr[o3 + 2] = sz + (tz - sz) * frac;

            colArr[o3]     = _c.r;
            colArr[o3 + 1] = _c.g;
            colArr[o3 + 2] = _c.b;

            // Store age in a parallel typed array
            pulseAge[idx] = 1.0;   // re-purpose pulseAge for particle lifetime (reuse size)
        }
    }

    pulseParticles.geometry.attributes.position.needsUpdate = true;
    pulseParticles.geometry.attributes.color.needsUpdate    = true;
}

// Keep pulseAge for the PULSE_PTS particles (re-sized array if needed)
let particleLife = null;

function ensureParticleLife() {
    if (!particleLife || particleLife.length !== PULSE_PTS) {
        particleLife = new Float32Array(PULSE_PTS);
    }
}

function updatePulseParticles() {
    if (!pulseParticles) return;
    ensureParticleLife();

    const posArr = pulseParticles.geometry.attributes.position.array;
    const colArr = pulseParticles.geometry.attributes.color.array;
    let anyActive = false;

    for (let i = 0; i < PULSE_PTS; i++) {
        if (particleLife[i] <= 0) continue;
        particleLife[i] -= 0.018;
        anyActive = true;

        if (particleLife[i] <= 0) {
            // Hide
            posArr[i * 3] = 1e5;
            posArr[i * 3 + 1] = 1e5;
            posArr[i * 3 + 2] = 1e5;
            colArr[i * 3] = 0; colArr[i * 3 + 1] = 0; colArr[i * 3 + 2] = 0;
        } else {
            // Drift outward slightly + fade
            posArr[i * 3 + 1] += 0.015;
            const fade = particleLife[i];
            colArr[i * 3]     *= 0.99;
            colArr[i * 3 + 1] *= 0.99;
            colArr[i * 3 + 2] *= 0.99;
        }
    }

    if (anyActive) {
        pulseParticles.geometry.attributes.position.needsUpdate = true;
        pulseParticles.geometry.attributes.color.needsUpdate    = true;
    }
}

// Replace pulseAge with particleLife for spawn
function _spawnParticle(idx) {
    ensureParticleLife();
    particleLife[idx] = 1.0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: setHebbianFrame  (called when user scrubs to board N)
// ═══════════════════════════════════════════════════════════════════════════════
export function setHebbianFrame(frame) {
    if (!nodeInstances || !topo) return;

    const hw = frame.hebbian_weights;
    const ns = frame.node_strength;

    // Save prev before update
    prevWeights.set(hebbWeights);

    // Update current
    for (let i = 0; i < E; i++) hebbWeights[i] = hw[i] || 0;
    for (let i = 0; i < N; i++) nodeStrength[i] = ns[i] || 0;

    // ── Edge colors ──
    const eCol = edgeLines.geometry.attributes.color.array;
    for (let i = 0; i < E; i++) {
        const h = hebbWeights[i];
        hebbEdgeColor(h, _c);

        // Brighten if recently strengthened
        const delta = hebbWeights[i] - prevWeights[i];
        const boost = delta > 0.002 ? Math.min(delta * 8, 1.0) : 0;

        const o = i * 6;
        const br = _c.r + boost * 0.4;
        const bg = _c.g + boost * 0.3;
        const bb = _c.b + boost * 0.2;

        eCol[o]     = br; eCol[o + 1] = bg; eCol[o + 2] = bb;
        eCol[o + 3] = br; eCol[o + 4] = bg; eCol[o + 5] = bb;
    }
    edgeLines.geometry.attributes.color.needsUpdate = true;

    // ── Nodes ──
    const dummy = new THREE.Object3D();
    for (let i = 0; i < N; i++) {
        const s = nodeStrength[i];
        nodeColor(s, _c);
        nodeInstances.setColorAt(i, _c);

        const sc = nodeScales[i] * (0.5 + s * 0.8);
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(sc);
        dummy.updateMatrix();
        nodeInstances.setMatrixAt(i, dummy.matrix);

        // Glow larger for active nodes
        const gsc = nodeScales[i] * (1.8 + s * 4.0);
        dummy.scale.setScalar(gsc);
        dummy.updateMatrix();
        glowInstances.setMatrixAt(i, dummy.matrix);
        nodeColor(s, _c);
        glowInstances.setColorAt(i, _c);
    }
    nodeInstances.instanceMatrix.needsUpdate = true;
    nodeInstances.instanceColor.needsUpdate  = true;
    glowInstances.instanceMatrix.needsUpdate = true;
    glowInstances.instanceColor.needsUpdate  = true;

    // ── Spawn pulse particles along strengthened edges ──
    spawnPulses();

    // ── Pulse lights track overall strength ──
    const meanH = hebbWeights.reduce((a, b) => a + b, 0) / E;
    if (pulseLightA) pulseLightA.intensity = meanH * 4;
    if (pulseLightB) pulseLightB.intensity = meanH * 3;
}

function spawnPulses() {
    if (!hebbWeights || !prevWeights) return;
    ensureParticleLife();

    const posArr = pulseParticles.geometry.attributes.position.array;
    const colArr = pulseParticles.geometry.attributes.color.array;
    let spawned = 0;
    const MAX_SPAWN = Math.min(500, PULSE_PTS / 2);

    for (let i = 0; i < E && spawned < MAX_SPAWN; i++) {
        const delta = hebbWeights[i] - prevWeights[i];
        if (delta < 0.001) continue;

        const count = Math.min(Math.ceil(delta * 20), 6);
        const s = edgeSrcArr[i], t = edgeTgtArr[i];
        const sx = positions3D[s * 3], sy = positions3D[s * 3 + 1], sz = positions3D[s * 3 + 2];
        const tx = positions3D[t * 3], ty = positions3D[t * 3 + 1], tz = positions3D[t * 3 + 2];

        hebbEdgeColor(hebbWeights[i], _c);

        for (let p = 0; p < count; p++) {
            const idx = pulseCursor % PULSE_PTS;
            pulseCursor++;
            spawned++;

            const frac = Math.random();
            const o = idx * 3;
            posArr[o]     = sx + (tx - sx) * frac + (Math.random() - 0.5) * 0.3;
            posArr[o + 1] = sy + (ty - sy) * frac + (Math.random() - 0.5) * 0.3;
            posArr[o + 2] = sz + (tz - sz) * frac + (Math.random() - 0.5) * 0.3;

            // Bright version of edge color
            colArr[o]     = Math.min(_c.r * 1.5, 1);
            colArr[o + 1] = Math.min(_c.g * 1.5, 1);
            colArr[o + 2] = Math.min(_c.b * 1.5, 1);

            particleLife[idx] = 0.7 + Math.random() * 0.5;
        }
    }

    pulseParticles.geometry.attributes.position.needsUpdate = true;
    pulseParticles.geometry.attributes.color.needsUpdate    = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: activate / deactivate
// ═══════════════════════════════════════════════════════════════════════════════
export function setHebb3DActive(active) {
    const was = isActive;
    isActive = active;
    if (active && !was) animate();
    if (!active && animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

export function toggleHebb3DRotate() {
    if (controls) {
        controls.autoRotate = !controls.autoRotate;
        return controls.autoRotate;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: cleanup
// ═══════════════════════════════════════════════════════════════════════════════
export function cleanupHebb3D() {
    isActive = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (renderer) {
        renderer.dispose();
        if (containerEl && renderer.domElement.parentNode === containerEl) {
            containerEl.removeChild(renderer.domElement);
        }
    }
    scene = camera = renderer = controls = null;
    nodeInstances = glowInstances = edgeLines = pulseParticles = ambientParticles = null;
    hebbWeights = prevWeights = nodeStrength = pulseAge = particleLife = null;
}
