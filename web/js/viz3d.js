// ──────────────────────────────────────────────────────────────────────────────
// viz3d.js — Three.js 3D Neural Network Walkthrough
// Immersive fly-through visualization of the BDH neural topology
// ──────────────────────────────────────────────────────────────────────────────

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── State ────────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let nodeInstances, glowInstances, edgeLines, particles;
let topology = null;
let positions3D = null;   // Float64Array  [x0,y0,z0, x1,y1,z1, ...]
let nodeScales  = [];     // base scale per node (from degree)
let containerEl = null;
let animFrameId = null;
let isActive    = false;
let centerLight = null;
let time        = 0;

// ── Constants ────────────────────────────────────────────────────────────────
const NODE_SEG         = 12;
const BASE_SIZE        = 0.35;
const GLOW_MULT        = 2.8;
const PARTICLE_COUNT   = 2500;

// Force layout parameters
const SPRING_LEN = 6;

// Design system palette
const C_INACT = new THREE.Color(0x27272a);  // zinc-800
const C_LOW   = new THREE.Color(0x6366f1);  // indigo-500
const C_MED   = new THREE.Color(0x0ea5e9);  // sky-500
const C_HIGH  = new THREE.Color(0xfbbf24);  // amber-400
const C_PEAK  = new THREE.Color(0xfefce8);  // near-white

// ══════════════════════════════════════════════════════════════════════════════
// 3-D FORCE LAYOUT  (flat Float64Array for speed)
// ══════════════════════════════════════════════════════════════════════════════
function compute3DLayout(nodes, edges) {
    const N = nodes.length;
    const pos = new Float64Array(N * 3);
    const vel = new Float64Array(N * 3);

    // Build edge-pair index for fast iteration
    const EP = [];
    for (const e of edges) {
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;
        EP.push(s, t);
    }
    const edgePairs = new Int32Array(EP);
    const E = edgePairs.length / 2;

    // Random initial positions on a sphere shell
    for (let i = 0; i < N; i++) {
        const θ = Math.random() * Math.PI * 2;
        const φ = Math.acos(2 * Math.random() - 1);
        const r = 15 + Math.random() * 25;
        pos[i * 3]     = r * Math.sin(φ) * Math.cos(θ);
        pos[i * 3 + 1] = r * Math.sin(φ) * Math.sin(θ);
        pos[i * 3 + 2] = r * Math.cos(φ);
    }

    const ITER = N < 500 ? 150 : N < 1500 ? 100 : 60;
    const useSample = N > 1500;
    const SAMP = useSample ? N * 12 : 0;

    for (let it = 0; it < ITER; it++) {
        const cool = 1.0 - it / ITER;
        const repK = 50 * cool;
        const sprK = 0.005 * cool;

        // ── Repulsion ──
        if (useSample) {
            const sc = (N * (N - 1)) / (2 * SAMP);
            for (let s = 0; s < SAMP; s++) {
                const i = (Math.random() * N) | 0;
                let j = (Math.random() * (N - 1)) | 0;
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
            for (let i = 0; i < N; i++) {
                const ii = i * 3;
                for (let j = i + 1; j < N; j++) {
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

        // ── Spring attraction (edges) ──
        for (let e = 0; e < E; e++) {
            const si = edgePairs[e * 2] * 3, ti = edgePairs[e * 2 + 1] * 3;
            const dx = pos[ti] - pos[si], dy = pos[ti + 1] - pos[si + 1], dz = pos[ti + 2] - pos[si + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
            const f = (d - SPRING_LEN) * sprK;
            const fx = dx / d * f, fy = dy / d * f, fz = dz / d * f;
            vel[si] += fx; vel[si + 1] += fy; vel[si + 2] += fz;
            vel[ti] -= fx; vel[ti + 1] -= fy; vel[ti + 2] -= fz;
        }

        // ── Centering + damping + integrate ──
        const damp = 0.88, cen = 0.001;
        for (let i = 0; i < N; i++) {
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

// ══════════════════════════════════════════════════════════════════════════════
// ACTIVATION → COLOR
// ══════════════════════════════════════════════════════════════════════════════
const _tmpC = new THREE.Color();
function actColor(val, out) {
    const t = Math.min(val / 1.5, 1.0);
    if (t < 0.08)      out.copy(C_INACT);
    else if (t < 0.35)  out.copy(C_LOW).lerp(C_MED,  (t - 0.08) / 0.27);
    else if (t < 0.65)  out.copy(C_MED).lerp(C_HIGH, (t - 0.35) / 0.30);
    else                 out.copy(C_HIGH).lerp(C_PEAK, (t - 0.65) / 0.35);
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC: init3DScene
// ══════════════════════════════════════════════════════════════════════════════
export function init3DScene(cEl, topologyData) {
    containerEl = cEl;
    topology = topologyData;
    const W = cEl.clientWidth;
    const H = cEl.clientHeight;

    // ── Scene ──
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    scene.fog = new THREE.FogExp2(0x09090b, 0.0025);

    // ── Camera ──
    camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 800);
    camera.position.set(0, 20, 70);

    // ── Renderer ──
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;
    cEl.appendChild(renderer.domElement);

    // ── Controls ──
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.06;
    controls.autoRotate     = true;
    controls.autoRotateSpeed = 0.4;
    controls.minDistance     = 8;
    controls.maxDistance     = 250;

    // ── Lights ──
    scene.add(new THREE.AmbientLight(0x404060, 0.4));

    centerLight = new THREE.PointLight(0x6366f1, 2.5, 160);
    centerLight.position.set(0, 0, 0);
    scene.add(centerLight);

    const rim = new THREE.PointLight(0x0ea5e9, 1.8, 120);
    rim.position.set(35, 25, -18);
    scene.add(rim);

    const warm = new THREE.PointLight(0xfbbf24, 0.6, 100);
    warm.position.set(-30, -20, 25);
    scene.add(warm);

    // ── Compute 3-D force layout ──
    console.time('[3D] force layout');
    positions3D = compute3DLayout(topology.nodes, topology.edges);
    console.timeEnd('[3D] force layout');

    // ── Degree → base scale ──
    const N = topology.nodes.length;
    const deg = new Uint16Array(N);
    for (const e of topology.edges) {
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;
        if (s < N) deg[s]++;
        if (t < N) deg[t]++;
    }
    const maxDeg = Math.max(...deg, 1);
    nodeScales = Array.from(deg, d => BASE_SIZE + (d / maxDeg) * BASE_SIZE * 2.5);

    // ── Nodes (InstancedMesh) ──
    const sphGeo = new THREE.SphereGeometry(1, NODE_SEG, NODE_SEG);
    const sphMat = new THREE.MeshPhongMaterial({
        color: 0xffffff, emissive: 0x000000, shininess: 90,
    });

    nodeInstances = new THREE.InstancedMesh(sphGeo, sphMat, N);
    nodeInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    const dummy = new THREE.Object3D();
    for (let i = 0; i < N; i++) {
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(nodeScales[i]);
        dummy.updateMatrix();
        nodeInstances.setMatrixAt(i, dummy.matrix);
        nodeInstances.setColorAt(i, C_INACT);
    }
    nodeInstances.instanceMatrix.needsUpdate = true;
    nodeInstances.instanceColor.needsUpdate  = true;
    scene.add(nodeInstances);

    // ── Glow shells (additive blending) ──
    const glowGeo = new THREE.SphereGeometry(1, 8, 8);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0x6366f1, transparent: true, opacity: 0.12,
        blending: THREE.AdditiveBlending, depthWrite: false,
    });

    glowInstances = new THREE.InstancedMesh(glowGeo, glowMat, N);
    glowInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < N; i++) {
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(nodeScales[i] * GLOW_MULT);
        dummy.updateMatrix();
        glowInstances.setMatrixAt(i, dummy.matrix);
        glowInstances.setColorAt(i, C_LOW);
    }
    glowInstances.instanceMatrix.needsUpdate = true;
    glowInstances.instanceColor.needsUpdate  = true;
    scene.add(glowInstances);

    // ── Edges (LineSegments) ──
    const eCnt = topology.edges.length;
    const ePos = new Float32Array(eCnt * 6);
    const eCol = new Float32Array(eCnt * 6);

    for (let i = 0; i < eCnt; i++) {
        const e = topology.edges[i];
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;
        const o = i * 6;
        ePos[o]   = positions3D[s * 3];     ePos[o + 1] = positions3D[s * 3 + 1]; ePos[o + 2] = positions3D[s * 3 + 2];
        ePos[o + 3] = positions3D[t * 3]; ePos[o + 4] = positions3D[t * 3 + 1]; ePos[o + 5] = positions3D[t * 3 + 2];
        // dim initial color
        eCol[o] = 0.04; eCol[o + 1] = 0.04; eCol[o + 2] = 0.06;
        eCol[o + 3] = 0.04; eCol[o + 4] = 0.04; eCol[o + 5] = 0.06;
    }

    const eGeo = new THREE.BufferGeometry();
    eGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
    eGeo.setAttribute('color',    new THREE.BufferAttribute(eCol, 3));

    edgeLines = new THREE.LineSegments(eGeo, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.7,
        blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(edgeLines);

    // ── Ambient particles ──
    const pPos = new Float32Array(PARTICLE_COUNT * 3);
    const pCol = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const θ = Math.random() * Math.PI * 2;
        const φ = Math.acos(2 * Math.random() - 1);
        const r = 60 + Math.random() * 180;
        pPos[i * 3]     = r * Math.sin(φ) * Math.cos(θ);
        pPos[i * 3 + 1] = r * Math.sin(φ) * Math.sin(θ);
        pPos[i * 3 + 2] = r * Math.cos(φ);
        const b = 0.15 + Math.random() * 0.25;
        pCol[i * 3] = b * 0.8;  pCol[i * 3 + 1] = b * 0.85;  pCol[i * 3 + 2] = b;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute('color',    new THREE.BufferAttribute(pCol, 3));

    particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
        size: 0.35, vertexColors: true, transparent: true, opacity: 0.45,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    }));
    scene.add(particles);

    // ── Resize handler ──
    const onResize = () => {
        const w = containerEl.clientWidth, h = containerEl.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    // ── Kick off render loop ──
    isActive = true;
    animate();
    console.log('[3D] scene initialised —', N, 'nodes,', eCnt, 'edges');
}

// ══════════════════════════════════════════════════════════════════════════════
// ANIMATION LOOP
// ══════════════════════════════════════════════════════════════════════════════
function animate() {
    if (!isActive) return;
    animFrameId = requestAnimationFrame(animate);
    time += 0.01;

    controls.update();

    // Gentle light pulse
    if (centerLight) {
        centerLight.intensity = 2.5 + Math.sin(time * 0.8) * 0.4;
    }

    // Slow particle drift
    if (particles) particles.rotation.y += 0.00004;

    renderer.render(scene, camera);
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC: update with frame data (called each frame step)
// ══════════════════════════════════════════════════════════════════════════════
export function update3DFrame(frame) {
    if (!nodeInstances || !topology) return;
    const { activations, prev_activations } = frame;
    const N = topology.nodes.length;
    const dummy = new THREE.Object3D();
    const c = new THREE.Color();

    // ── Nodes + glow ──
    for (let i = 0; i < N; i++) {
        const act = activations[i] || 0;
        const t   = Math.min(act / 1.5, 1.0);

        actColor(act, c);
        nodeInstances.setColorAt(i, c);

        const sc = nodeScales[i] * (1.0 + t * 0.6);
        dummy.position.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
        dummy.scale.setScalar(sc);
        dummy.updateMatrix();
        nodeInstances.setMatrixAt(i, dummy.matrix);

        const gsc = nodeScales[i] * GLOW_MULT * (1.0 + t * 2.0);
        dummy.scale.setScalar(gsc);
        dummy.updateMatrix();
        glowInstances.setMatrixAt(i, dummy.matrix);
        actColor(act, c);
        glowInstances.setColorAt(i, c);
    }
    nodeInstances.instanceMatrix.needsUpdate = true;
    nodeInstances.instanceColor.needsUpdate  = true;
    glowInstances.instanceMatrix.needsUpdate = true;
    glowInstances.instanceColor.needsUpdate  = true;

    // ── Edges ──
    const eArr = edgeLines.geometry.attributes.color.array;
    for (let i = 0; i < topology.edges.length; i++) {
        const e = topology.edges[i];
        const s = typeof e.source === 'object' ? e.source.index : +e.source;
        const t = typeof e.target === 'object' ? e.target.index : +e.target;

        const ySrc = prev_activations[s] || 0;
        const xTgt = activations[t]      || 0;
        const flow = Math.abs(ySrc * e.weight * xTgt);
        const ft   = Math.min(flow * 15, 1.0);
        const o    = i * 6;

        if (ft < 0.04) {
            eArr[o] = 0.04; eArr[o + 1] = 0.04; eArr[o + 2] = 0.06;
            eArr[o + 3] = 0.04; eArr[o + 4] = 0.04; eArr[o + 5] = 0.06;
        } else {
            c.copy(C_LOW).lerp(C_PEAK, ft);
            eArr[o]     = c.r * ft; eArr[o + 1] = c.g * ft; eArr[o + 2] = c.b * ft;
            eArr[o + 3] = c.r * ft; eArr[o + 4] = c.g * ft; eArr[o + 5] = c.b * ft;
        }
    }
    edgeLines.geometry.attributes.color.needsUpdate = true;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC: activate / deactivate (pause render loop when tab hidden)
// ══════════════════════════════════════════════════════════════════════════════
export function set3DActive(active) {
    const was = isActive;
    isActive = active;
    if (active && !was) animate();
    if (!active && animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC: toggle auto-rotate
// ══════════════════════════════════════════════════════════════════════════════
export function toggleAutoRotate() {
    if (controls) {
        controls.autoRotate = !controls.autoRotate;
        return controls.autoRotate;
    }
    return false;
}

// ══════════════════════════════════════════════════════════════════════════════
// PUBLIC: cleanup
// ══════════════════════════════════════════════════════════════════════════════
export function cleanup3D() {
    isActive = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (renderer) {
        renderer.dispose();
        if (containerEl && renderer.domElement.parentNode === containerEl) {
            containerEl.removeChild(renderer.domElement);
        }
    }
    scene = camera = renderer = controls = null;
    nodeInstances = glowInstances = edgeLines = particles = null;
}
