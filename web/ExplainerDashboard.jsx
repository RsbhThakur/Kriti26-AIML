import React, { useState } from 'react';
import { Play, Pause, FastForward, Activity, Hexagon, Database, BookOpen, ChevronRight } from 'lucide-react';

export default function ExplainerDashboard() {
    const [isPlaying, setIsPlaying] = useState(false);

    return (
        <div className="min-h-screen bg-zinc-950 text-zinc-300 font-sans selection:bg-indigo-500/30">
            {/* Top Navigation / Header */}
            <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 bg-zinc-900/50 backdrop-blur-md border-b border-white/5 shadow-2xl shadow-black/40">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center text-white font-bold shadow-lg shadow-indigo-500/20">
                        <Hexagon size={20} />
                    </div>
                    <div>
                        <h1 className="text-sm font-semibold text-zinc-100 tracking-wide">BDH Neural Pathfinding</h1>
                        <p className="text-xs text-zinc-500 font-medium">Interactive Explainer Dashboard</p>
                    </div>
                </div>

                {/* Global Controls */}
                <div className="flex items-center gap-4 bg-zinc-950/50 px-4 py-2 rounded-xl border border-white/5 transition-all">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 text-sm font-medium text-zinc-300 hover:text-white transition-colors"
                    >
                        {isPlaying ? <Pause size={16} className="text-indigo-400" /> : <Play size={16} className="text-emerald-400" />}
                        {isPlaying ? 'PAUSE' : 'RUN ENGINE'}
                    </button>
                    <div className="h-4 w-px bg-zinc-800"></div>
                    <button className="text-zinc-500 hover:text-zinc-300 transition-colors">
                        <FastForward size={16} />
                    </button>
                </div>
            </header>

            {/* Main 3-Pane Layout (Bento Box) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 max-w-[1800px] mx-auto h-[calc(100vh-73px)]">

                {/* Left Sidebar: Navigation & Controls */}
                <aside className="lg:col-span-2 flex flex-col gap-6">
                    <nav className="flex flex-col gap-2">
                        <h2 className="text-xs font-bold uppercase tracking-[0.15em] text-zinc-500 mb-2">Chapters</h2>
                        {['Neuron Graph', 'Sparse Brain', 'Topology', 'Math Theory'].map((item, i) => (
                            <button key={item} className={`flex items-center justify-between p-3 rounded-xl border transition-all duration-200 ${i === 3 ? 'bg-zinc-900 border-zinc-700 text-zinc-100 shadow-sm' : 'border-transparent text-zinc-400 hover:bg-zinc-900/50 hover:border-white/5 hover:text-zinc-200'}`}>
                                <span className="text-sm font-medium">{item}</span>
                                {i === 3 && <ChevronRight size={14} className="text-indigo-500" />}
                            </button>
                        ))}
                    </nav>

                    <div className="mt-auto p-4 rounded-xl bg-zinc-900 border border-white/5 shadow-lg shadow-black/20">
                        <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-zinc-500 mb-3">Model Status</h3>
                        <div className="flex items-center gap-3 mb-2">
                            <div className="relative flex h-2.5 w-2.5">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                            </div>
                            <span className="text-sm text-zinc-300 font-medium">GPU Inference Active</span>
                        </div>
                        <p className="text-xs text-zinc-500 font-mono mt-2">Epoch 109 / 120 (Layer 28)</p>
                    </div>
                </aside>

                {/* Center Stage: Visualizer & Math Explanations */}
                <main className="lg:col-span-7 flex flex-col gap-6 overflow-y-auto pr-4 custom-scrollbar">

                    {/* Interactive Pathfinding Grid Canvas */}
                    <div className="rounded-xl bg-zinc-900 border border-white/5 p-6 flex flex-col items-center shadow-2xl shadow-black/40 relative group">
                        <div className="w-full flex justify-between items-center mb-6">
                            <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-zinc-500">Live Grid Visualization</h3>
                            <div className="flex gap-4 text-xs font-medium text-zinc-400">
                                <span className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div> Start</span>
                                <span className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]"></div> End</span>
                                <span className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-amber-500/80"></div> Path</span>
                                <span className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-sm bg-slate-800 border border-slate-700"></div> Wall</span>
                            </div>
                        </div>

                        {/* 16x16 Grid Simulator */}
                        <div className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 shadow-inner w-full max-w-xl aspect-square">
                            <div className="grid grid-cols-16 grid-rows-16 w-full h-full gap-0 border border-zinc-800/50">
                                {/* Grid Cells Placeholder (React Mapping Logic Goes Here) */}
                                {Array.from({ length: 256 }).map((_, i) => {
                                    let bg = "bg-zinc-900 border-zinc-800/50";
                                    if (i === 18) bg = "bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.4)] z-10 border-emerald-400"; // Start
                                    else if (i === 234) bg = "bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.4)] z-10 border-rose-400"; // End
                                    else if ([34, 50, 66, 67, 68, 84, 100, 116, 117, 118, 134, 150, 166, 182, 198, 214, 230].includes(i)) bg = "bg-amber-500/80 border-amber-500/50"; // Path
                                    else if (i % 7 === 0 && i > 30 && i < 200) bg = "bg-slate-800 border-slate-700"; // Walls

                                    return (
                                        <div key={i} className={`w-full h-full border-[0.5px] ${bg} hover:border-white/20 transition-all duration-200 cursor-pointer`}></div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>

                    {/* Theory / Math Box */}
                    <div className="rounded-xl bg-gradient-to-br from-zinc-900 to-zinc-950 border-l-2 border-indigo-500 border-y border-r border-y-white/5 border-r-white/5 p-6 shadow-2xl shadow-black/40 overflow-hidden relative">
                        {/* Subtle background glow element */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>

                        <div className="relative z-10">
                            <div className="flex items-start gap-4 mb-4">
                                <div className="p-2.5 bg-indigo-500/10 rounded-lg text-indigo-400 border border-indigo-500/20 shadow-inner">
                                    <BookOpen size={20} />
                                </div>
                                <div>
                                    <h2 className="text-lg font-semibold text-zinc-100">Hebbian Memory Formation</h2>
                                    <p className="text-sm text-zinc-400 mt-1">Understanding the gating mechanism and synapse updates across layers.</p>
                                </div>
                            </div>

                            <div className="prose prose-invert max-w-none text-sm text-zinc-300 leading-relaxed space-y-4">
                                <p>
                                    The Neural Pathfinding Engine operates by treating the grid as a graph where functional connectivity emerges through training. The memory component $y$ forms a trace of active transitions.
                                </p>

                                <div className="my-6 p-5 rounded-lg bg-zinc-950 border border-zinc-800 flex items-center justify-center font-mono text-indigo-300 shadow-inner overflow-x-auto">
                                    {/* LaTeX Math Formula Goes Here */}
                                    <span className="text-base tracking-wide italic">{`$$ G_x = E \\otimes D_x \\quad \\text{where} \\quad x = \\text{ReLU}(v^* \\cdot D_x) $$`}</span>
                                </div>

                                <p>
                                    As the signal propagates from the <strong>Start Node</strong> to the <strong>End Node</strong>, the Hebbian update rules strengthen the connections along the shortest path, bypassing the dark slate walls. The memory trace $y$ is updated as:
                                </p>

                                <div className="my-6 p-5 rounded-lg bg-zinc-950 border border-zinc-800 flex items-center justify-center font-mono text-emerald-300 shadow-inner overflow-x-auto">
                                    {/* LaTeX Math Formula Goes Here */}
                                    <span className="text-base tracking-wide italic">{`$$ y^{(l+1)} = \\alpha y^{(l)} + \\eta (y^{(l)} \\cdot W^T) \\odot x^{(l)} $$`}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                </main>

                {/* Right Sidebar: Telemetry & Live Graphs (Stacked Bento Box) */}
                <aside className="lg:col-span-3 flex flex-col gap-6">

                    {/* Bento Box 1: Topology Stats */}
                    <div className="rounded-xl bg-zinc-900 border border-white/5 p-5 shadow-xl shadow-black/40">
                        <div className="flex items-center gap-2 mb-4 text-zinc-400">
                            <Database size={16} />
                            <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-zinc-500">Graph Topology</h3>
                        </div>

                        <div className="grid grid-cols-2 gap-3 mb-5">
                            <div className="bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded-lg p-3 flex flex-col items-center justify-center transition-all hover:bg-indigo-500/20 hover:border-indigo-500/40 cursor-default">
                                <span className="text-[10px] uppercase font-bold tracking-[0.1em] mb-1 opacity-70">Avg Degree</span>
                                <span className="text-2xl font-bold">14.2</span>
                            </div>
                            <div className="bg-teal-500/10 text-teal-300 border border-teal-500/20 rounded-lg p-3 flex flex-col items-center justify-center transition-all hover:bg-teal-500/20 hover:border-teal-500/40 cursor-default">
                                <span className="text-[10px] uppercase font-bold tracking-[0.1em] mb-1 opacity-70">Max Degree</span>
                                <span className="text-2xl font-bold">58</span>
                            </div>
                        </div>

                        <div className="h-36 bg-zinc-950 rounded-lg border border-zinc-800 flex flex-col items-center justify-center text-zinc-600 text-sm shadow-inner overflow-hidden p-4 relative">
                            {/* Recharts / Chart.js Degree Distribution Goes Here */}
                            <span className="opacity-50 font-mono text-xs z-10">[ Recharts: Degree Dist. ]</span>
                            {/* Fake faint lines representing bar chart */}
                            <div className="absolute bottom-0 left-0 w-full h-24 flex items-end gap-1 px-4 opacity-20">
                                {[4, 8, 12, 24, 32, 18, 10, 6, 4, 2, 1].map((h, i) => (
                                    <div key={i} className="flex-1 bg-indigo-500 rounded-t-sm" style={{ height: `${h * 3}px` }}></div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Bento Box 2: Memory Formation Charts */}
                    <div className="rounded-xl bg-zinc-900 border border-white/5 p-5 shadow-xl shadow-black/40 flex-1 flex flex-col">
                        <div className="flex items-center gap-2 mb-4 text-zinc-400">
                            <Activity size={16} />
                            <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-zinc-500">Layer Activations</h3>
                        </div>

                        <div className="flex-1 bg-zinc-950 rounded-lg border border-zinc-800 p-4 flex flex-col items-center justify-center text-zinc-600 text-sm min-h-[180px] shadow-inner relative overflow-hidden">
                            {/* Primary Data: indigo-400, Secondary: teal-400, Threshold: rose-500/50 */}
                            {/* Faint Gridlines: stroke-white/5 */}
                            {/* Recharts / Chart.js Sparse Brain Chart Goes Here */}
                            <span className="opacity-50 font-mono text-xs z-10">[ Recharts: Mean Mag. ]</span>

                            {/* Fake faint Sine wave overlay */}
                            <svg className="absolute inset-0 w-full h-full opacity-30" preserveAspectRatio="none">
                                <path d="M0,100 C50,20 100,20 150,100 S250,180 300,100" stroke="#38b2ac" fill="none" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        </div>

                        <div className="flex gap-3 mt-5">
                            <div className="flex-1 shrink-0 bg-zinc-950 border border-zinc-800 rounded-lg p-3 group hover:border-indigo-500/30 transition-colors">
                                <span className="block text-[10px] text-zinc-500 uppercase font-bold tracking-[0.1em] mb-1">X Avg Active</span>
                                <span className="text-lg font-bold text-zinc-200 group-hover:text-indigo-400 transition-colors">97.78%</span>
                            </div>
                            <div className="flex-1 shrink-0 bg-zinc-950 border border-zinc-800 rounded-lg p-3 group hover:border-teal-500/30 transition-colors">
                                <span className="block text-[10px] text-zinc-500 uppercase font-bold tracking-[0.1em] mb-1">Y Avg Active</span>
                                <span className="text-lg font-bold text-zinc-200 group-hover:text-teal-400 transition-colors">72.08%</span>
                            </div>
                        </div>
                    </div>

                </aside>
            </div>
        </div>
    );
}
