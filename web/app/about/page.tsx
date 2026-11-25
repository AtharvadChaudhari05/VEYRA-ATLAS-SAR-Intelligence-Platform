"use client";

import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

const models = [
  { title: "SAR representation", number: "Model 01", image: "/images/sar-river-scene.png", alt: "False-color SAR river scene", body: "Model 1 translates difficult-to-read SAR backscatter into an optical-like RGB interpretation. It also generates uncertainty and SAR-preservation layers, so users can compare the visualization against the original signal instead of treating it as a real optical photograph.", points: ["Works with cloud-penetrating radar input", "Produces optical-like, uncertainty, and preservation layers", "Designed for clear interpretation with visible confidence limits"] },
  { title: "Flood intelligence", number: "Model 02", image: "/images/satellite-earth.png", alt: "Satellite observing Earth", body: "Model 2 turns a SAR scene into a flood-probability field, then applies the selected threshold to create a segmentation mask. With valid GeoTIFF coordinates, it also provides a boundary, coverage, and affected-area estimate for response planning.", points: ["Measures flood likelihood for every pixel", "Creates probability, mask, boundary, and area outputs", "Uses a guarded deployment gate before a trained ONNX model becomes live"] },
];

export default function About() {
  const query = useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: 60000 });
  const [open, setOpen] = useState<number | null>(null);
  const record = (query.data?.model_1 || {}) as Record<string, unknown>;
  const model2 = (query.data?.model_2 || {}) as Record<string, unknown>;
  const list = (value: unknown) => Array.isArray(value) ? value.join(" · ") : "—";

  return <main className="mx-auto max-w-6xl px-5 py-12"><p className="label">Methods & models</p><h1 className="mt-2 text-4xl font-semibold">A legible layer over radar data.</h1><p className="mt-3 max-w-2xl leading-7 text-slate-600">Select a model to see the image, method, and outputs behind each stage of the Veyra Atlas workflow.</p><div className="mt-8 grid gap-4 md:grid-cols-2">{models.map((model, index) => <ModelCard key={model.title} model={model} isOpen={open === index} onToggle={() => setOpen(open === index ? null : index)} />)}<article className="panel p-6"><h2 className="text-lg font-semibold text-cyan">Why SAR?</h2><p className="mt-3 leading-7 text-slate-600">Synthetic Aperture Radar actively sends microwave energy toward Earth and measures the return signal. Unlike an optical camera, it works day and night and can observe through cloud cover. That makes SAR especially valuable during floods, storms, and other events when clear optical imagery is unavailable.</p><p className="mt-4 rounded-xl bg-sky-50 px-4 py-3 text-sm font-semibold text-ocean">SAR is a weather-resilient source of evidence, not simply another photograph.</p></article><article className="panel p-6"><h2 className="text-lg font-semibold text-cyan">Runtime status</h2><p className="mono mt-3 whitespace-pre-wrap text-xs leading-7 text-slate-500">{query.isLoading ? "Loading model metadata…" : query.isError ? `Unavailable: ${query.error.message}` : `${record.name || "Model 01"}\n${record.version || "—"} · ${record.runtime || "—"}\nStatus: ${record.status || "—"}\nOutputs: ${list(record.outputs)}\n\n${model2.name || "Model 02"}\n${model2.version || "—"} · ${model2.runtime || "—"}\nStatus: ${model2.status || "—"}\nDisasters: ${list(model2.supported_disasters)}\n${model2.activation_rule || ""}`}</p></article></div></main>;
}

function ModelCard({ model, isOpen, onToggle }: { model: typeof models[number]; isOpen: boolean; onToggle: () => void }) {
  return <article className="panel overflow-hidden"><button type="button" onClick={onToggle} aria-expanded={isOpen} className="flex w-full items-center justify-between p-6 text-left transition hover:bg-sky-50/70"><span><span className="text-xs font-bold uppercase tracking-[.16em] text-violet-600">{model.number}</span><h2 className="mt-1 text-xl font-semibold text-cyan">{model.title}</h2></span><motion.span animate={{ rotate: isOpen ? 180 : 0 }} className="grid h-9 w-9 place-items-center rounded-full bg-cyan/10 text-cyan"><ChevronDown size={19} /></motion.span></button><AnimatePresence initial={false}>{isOpen && <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: .32 }} className="overflow-hidden"><div className="border-t border-sky-100 px-6 pb-6"><div className="relative mt-5 h-44 overflow-hidden rounded-2xl"><Image src={model.image} alt={model.alt} fill sizes="(max-width: 768px) 100vw, 50vw" className="object-cover" /></div><p className="mt-5 leading-7 text-slate-600">{model.body}</p><ul className="mt-4 grid gap-2 text-sm leading-6 text-slate-600">{model.points.map(point => <li key={point} className="flex gap-2"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-gradient-to-br from-cyan to-violet-600" />{point}</li>)}</ul></div></motion.div>}</AnimatePresence></article>;
}
