"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { api } from "@/lib/api";
import { downloadLayer } from "@/components/layer-viewer";
import { Copy, Download, Image as ImageIcon, MapPinned } from "lucide-react";
import { toast } from "sonner";

type Props = { boundary?: string; bounds?: number[] | null; crs?: string | null; layers?: Record<string, string> };
const names = ["optical_like", "probability", "segmentation_mask"] as const;
const satelliteStyle: maplibregl.StyleSpecification = { version: 8, sources: { imagery: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, attribution: "Tiles © Esri" } }, layers: [{ id: "world-imagery", type: "raster", source: "imagery" }] };

function isWgs84(bounds?: number[] | null, crs?: string | null) {
  if (!bounds || bounds.length !== 4 || !crs) return false;
  const [west, south, east, north] = bounds;
  return /4326|crs84|wgs\s*84/i.test(crs) && west >= -180 && east <= 180 && south >= -90 && north <= 90 && west < east && south < north;
}

export function AnalysisMap({ boundary, bounds, crs, layers = {} }: Props) { return isWgs84(bounds, crs) ? <GeoMap boundary={boundary} bounds={bounds!} layers={layers}/> : <SceneCanvas layers={layers}/>; }

function GeoMap({ boundary, bounds, layers }: { boundary?: string; bounds: number[]; layers?: Record<string, string> }) {
  const el = useRef<HTMLDivElement>(null), ref = useRef<maplibregl.Map | null>(null);
  const [on, setOn] = useState<Record<string, boolean>>({ optical_like: true, probability: false, segmentation_mask: false, boundary: true });
  useEffect(() => { if (!el.current) return; const [west, south, east, north] = bounds; const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [[west, north], [east, north], [east, south], [west, south]]; const map = new maplibregl.Map({ container: el.current, style: satelliteStyle, center: [(west + east) / 2, (south + north) / 2], zoom: 8 }); ref.current = map; map.addControl(new maplibregl.NavigationControl(), "top-right"); map.on("load", async () => { map.fitBounds([[west, south], [east, north]], { padding: 42, maxZoom: 14, duration: 0 }); names.forEach(name => { if (!layers?.[name]) return; map.addSource(name, { type: "image", url: api.layerUrl(layers[name]), coordinates }); map.addLayer({ id: name, type: "raster", source: name, layout: { visibility: name === "optical_like" ? "visible" : "none" }, paint: { "raster-opacity": name === "optical_like" ? 0.68 : 0.56 } }); }); if (!boundary) return; try { const data = await fetch(api.layerUrl(boundary)).then(response => response.json()); map.addSource("flood", { type: "geojson", data }); map.addLayer({ id: "flood-fill", type: "fill", source: "flood", paint: { "fill-color": "#e75953", "fill-opacity": 0.22 } }); map.addLayer({ id: "flood-line", type: "line", source: "flood", paint: { "line-color": "#e75953", "line-width": 2.5 } }); } catch { toast.error("Affected boundary could not be loaded"); } }); return () => map.remove(); }, [boundary, bounds, layers]);
  return <div className="relative h-[480px] overflow-hidden rounded-2xl border border-sky-100"><div ref={el} className="h-full w-full"/><MapControls on={on} setOn={setOn} map={ref.current} boundary={boundary} geographic/></div>;
}

function SceneCanvas({ layers = {} }: Pick<Props, "layers">) {
  const [on, setOn] = useState<Record<string, boolean>>({ optical_like: true, probability: false, segmentation_mask: false, boundary: true });
  return <div className="relative h-[480px] overflow-hidden rounded-2xl border border-sky-100 bg-slate-950">{layers.optical_like ? <img className="absolute inset-0 h-full w-full object-contain" src={api.layerUrl(layers.optical_like)} alt="Optical-like scene"/> : <div className="grid h-full place-items-center text-sm text-slate-400">Scene output will appear here.</div>}{on.probability && layers.probability && <img className="absolute inset-0 h-full w-full object-contain opacity-60 mix-blend-screen" src={api.layerUrl(layers.probability)} alt="Flood probability overlay"/>}{on.segmentation_mask && layers.segmentation_mask && <img className="absolute inset-0 h-full w-full object-contain opacity-60 mix-blend-screen" src={api.layerUrl(layers.segmentation_mask)} alt="Flood mask overlay"/>}<div className="absolute right-3 top-3 z-10 inline-flex items-center gap-2 rounded-full bg-slate-950/80 px-3 py-1.5 text-xs font-semibold text-white"><ImageIcon size={14}/> Scene analysis canvas</div><div className="absolute bottom-3 right-3 z-10 max-w-[270px] rounded-xl bg-white/92 p-3 text-xs leading-5 text-slate-600 shadow-lg"><b className="text-ink">No location metadata</b><br/>Layers are aligned to your uploaded image, not placed on an invented world location.</div><MapControls on={on} setOn={setOn} geographic={false}/></div>;
}

function MapControls({ on, setOn, map, boundary, geographic }: { on: Record<string, boolean>; setOn: React.Dispatch<React.SetStateAction<Record<string, boolean>>>; map?: maplibregl.Map | null; boundary?: string; geographic: boolean }) {
  function toggle(key: string) { const next = !on[key]; setOn(previous => ({ ...previous, [key]: next })); if (!map) return; if (key === "boundary") ["flood-fill", "flood-line"].forEach(id => map.getLayer(id) && map.setLayoutProperty(id, "visibility", next ? "visible" : "none")); else if (map.getLayer(key)) map.setLayoutProperty(key, "visibility", next ? "visible" : "none"); }
  async function copy() { try { if (!boundary) return; await navigator.clipboard.writeText(await fetch(api.layerUrl(boundary)).then(response => response.text())); toast.success("Boundary GeoJSON copied to clipboard"); } catch { toast.error("Could not copy GeoJSON"); } }
  return <div className="absolute left-3 top-3 z-10 rounded-xl border border-white/20 bg-slate-950/90 p-3 text-xs text-white shadow-lg"><p className="mb-2 flex items-center gap-1.5 font-semibold text-cyan"><MapPinned size={14}/>{geographic ? "Real-world layers" : "Image layers"}</p>{names.map(key => <label className="mb-1 flex cursor-pointer gap-2 capitalize" key={key}><input className="accent-cyan" type="checkbox" checked={on[key]} onChange={() => toggle(key)}/>{key.replaceAll("_", " ")}</label>)}{geographic && <label className="flex cursor-pointer gap-2"><input className="accent-flood" type="checkbox" checked={on.boundary} onChange={() => toggle("boundary")}/>affected boundary</label>}{boundary && geographic && <div className="mt-2 flex gap-3 border-t border-white/10 pt-2"><button title="Copy GeoJSON" onClick={copy}><Copy size={14}/></button><button title="Download boundary" onClick={() => downloadLayer(boundary, "affected_boundary")}><Download size={14}/></button></div>}</div>;
}
