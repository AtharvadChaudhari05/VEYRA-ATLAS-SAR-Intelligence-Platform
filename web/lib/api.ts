import { z } from "zod";

const baseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const layerSchema = z.record(z.string());
export const healthSchema = z.object({ status:z.string(), app_name:z.string(), version:z.string(), models:z.object({ sar_to_optical:z.string(), disaster_segmentation:z.string(), supported_disasters:z.array(z.string()) }) });
export const sceneSchema = z.object({ id:z.string(), filename:z.string(), width:z.number(), height:z.number(), crs:z.string().nullable(), transform:z.array(z.number()).nullable(), bounds:z.array(z.number()).nullable(), metadata:z.record(z.unknown()), created_at:z.string() });
export const predictionSchema = z.object({ scene_id:z.string(), prediction_id:z.string(), job_id:z.string(), model_version:z.string(), confidence:z.number(), metadata:z.record(z.unknown()), layers:layerSchema, created_at:z.string() });
// /analyze deliberately has a different response contract from /predict: it
// returns analysis fields and layer paths, but not model_version or metadata.
export const analysisSchema = z.object({ scene_id:z.string(), prediction_id:z.string(), analysis_id:z.string(), job_id:z.string(), disaster_type:z.string(), threshold:z.number(), affected_area_km2:z.number().nullable(), coverage_percent:z.number(), confidence:z.number(), area_method:z.string(), area_warning:z.string().nullable(), layers:layerSchema, boundary_geojson_path:z.string(), created_at:z.string() });
export const jobSchema = z.object({ id:z.string(), kind:z.string(), status:z.string(), message:z.string(), result_id:z.string().nullable(), created_at:z.string(), updated_at:z.string() });
export const modelsSchema = z.object({ model_1:z.record(z.unknown()), model_2:z.record(z.unknown()) });
export const datasetSchema = z.object({ root_path:z.string(), sar_files:z.number(), optical_files:z.number(), pair_count:z.number(), valid_pair_count:z.number(), invalid_pair_count:z.number(), missing_optical:z.array(z.string()), orphan_optical:z.array(z.string()), pairs:z.array(z.record(z.unknown())), manifests:z.record(z.unknown()) });
export type Scene = z.infer<typeof sceneSchema>; export type Prediction = z.infer<typeof predictionSchema>; export type Analysis = z.infer<typeof analysisSchema>; export type Job = z.infer<typeof jobSchema>;

async function request<T>(path:string, schema:z.ZodType<T>, init?:RequestInit): Promise<T> { const res=await fetch(`${baseUrl}${path}`,init); if(!res.ok){ const body=await res.json().catch(()=>null); throw new Error(body?.detail || `Request failed (${res.status})`); } return schema.parse(await res.json()); }
function form(file:File, values: Record<string,string|number|undefined>) { const data=new FormData(); data.append("file",file); Object.entries(values).forEach(([k,v])=>v!==undefined && v!=="" && data.append(k,String(v))); return data; }
export const api = { baseUrl, layerUrl:(path:string)=>`${baseUrl}${path}`, health:()=>request("/health",healthSchema), models:()=>request("/models/status",modelsSchema), scenes:()=>request("/metadata",z.array(sceneSchema)), scene:(id:string)=>request(`/metadata/${id}`,sceneSchema), job:(id:string)=>request(`/jobs/${id}`,jobSchema), predict:(file:File, geo:Geo={})=>request("/predict",predictionSchema,{method:"POST",body:form(file,geo)}), analyze:(file:File, values:Geo & {threshold:number,disaster_type:string})=>request("/analyze",analysisSchema,{method:"POST",body:form(file,values)}), scan:(root_path:string)=>request("/datasets/scan",datasetSchema,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({root_path})}) };
export type Geo={crs?:string;transform?:string;bounds?:string};
