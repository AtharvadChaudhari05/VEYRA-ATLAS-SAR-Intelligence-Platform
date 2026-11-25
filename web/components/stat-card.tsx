import { ReactNode } from "react";
export function StatCard({label,value,sub}:{label:string;value:ReactNode;sub?:string}){return <div className="panel shine p-4"><p className="label">{label}</p><p className="mt-2 bg-gradient-to-r from-cyan to-violet-600 bg-clip-text text-2xl font-semibold text-transparent">{value}</p>{sub&&<p className="mt-1 text-xs text-slate-500">{sub}</p>}</div>}
