import Link from "next/link";
import Image from "next/image";

export function VeyraLogo(){return <Link href="/" className="group flex items-center gap-2.5 font-bold tracking-tight text-ink" aria-label="Veyra Atlas home"><span className="grid h-10 w-10 place-items-center overflow-hidden rounded-xl bg-slate-950 shadow-[0_6px_15px_rgba(20,108,148,.22)] transition group-hover:scale-105"><Image src="/images/veyra-orbit-logo.png" alt="" width={96} height={96} className="h-12 w-12 max-w-none object-contain"/></span><span className="leading-none">VEYRA <em className="not-italic text-cyan">ATLAS</em><small className="mt-1 block text-[8px] font-semibold tracking-[.18em] text-slate-500">EARTH SIGNAL STUDIO</small></span></Link>}
