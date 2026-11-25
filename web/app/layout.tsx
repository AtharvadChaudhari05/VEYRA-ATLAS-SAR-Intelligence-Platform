import "./globals.css"; import "maplibre-gl/dist/maplibre-gl.css"; import Providers from "./providers"; import { Nav } from "@/components/nav";
export const metadata={title:"Veyra Atlas | Disaster Intelligence",description:"SAR colorization and flood intelligence"};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body><Providers><Nav/>{children}</Providers></body></html>}
