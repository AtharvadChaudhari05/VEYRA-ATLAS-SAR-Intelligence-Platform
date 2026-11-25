import type { Config } from "tailwindcss";
export default { content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"], theme: { extend: { colors: { ink: "#10243e", panel: "#ffffff", cyan: "#146c94", lime: "#f4a543", flood: "#e75953" }, boxShadow: { glow: "0 14px 34px rgb(98 70 217 / .18)" } } }, plugins: [] } satisfies Config;
