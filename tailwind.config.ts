import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        lab: {
          bg: "#0B0F12",
          panel: "#12181D",
          panel2: "#171F25",
          border: "#232D34",
          ink: "#E7EEF2",
          dim: "#8A9AA5",
        },
        base: {
          A: "#4ADE80",
          C: "#38BDF8",
          G: "#FBBF24",
          T: "#FB7185",
        },
        accent: {
          DEFAULT: "#2DD4BF",
          dim: "#0F766E",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
