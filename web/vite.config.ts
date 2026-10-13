import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Relative assets work both at Vite's / and when the build is mounted at /console.
  base: "./",
  // Keep builds usable when the mounted dist directory is being watched by the engine on Windows.
  build: { emptyOutDir: false },
});
