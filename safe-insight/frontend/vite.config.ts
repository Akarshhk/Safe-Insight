import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite config tuned for Tauri.
 *
 * - Port 1420 is fixed and `strictPort` is on because `tauri.conf.json`'s
 *   `devUrl` points at exactly that address; a silent port bump would leave the
 *   desktop window staring at a blank page.
 * - `host: false` keeps the dev server on loopback. The whole product promise is
 *   that nothing is reachable from the network, and that should hold in dev too.
 */
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: false,
    watch: {
      // Rust sources are rebuilt by Tauri, not by Vite.
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    // Matches the Tauri 2 minimum webview targets (WebView2 / WKWebView).
    target: "es2021",
    outDir: "dist",
    sourcemap: false,
  },
});
